import os
import sys
import time
import json
from pathlib import Path
import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.accel_traffic_env import AccelTrafficEnv
from simulation.config import MODELS_DIR, TENSORBOARD_DIR, LOGS_DIR, resolve_model_path

class TrainingProgressCallback(BaseCallback):
    """Callback to periodically log training telemetry and dump progress updates."""
    def __init__(self, check_freq: int = 500, log_path=LOGS_DIR / "training_progress.json", verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_path = Path(log_path)
        self.start_time = None
        self.history = []

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.history = []
        try:
            with open(self.log_path, "w") as f:
                json.dump([], f)
        except Exception:
            pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - (self.start_time or time.time())
            fps = int(self.n_calls / max(elapsed, 1e-3))
            ep_rew_mean = None
            if len(self.model.ep_info_buffer) > 0:
                ep_rew_mean = float(np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
            
            total_ts = getattr(self.model, "_total_timesteps", 600000)
            progress_pct = round((self.num_timesteps / max(total_ts, 1)) * 100, 1)

            record = {
                "step": int(self.num_timesteps),
                "total_timesteps": int(total_ts),
                "progress_pct": progress_pct,
                "elapsed_sec": round(elapsed, 1),
                "fps": fps,
                "ep_rew_mean": round(ep_rew_mean, 2) if ep_rew_mean is not None else None,
            }
            self.history.append(record)
            try:
                with open(self.log_path, "w") as f:
                    json.dump(self.history, f, indent=2)
            except Exception:
                pass
            print(f"[PROGRESS] Step {self.num_timesteps}/{total_ts} ({progress_pct}%) | FPS: {fps} | Mean Reward: {record['ep_rew_mean']} | Elapsed: {record['elapsed_sec']}s", flush=True)
        return True

class ActionNoiseDecayCallback(BaseCallback):
    """Linearly decays action noise sigma over training progress to ensure smooth convergence."""
    def __init__(self, initial_sigma: float = 0.30, final_sigma: float = 0.03, total_timesteps: int = 500000, verbose: int = 0):
        super().__init__(verbose)
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(self.total_timesteps, 1))
        current_sigma = self.initial_sigma + frac * (self.final_sigma - self.initial_sigma)
        if hasattr(self.model, "action_noise") and self.model.action_noise is not None:
            act_noise = self.model.action_noise
            if hasattr(act_noise, "noises"):
                for n in act_noise.noises:
                    if hasattr(n, "_sigma"):
                        n._sigma = current_sigma * np.ones_like(n._sigma)
            if hasattr(act_noise, "base_noise") and hasattr(act_noise.base_noise, "_sigma"):
                act_noise.base_noise._sigma = current_sigma * np.ones_like(act_noise.base_noise._sigma)
            if hasattr(act_noise, "_sigma"):
                act_noise._sigma = current_sigma * np.ones_like(act_noise._sigma)
        return True

def get_next_run_id(log_dir=TENSORBOARD_DIR, models_dir=MODELS_DIR, prefix="TD3"):
    """Auto-detect the next run number based on existing tensorboard logs and saved models."""
    existing = []
    log_path = Path(log_dir)
    if log_path.exists():
        for item in log_path.iterdir():
            if item.is_dir() and item.name.startswith(f"{prefix}_"):
                try:
                    num = int(item.name.split("_")[1])
                    existing.append(num)
                except (ValueError, IndexError):
                    pass
    models_path = Path(models_dir)
    if models_path.exists():
        for item in models_path.iterdir():
            if item.name.startswith(f"{prefix}_"):
                try:
                    stem = item.stem.split(".")[0]
                    num = int(stem.split("_")[1])
                    existing.append(num)
                except (ValueError, IndexError):
                    pass
    next_num = max(existing, default=0) + 1
    return f"{prefix}_{next_num}"

def calibrate_zero_drift(model, obs_dim=6):
    """
    Analytically calibrates the actor's output bias so that a(0) == 0.00000000 m/s^2.
    Ensures zero acceleration drift at exact equilibrium.
    """
    zeros_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device=model.device)
    with torch.no_grad():
        x = zeros_obs
        for i in range(len(model.actor.mu) - 2):
            x = model.actor.mu[i](x)
        last_linear_idx = len(model.actor.mu) - 2
        pre_bias = torch.matmul(x, model.actor.mu[last_linear_idx].weight.t())
        model.actor.mu[last_linear_idx].bias.copy_(-pre_bias.squeeze())
        calib_act = model.actor(zeros_obs).item()
        print(f"Equilibrium analytical calibration verified: a(0) = {calib_act:+.8f} m/s^2")
    return model

def train(fine_tune: str = "", timesteps: int = 350000, lr: float = 0.0002, gradient_steps: int = 2, run_id: str = ""):
    """
    Main decentralized Multi-Agent TD3 policy training loop.
    Optimizes direct continuous longitudinal acceleration commands.
    """
    if not run_id:
        run_id = get_next_run_id()
    print(f"==================================================")
    print(f" Launching TD3 Multi-Agent Policy Training: {run_id}")
    print(f" Target Steps: {timesteps} | LR: {lr} | Gradient Steps: {gradient_steps}")
    print(f"==================================================")

    # Instantiate decentralized VecEnv
    raw_env = AccelTrafficEnv()

    fine_tune_resolved = resolve_model_path(fine_tune) if fine_tune else ""
    vecnorm_pkl = fine_tune_resolved.replace(".zip", "_vecnormalize.pkl") if fine_tune_resolved else ""

    if fine_tune_resolved and os.path.exists(vecnorm_pkl):
        print(f"Loading VecNormalize statistics from {vecnorm_pkl}")
        env = VecNormalize.load(vecnorm_pkl, raw_env)
        env.training = True
        env.norm_obs = False
        env.norm_reward = True
        env.clip_reward = 35.0
    else:
        env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=35.0)

    env = VecMonitor(env)

    # Action noise: initial sigma with smooth linear decay to 0.02
    n_actions = env.action_space.shape[-1]
    noise_sigma = 0.12 if fine_tune_resolved else 0.25
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_sigma * np.ones(n_actions)
    )

    tb_log_dir = str(TENSORBOARD_DIR)

    if fine_tune_resolved and os.path.exists(fine_tune_resolved):
        print(f"Fine-tuning from base model: {fine_tune_resolved}")
        model = TD3.load(
            fine_tune_resolved,
            env=env,
            learning_rate=lr,
            action_noise=action_noise,
            tensorboard_log=tb_log_dir
        )
        model.gradient_steps = gradient_steps
        model.learning_starts = 2500
    else:
        print("Training fresh TD3 policy from scratch...")
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            buffer_size=300000,
            learning_starts=3000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[400, 300]),
            tensorboard_log=tb_log_dir
        )

    # Checkpoint setup (save every ~50,000 timesteps across multi-agent VecEnv)
    checkpoints_dir = MODELS_DIR / "checkpoints" / run_id
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    num_envs = getattr(raw_env, 'num_rl_cars', 14)
    ckpt_save_freq = max(1, 50000 // num_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_save_freq,
        save_path=str(checkpoints_dir),
        name_prefix=run_id
    )
    progress_callback = TrainingProgressCallback(check_freq=500)
    noise_decay_callback = ActionNoiseDecayCallback(
        initial_sigma=noise_sigma,
        final_sigma=0.02,
        total_timesteps=int(timesteps * 0.85)
    )
    callback = CallbackList([checkpoint_callback, progress_callback, noise_decay_callback])

    print(f"Training policy across {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        log_interval=5,
        tb_log_name=run_id,
        callback=callback
    )
    print("Training finished.")

    # Analytical Zero-Drift Calibration
    obs_dim = env.observation_space.shape[0]
    calibrate_zero_drift(model, obs_dim=obs_dim)

    # Save versioned model to models/
    versioned_path = MODELS_DIR / run_id
    model.save(str(versioned_path))
    env.save(str(MODELS_DIR / f"{run_id}_vecnormalize.pkl"))
    print(f"Versioned model saved to: {versioned_path}.zip")

    # Update active model in models/ and PROJECT_ROOT
    model.save(str(MODELS_DIR / "td3_accel_agent"))
    env.save(str(MODELS_DIR / "td3_accel_agent_vecnormalize.pkl"))
    model.save(str(PROJECT_ROOT / "td3_accel_agent"))
    env.save(str(PROJECT_ROOT / "td3_accel_agent_vecnormalize.pkl"))
    print(f"Default active model updated in models/ and root as 'td3_accel_agent.zip'")

    # Quick evaluation
    print("\nRunning post-training quick rollout evaluation...")
    obs = env.reset()
    total_reward = 0.0
    for i in range(1000):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(actions)
        total_reward += float(np.sum(rewards))
        if any(dones):
            print(f"Rollout finished after {i + 1} steps.")
            break
    print(f"Total Post-Training Reward: {total_reward:.4f}")

def blend_model_soup(model_paths, weights, output_path):
    """
    Blends multiple trained TD3 policy checkpoints into a Polyak model soup.
    Performs analytical zero-drift calibration on the merged actor network.
    """
    assert len(model_paths) == len(weights), "Number of models must match number of weights"
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.sum(weights)

    print(f"Blending {len(model_paths)} checkpoints with weights {weights}...")
    base_model = TD3.load(resolve_model_path(model_paths[0]))
    base_state_dict = base_model.actor.state_dict()

    blended_state_dict = {k: v.clone() * weights[0] for k, v in base_state_dict.items()}

    for i in range(1, len(model_paths)):
        m = TD3.load(resolve_model_path(model_paths[i]))
        m_sd = m.actor.state_dict()
        for k in blended_state_dict:
            blended_state_dict[k] += m_sd[k] * weights[i]

    base_model.actor.load_state_dict(blended_state_dict)

    # Analytical zero-drift calibration on blended actor
    obs_dim = base_model.observation_space.shape[0]
    zeros_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device=base_model.device)
    with torch.no_grad():
        x = zeros_obs
        for i in range(len(base_model.actor.mu) - 2):
            x = base_model.actor.mu[i](x)
        last_linear_idx = len(base_model.actor.mu) - 2
        pre_bias = torch.matmul(x, base_model.actor.mu[last_linear_idx].weight.t())
        base_model.actor.mu[last_linear_idx].bias.copy_(-pre_bias.squeeze())
        calib_act = base_model.actor(zeros_obs).item()
        print(f"Blended model zero-drift calibrated: a(0) = {calib_act:+.8f} m/s^2")

    out_p = Path(output_path)
    base_model.save(str(out_p))
    print(f"Polyak model soup saved to: {out_p}")
    return base_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TD3 Multi-Agent Direct Acceleration Policy")
    parser.add_argument("--fine-tune", type=str, default="", help="Path to base model to fine-tune from")
    parser.add_argument("--timesteps", type=int, default=350000, help="Total training timesteps (default 350,000)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default 0.0002)")
    parser.add_argument("--gradient-steps", type=int, default=2, help="Gradient steps per env step")
    parser.add_argument("--run-id", type=str, default="", help="Custom run ID (e.g. TD3_59)")
    parser.add_argument("--soup-models", nargs="+", default=[], help="List of model checkpoint paths to blend")
    parser.add_argument("--soup-weights", nargs="+", type=float, default=[], help="Weights for Polyak soup blending")
    parser.add_argument("--soup-output", type=str, default="models/TD3_58.zip", help="Output path for blended model")
    args, _ = parser.parse_known_args()

    if args.soup_models:
        weights = args.soup_weights if args.soup_weights else [1.0 / len(args.soup_models)] * len(args.soup_models)
        blend_model_soup(args.soup_models, weights, args.soup_output)
    else:
        train(
            fine_tune=args.fine_tune,
            timesteps=args.timesteps,
            lr=args.lr,
            gradient_steps=args.gradient_steps,
            run_id=args.run_id
        )
