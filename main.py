"""
Main Unified Entrypoint for Dynamic ACC/BCC Integration & RL Platoon Simulation.

Usage:
    python main.py                # Launch interactive GUI simulation dashboard (default)
    python main.py --gui          # Launch interactive GUI simulation dashboard
    python main.py --eval         # Run all 7 benchmark evaluation scenarios
    python main.py --train        # Train TD3 direct acceleration policy
    python main.py --headless     # Run headless multi-model comparison simulation
    python main.py --energy       # Run comparative energy consumption analysis
    python main.py --analytics    # Run policy rollout analytics
    python main.py --test         # Run platoon stop-and-resume verification tests
    python main.py --rewards      # Generate 24-component reward breakdown log & telemetry
    python main.py --intuitiveness # Run driver intuitiveness & IDM reference evaluation suite
"""
import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def launch_gui():
    """Launches the interactive GUI control window."""
    import tkinter as tk
    from gui.rl_control_window import RLControlWindow

    print("Launching RL-Enhanced Traffic Simulation Dashboard...")
    root = tk.Tk()
    root.title("RL-Enhanced Traffic Simulation")
    app = RLControlWindow(root)
    root.mainloop()

def run_eval(model_path="td3_accel_agent.zip"):
    """Runs the 7 benchmark scenarios evaluation suite."""
    from evaluation.evaluate_all_scenarios import evaluate_model_on_scenarios
    evaluate_model_on_scenarios(model_path)

def run_train(fine_tune="", timesteps=150000, lr=0.0003):
    """Runs TD3 model training."""
    from training.train_td3_accel import train
    train(fine_tune=fine_tune, timesteps=timesteps, lr=lr)

def run_headless_sim(model_path="td3_accel_agent.zip", duration=60.0):
    """Runs headless multi-model simulation comparison."""
    from evaluation.run_headless import run_headless
    run_headless(model_path=model_path, duration=duration)

def run_energy_sim(model_path="td3_accel_agent.zip"):
    """Runs energy consumption analysis."""
    from evaluation.get_energy import calculate_energy
    calculate_energy(model_path=model_path)

def run_analytics_sim(model_path="td3_accel_agent.zip"):
    """Runs rollout analytics."""
    from evaluation.evaluate_analytics import evaluate
    evaluate(model_path=model_path)

def run_test_suite(model_path="td3_accel_agent.zip"):
    """Runs automated verification tests."""
    print("==================================================")
    print(" Running Platoon Verification Test Suite")
    print("==================================================")
    from tests.test_driver_intuitiveness import run_intuitiveness_tests
    from tests.test_stop_and_resume import test_stop_and_resume
    from tests.test_diagnostics import run_diagnostics

    print("\n[Test 1/4] Driver Intuitiveness & IDM Reference Verification...")
    run_intuitiveness_tests(model_path=model_path)

    print("\n[Test 2/4] 15-Car Stop and Resume Verification...")
    test_stop_and_resume(model_path=model_path)

    print("\n[Test 3/4] Diagnostic Scenario Rollouts...")
    run_diagnostics(model_path=model_path)

    print("\n[Test 4/4] GUI Step Loop (60s Headless) & Metric Label Audit...")
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_gui_intuitiveness')
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    if not res.wasSuccessful():
        raise RuntimeError("GUI Intuitiveness verification tests failed!")
    print("\n>>> All verification tests passed successfully!")

def run_reward_logger(model_path="td3_accel_agent.zip"):
    """Generates reward breakdown log and telemetry CSV."""
    from tools.log_reward_breakdown import generate_reward_breakdown_log
    generate_reward_breakdown_log(model_path=model_path)

def run_intuitiveness(model_path="td3_accel_agent.zip"):
    """Runs Driver Intuitiveness and IDM human reference evaluation."""
    from evaluation.driver_intuitiveness import run_driver_intuitiveness_suite
    run_driver_intuitiveness_suite(model_path=model_path)

def main():
    parser = argparse.ArgumentParser(
        description="Dynamic ACC/BCC Integration & RL Direct Acceleration Control Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gui", action="store_true", help="Launch interactive GUI dashboard (default)")
    group.add_argument("--eval", action="store_true", help="Run 7-scenario benchmark evaluation suite")
    group.add_argument("--train", action="store_true", help="Run TD3 policy training")
    group.add_argument("--headless", action="store_true", help="Run headless 4-model comparison simulation")
    group.add_argument("--energy", action="store_true", help="Run energy consumption analysis")
    group.add_argument("--analytics", action="store_true", help="Run rollout analytics")
    group.add_argument("--test", action="store_true", help="Run verification test suite")
    group.add_argument("--rewards", action="store_true", help="Generate 24-component reward audit log & telemetry CSV")
    group.add_argument("--intuitiveness", nargs="?", const="td3_accel_agent.zip", default=None,
                       help="Run driver intuitiveness and IDM human reference evaluation (optional arg: model path)")

    parser.add_argument("--model", type=str, default="td3_accel_agent.zip", help="Path or name of target model")
    parser.add_argument("--timesteps", type=int, default=150000, help="Timesteps for training")
    parser.add_argument("--fine-tune", type=str, default="", help="Base model for fine-tuning")

    args = parser.parse_args()

    if args.intuitiveness is not None:
        target_model = args.intuitiveness if args.intuitiveness != "td3_accel_agent.zip" or args.model == "td3_accel_agent.zip" else args.model
        run_intuitiveness(model_path=target_model)
    elif args.eval:
        run_eval(model_path=args.model)
    elif args.train:
        run_train(fine_tune=args.fine_tune, timesteps=args.timesteps)
    elif args.headless:
        run_headless_sim(model_path=args.model)
    elif args.energy:
        run_energy_sim(model_path=args.model)
    elif args.analytics:
        run_analytics_sim(model_path=args.model)
    elif args.test:
        run_test_suite(model_path=args.model)
    elif args.rewards:
        run_reward_logger(model_path=args.model)
    else:
        # Default action is to launch GUI
        launch_gui()

if __name__ == "__main__":
    main()
