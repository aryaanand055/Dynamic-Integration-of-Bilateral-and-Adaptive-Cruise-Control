"""
Run the trained TD3 model to control traffic.
Make sure td3_traffic_final.zip is in this folder.
"""

import numpy as np

# Check if stable-baselines3 is installed
try:
    from stable_baselines3 import TD3
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Run: pip install stable-baselines3")
    exit(1)

# Check if gymnasium is installed  
try:
    import gymnasium as gym
except ImportError:
    print("ERROR: gymnasium not installed.")
    print("Run: pip install gymnasium")
    exit(1)

from traffic_env import TrafficControlEnv

def run_demo():
    print("Loading trained TD3 model...")
    
    try:
        model = TD3.load("td3_traffic_final")
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("ERROR: td3_traffic_final.zip not found!")
        print("Make sure the file is in:", __file__)
        return
    
    # Create environment
    env = TrafficControlEnv(num_cars=6)
    
    print("\n" + "="*60)
    print("Running trained agent for 500 steps...")
    print("="*60 + "\n")
    
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(500):
        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            print(f"Step {step+1:3d}: Min Gap = {info['min_gap']:6.2f}m, "
                  f"Avg Speed = {info['avg_speed']:5.2f} m/s, "
                  f"Weights = [{', '.join(f'{w:.2f}' for w in action)}]")
        
        if terminated:
            print(f"\nCollision at step {step+1}!")
            break
    
    print("\n" + "="*60)
    print(f"Demo complete!")
    print(f"  Total steps: {step + 1}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Final min gap: {info['min_gap']:.2f}m")
    print(f"  Status: {'COLLISION' if terminated else 'SUCCESS'}")
    print("="*60)
    
    # Print the learned weights
    print(f"\nLearned weights (w1-w5):")
    print(f"  w1 (front gap):     {action[0]:.3f}")
    print(f"  w2 (back gap):      {action[1]:.3f}")
    print(f"  w3 (front vel):     {action[2]:.3f}")
    print(f"  w4 (back vel):      {action[3]:.3f}")
    print(f"  w5 (target vel):    {action[4]:.3f}")

if __name__ == "__main__":
    run_demo()
