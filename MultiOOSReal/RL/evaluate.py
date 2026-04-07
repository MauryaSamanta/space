import torch
import numpy as np
import os

from rl_env import SatelliteEnv
from dqn import DQN


# -----------------------------
# INIT
# -----------------------------
env = SatelliteEnv()

state_dim = len(env.reset())
action_dim = 4

model = DQN(state_dim, action_dim)

model_path = os.path.join(os.path.dirname(__file__), "dqn_satellite.pth")
model.load_state_dict(torch.load(model_path))
model.eval()


# -----------------------------
# POLICIES
# -----------------------------
def policy(state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32)
        q = model(s)
        return torch.argmax(q).item()


def random_policy(state):
    return np.random.randint(action_dim)


# -----------------------------
# RUN EPISODE
# -----------------------------
def run_episode(policy_fn, max_steps=500, debug=False):
    state = env.reset()
    total_reward = 0

    for t in range(max_steps):
        action = policy_fn(state)
        next_state, reward, done = env.step(action)

        total_reward += reward

        if debug:
            print(f"Step {t} | Action: {action} | Reward: {reward:.3f}")

            if policy_fn == policy and t % 20 == 0:
                with torch.no_grad():
                    q = model(torch.tensor(state, dtype=torch.float32))
                    print("Q-values:", q.numpy())

        state = next_state

        if done:
            return total_reward, True

    return total_reward, False


# -----------------------------
# EVALUATION
# -----------------------------
episodes = 50

model_rewards = []
random_rewards = []

model_success = 0
random_success = 0


print("\n===== MODEL EVALUATION =====")

for ep in range(episodes):
    reward, success = run_episode(policy)

    model_rewards.append(reward)
    if success:
        model_success += 1

    print(f"[MODEL] Episode {ep} | Reward: {reward:.2f}")


print("\n===== RANDOM BASELINE =====")

for ep in range(episodes):
    reward, success = run_episode(random_policy)

    random_rewards.append(reward)
    if success:
        random_success += 1

    print(f"[RANDOM] Episode {ep} | Reward: {reward:.2f}")


# -----------------------------
# RESULTS
# -----------------------------
print("\n===== RESULTS =====")

print(f"Model Avg Reward   : {np.mean(model_rewards):.3f}")
print(f"Random Avg Reward  : {np.mean(random_rewards):.3f}")

print(f"Model Success Rate : {model_success / episodes:.2f}")
print(f"Random Success Rate: {random_success / episodes:.2f}")


# -----------------------------
# INTERPRETATION
# -----------------------------
print("\n===== INTERPRETATION =====")

if np.mean(model_rewards) > np.mean(random_rewards):
    print("✅ Model is better than random → LEARNING SUCCESS")
else:
    print("❌ Model not better than random → NOT LEARNING")

if model_success > random_success:
    print("✅ Model completes missions more often")
else:
    print("❌ Model not improving success rate")


# -----------------------------
# OPTIONAL: PLOT
# -----------------------------
try:
    import matplotlib.pyplot as plt

    plt.plot(model_rewards, label="Model")
    plt.plot(random_rewards, label="Random")
    plt.legend()
    plt.title("Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

except:
    print("matplotlib not installed → skipping plot")