import numpy as np
import torch
import torch.nn.functional as F

from rl_env import SatelliteEnv
from dqn import DQN, ReplayBuffer


# ---------------- INIT ----------------
env = SatelliteEnv()

state_dim = len(env.reset())
action_dim = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(state_dim, action_dim).to(device)
target_model = DQN(state_dim, action_dim).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
buffer = ReplayBuffer(capacity=50000)


# ---------------- HYPERPARAMS ----------------
gamma = 0.99
batch_size = 128

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999

tau = 0.01  # soft update


# ---------------- TRAIN LOOP ----------------
episodes = 99
rewards_history = []

def evaluate_model(env, model, episodes=10):
    total_rewards = []

    for _ in range(episodes):
        state = env.reset()
        ep_reward = 0

        for _ in range(1000):
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).to(device)
                action = torch.argmax(model(s)).item()

            state, reward, done = env.step(action)
            ep_reward += reward

            if done:
                break

        total_rewards.append(ep_reward)

    return np.mean(total_rewards)

for episode in range(episodes):

    state = env.reset()
    total_reward = 0

    for t in range(1000):

        # -------- ACTION --------
        if np.random.rand() < epsilon:
            action = np.random.randint(0, action_dim)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).to(device)
                q_values = model(s)
                action = torch.argmax(q_values).item()

        # -------- STEP --------
        next_state, reward, done = env.step(action)

        buffer.push((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        # -------- TRAIN --------
        if len(buffer) > batch_size:

            batch = buffer.sample(batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            # Q(s,a)
            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()

            # -------- DOUBLE DQN --------
            next_actions = model(next_states).argmax(1)
            next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

            target = rewards + gamma * next_q * (1 - dones)

            loss = F.mse_loss(q_values, target)

            optimizer.zero_grad()
            loss.backward()

            # -------- GRADIENT CLIPPING --------
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # -------- SOFT TARGET UPDATE --------
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if done:
            break

    # -------- EPSILON DECAY --------
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # -------- LOGGING --------
    rewards_history.append(total_reward)
    # -------- EVALUATE EVERY 100 EPISODES --------
    if episode % 100 == 0 and episode > 0:
        print("\n🔍 Evaluating model...")
        
        eval_score = evaluate_model(env, model, episodes=10)
        
        print(f"📊 Eval Avg Reward: {eval_score:.2f}\n")
    if episode > 20:
        avg_reward = np.mean(rewards_history[-20:])
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Avg20: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
    else:
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")


# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "dqn_satellite.pth")
print("✅ Model saved!")