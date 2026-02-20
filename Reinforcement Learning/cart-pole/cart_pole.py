import datetime
import time
import gymnasium as gym
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Define the discretization parameters for each observation
n_bins = [10, 10, 10, 10]  # Number of bins for each observation parameter
state_bins = [np.linspace(-4.8, 4.8, n_bins[0] - 1),  # Cart position
              np.linspace(-3.5, 3.5, n_bins[1] - 1),  # Cart velocity
              np.linspace(-0.418, 0.418, n_bins[2] - 1),  # Pole angle
              np.linspace(-3.5, 3.5, n_bins[3] - 1)]  # Pole angular velocity

# Initialize Q-table with zeros
Q = np.zeros(n_bins + [env.action_space.n])

# Define Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1  # Exploration rate
epsilon_reduce = 0.99995

# Q-learning algorithm
print("start time equal with         "+time.strftime("%H:%M:%S"))
for episode in range(1000):
    states = env.reset()
    state = states[0]
    done = False

    while not done:
        # Discretize the current state
        state_discrete = tuple(np.digitize(state[i], state_bins[i]) for i in range(4))

        # Choose action using ε-greedy strategy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Random action
        else:
            action = np.argmax(Q[state_discrete])

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, xx , rr= env.step(action)

        # Discretize the next state
        next_state_discrete = tuple(np.digitize(next_state[i], state_bins[i]) for i in range(4))

        # Update Q-value for the current state-action pair
        Q[state_discrete + (action,)] += alpha * (reward + gamma * np.max(Q[next_state_discrete]) - Q[state_discrete + (action,)])

        # Transition to the next state
        state = next_state
    print(f"step {episode} has been ended....")
    epsilon *= epsilon_reduce

    # After training, you can use the learned Q-values to select actions in the environment
env.close()


print("end time equal with         "+time.strftime("%H:%M:%S"))

# show agent
ss=gym.make('CartPole-v1',render_mode="human")
states=ss.reset()
state=states[0]
done = False
while not done:
    state_discrete = tuple(np.digitize(state[i], state_bins[i]) for i in range(4))
    action = np.argmax(Q[state_discrete])
    next_state, reward, done, xx, rr = ss.step(action)
    next_state_discrete = tuple(np.digitize(next_state[i], state_bins[i]) for i in range(4))
    state = next_state
    ss.render()
time.sleep(2)
env.close()