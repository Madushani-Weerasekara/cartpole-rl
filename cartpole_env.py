import gym

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment
state = env.reset()

# Run the environment for 100 steps
for _ in range(100):
    #env.render(render_mode='human') # Visualize the environment
    action = env.action_space.sample() # Randomly select an action(left or right)
    state, reward, terminated, truncated, info = env.step(action) # Take the acction
    if terminated or truncated:
        break
env.close()