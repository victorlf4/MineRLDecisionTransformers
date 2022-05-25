import minerl
import random
import torch as th
import os
import gym
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT')

env = gym.make('MineRLObtainDiamond-v0')



for episode in range(1):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    # BC part to get some logs:
    for i in range(100):
        # Process the action:
        #   - Add/remove batch dimensions
        #   - Transpose image (needs to be channels-last)
        #   - Normalize image
        print(obs['pov'].shape)
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')