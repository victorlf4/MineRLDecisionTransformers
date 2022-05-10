import minerl
import random
import torch as th
import os
import gym


data_pipeline = minerl.data.make("MineRLObtainDiamondVectorObf-v0",  data_dir='data', num_workers=1)
#data_pipeline = minerl.data.make("MineRLTreechopVectorObf-v0",  data_dir='data', num_workers=1)
#data_pipeline = minerl.data.make(env, MINERL_DATA_ROOT)
bbi = minerl.data.BufferedBatchIter(data_pipeline, buffer_target_size=200000)
num_timesteps = 0
act=[]
for data_dict in bbi.buffered_batch_iter(batch_size=1, num_batches=100):
        act.append(data_dict[1]["vector"])




MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT')
env = gym.make('MineRLObtainDiamondVectorObf-v0')
random_act=env.action_space.sample()
mean_act=th.mean(th.tensor(act),dim=0)
mean_act = {"vector": mean_act}
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
        action = mean_act

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')