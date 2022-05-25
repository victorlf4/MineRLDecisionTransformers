import minerl
import random
import torch as th
import os
import gym
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT')


data_pipeline = minerl.data.make("MineRLTreechopVectorObf-v0",  data_dir='data', num_workers=1)
#data_pipeline = minerl.data.make("MineRLTreechopVectorObf-v0",  data_dir='data', num_workers=1)
#data_pipeline = minerl.data.make(env, MINERL_DATA_ROOT)
bbi = minerl.data.BufferedBatchIter(data_pipeline, buffer_target_size=200000)
num_timesteps = 0
for data_dict in bbi.buffered_batch_iter(batch_size=1, num_epochs=1):
        num_timesteps += 1
        #print(data_dict)

print(num_timesteps)


