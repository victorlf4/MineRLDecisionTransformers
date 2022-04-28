import minerl
import random
import torch as th
import os
def reward2go(rewards):
                rewards2go=[]
                remainingR=sum(rewards)

                for r in rewards:
                        reward2go=remainingR-r
                        rewards2go.append(reward2go)
                        remainingR=reward2go
                return rewards2go

def divide_sequences(list, sequence_lenght):
    return (list[i:i+sequence_lenght] for i in range(0, len(list), sequence_lenght))#TODO handle sequences no divisible by seq len

def calculateReward2go(trajectory):
                    rewards=[]
                    for dataset_observation, dataset_action, dataset_reward, _, done,timesteps in trajectory:
                                rewards.append(dataset_reward)
                    return reward2go(rewards)

MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT')
data_pipeline = minerl.data.make("MineRLObtainDiamondVectorObf-v0",  data_dir='data', num_workers=1)
#data_pipeline = minerl.data.make(env, MINERL_DATA_ROOT)
bbi = minerl.data.BufferedBatchIter(data_pipeline, buffer_target_size=20000)
num_timesteps = 0
for data_dict in bbi.buffered_batch_iter(batch_size=1, num_epochs=1):
        num_timesteps += 1
print(num_timesteps)
