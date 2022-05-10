import minerl
import torch as th
import os
from numpy import save
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

minerl.data.download(directory='data', environment='MineRLObtainDiamondVectorObf-v0')
data = minerl.data.make("MineRLObtainDiamondVectorObf-v0",  data_dir='data', num_workers=1)
trajectory_names = data.get_trajectory_names()
trajectories= []
print("num trajectories")
print(len(trajectory_names))
for name in trajectory_names:
    trajectories.append(data.load_data(name, skip_interval=0, include_metadata=False))
rewards[]
for data_dict in bbi.buffered_batch_iter(batch_size=1, num_epochs=1):
        num_timesteps += 1
        rewards
print(num_timesteps)

#TODO fix this , maybe add a file to the dataset