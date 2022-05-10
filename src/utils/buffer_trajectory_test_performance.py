import minerl
import os
import time
from copy import deepcopy
import numpy as np
from minerl.data.util import multimap
import random
import torch as th
from zmq import device

def stack(*args):
    return np.stack(args)
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

class BufferedTrajectoryIterTest:#TODO, add another version that samples whithout repetition
        def __init__(self,
                 data_pipeline,
                 buffer_target_size=5000,sequence_size=1,reward_to_go=False,reward_to_go_dictionary={},max_ep_len_dataset=100000,store_rewards2go=True):
            self.data_pipeline = data_pipeline
            self.data_buffer = []
            self.buffer_target_size = buffer_target_size
            self.traj_sizes = []
            self.avg_traj_size = 0
            self.all_trajectories = self.data_pipeline.get_trajectory_names()
            self.sequence_size=sequence_size
            self.reward_to_go=reward_to_go
            self.reward_to_go_dictionary=reward_to_go_dictionary
            self.max_ep_len_dataset=max_ep_len_dataset
            self.store_rewards2go=store_rewards2go
            # available_trajectories is a dynamic, per-epoch list that will keep track of
            # which trajectories we haven't yet used in a given epoch
            self.available_trajectories = deepcopy(self.all_trajectories)
            random.shuffle(self.available_trajectories)
        def optionally_fill_buffer(self):
                buffer_updated = False
                # Add trajectories to the buffer if the remaining space is
                # greater than our anticipated trajectory size (in the form of the empirical average)
                while (self.buffer_target_size - len(self.data_buffer)) > self.avg_traj_size:
                    if len(self.available_trajectories) == 0:
                        return
                    traj_to_load = self.available_trajectories.pop()
                    data_loader = self.data_pipeline.load_data(traj_to_load)
                    traj_len = 0
                    trajectory=[]
                    for data_frame in data_loader:
                        traj_len += 1
                        data_frame=data_frame+[traj_len,]#we add the timestep to the tuple
                        trajectory.append(data_frame)
                        if(traj_len>=self.max_ep_len_dataset-1):
                            break    

                        
                    if(self.reward_to_go) :
                        if str(traj_to_load) in self.reward_to_go_dictionary:
                            rtg=self.reward_to_go_dictionary[str(traj_to_load)]
                        else:
                            rtg=calculateReward2go(trajectory)
                            if self.store_rewards2go:
                                self.reward_to_go_dictionary[str(traj_to_load)]=rtg
                        timestep=0
                        trajectory_rtg =[]
                        for data_frame in trajectory:
                            data_frame=data_frame+[rtg[timestep],]#we add the reward to go to the tuple
                            #data_frame=minerl_performance_test(data_frame,device="cuda")
                            timestep+=1
                            trajectory_rtg.append(data_frame)
                        trajectory=trajectory_rtg

                    trajectory =list(divide_sequences(trajectory,self.sequence_size))#we divide the trajectory in chunks of sequentially ordered tuples#TODO maybe use multimap to make dictionaries instead or in adition to
                    self.traj_sizes.append(traj_len/self.sequence_size)
                    self.avg_traj_size = np.mean(self.traj_sizes)
                    self.data_buffer=self.data_buffer+ trajectory
                    buffer_updated = True
                if buffer_updated:
                    random.shuffle(self.data_buffer)
                    

            

        def get_batch(self, batch_size):
                '''ret_dict = dict(obs=data_tuple[0],
                            act=data_tuple[1],
                            reward=data_tuple[2],
                            next_obs=data_tuple[3],
                            done=data_tuple[4])
                ret_dict_list.append(ret_dict)
                return multimap(stack, *ret_dict_list)
                '''
                
                ret_batch = []
                for _ in range(batch_size):
                    data_sequence = self.data_buffer.pop()
                    data_sequence=minerlEncodeSequence_performance_test(data_sequence,"cuda",discrete_rewards=False,vae_model=None,convolutional_head=False)
                    ret_batch.append(data_sequence)

                
                return ret_batch

        def buffered_batch_iter(self, batch_size, num_epochs=None, num_batches=None):
            """
            The actual generator method that returns batches. You can specify either
            a desired number of batches, or a desired number of epochs, but not both,
            since they might conflict.

            ** You must specify one or the other **

            Args:
                batch_size: The number of transitions/timesteps to be returned in each batch
                num_epochs: Optional, how many full passes through all trajectories to return
                num_batches: Optional, how many batches to return

            """
            assert num_batches is not None or num_epochs is not None, "One of num_epochs or " \
                                                                    "num_batches must be non-None"
            assert num_batches is None or num_epochs is None, "You cannot specify both " \
                                                            "num_batches and num_epochs"

            epoch_count = 0
            batch_count = 0

            while True:
                # If we've hit the desired number of epochs
                if num_epochs is not None and epoch_count >= num_epochs:
                    return
                # If we've hit the desired number of batches
                if num_batches is not None and batch_count >= num_batches:
                    return
                # Refill the buffer if we need to
                # (doing this before getting batch so it'll run on the first iteration)
                self.optionally_fill_buffer()
                ret_batch = self.get_batch(batch_size=batch_size)
                batch_count += 1
                if len(self.data_buffer) < batch_size:
                    assert len(self.available_trajectories) == 0, "You've reached the end of your " \
                                                                "data buffer while still having " \
                                                                "trajectories available; " \
                                                                "something seems to have gone wrong"
                    epoch_count += 1
                    self.available_trajectories = deepcopy(self.all_trajectories)
                    random.shuffle(self.available_trajectories)

                #keys = ('obs', 'act', 'reward', 'next_obs', 'done')
                #yield tuple([ret_batch[key] for key in keys])
                yield ret_batch

def minerlEncodeSequence_performance_test(trajectory,device,discrete_rewards=False,vae_model=None,convolutional_head = False):
            sequence_actions = []
            sequence_pov_obs = []
            sequence_rewards = []
            sequence_rtg = []
            sequence_timesteps = []
            sequence_dones = []
            for dataset_observation, dataset_action, dataset_reward,_,done,timesteps,rtg in trajectory:
                    sequence_actions.append(dataset_action["vector"])

                    data_obs= dataset_observation["pov"]#.transpose(2, 0, 1)
                    sequence_pov_obs.append(data_obs)
                    sequence_rewards.append(dataset_reward)
                    sequence_dones.append(done)
                    sequence_rtg.append(rtg)
                    sequence_timesteps.append(timesteps)
            '''
            #vae_model.eval(iter((th.tensor(sequence_pov_obs,dtype=th.float32).div(256),)))#TODO erase test
            if vae_model:
                    sequence_pov_obs=vae_model.quantize(th.tensor(sequence_pov_obs,dtype=th.float32).to(device).div(256)).flatten(start_dim=1).cpu()
            elif convolutional_head:
                    sequence_pov_obs=th.tensor(sequence_pov_obs,dtype=th.float32).div(256)
            else:
                    sequence_pov_obs=th.tensor(sequence_pov_obs,dtype=th.float32)#.div(256).flatten(start_dim=1)
            
            sequence_rtg=th.tensor(sequence_rtg)
            sequence_rewards=th.tensor(sequence_rewards)
            sequence_actions=th.tensor(sequence_actions)
            if discrete_rewards is not True:#TODO implement dicrete rewards properly
                    sequence_rewards=th.unsqueeze(sequence_rewards,1)
                    sequence_rtg=th.unsqueeze(sequence_rtg,1)
         
            
            '''
            if discrete_rewards is not True:#TODO implement dicrete rewards properly
                    sequence_rewards=np.expand_dims(sequence_rewards,1)
                    sequence_rtg=np.expand_dims(sequence_rtg,1)
         
            sequence_dones = np.array(sequence_dones)
            trajectory ={#TODO check if we really need rewards for anything
            "observations": sequence_pov_obs,
            "actions": sequence_actions,
            "rewards": sequence_rewards,
            "done": sequence_dones,
            "rewards2go": sequence_rtg,
            "timesteps":sequence_timesteps,
            }
            return trajectory

def minerl_performance_test(trajectory,device,discrete_rewards=False,vae_model=None,convolutional_head = False):
            #vae_model.eval(iter((th.tensor(sequence_pov_obs,dtype=th.float32).div(256),)))#TODO erase test

            
            ''''if discrete_rewards is not True:#TODO implement dicrete rewards properly
                    trajectory[2]=th.unsqueeze(trajectory[2],1)
                    trajectory[4]=th.unsqueeze(trajectory[4],1)
            '''
            sequence_dones = np.array(trajectory[3])
            trajectory ={#TODO check if we really need rewards for anything
            "observations": trajectory[0]["pov"],#transpose(2, 0, 1))
            "actions": trajectory[1]["vector"],
            "rewards": trajectory[2],
            "done": sequence_dones,
            "rewards2go": trajectory[4],
            "timesteps":trajectory[5],
            }
            return trajectory



if __name__ == "__main__":

    #env = "MineRLBasaltMakeWaterfall-v0"
    env = "MineRLTreechopVectorObf-v0"
    test_batch_size = 32

    start_time = time.time()
    minerl.data.download(directory='data', environment=env)#TODO codigo de baselines competition
    data_pipeline =   minerl.data.make(env,  data_dir='data')
    bbi = BufferedTrajectoryIterTest(data_pipeline,sequence_size=20,reward_to_go=True, buffer_target_size=10000)
    num_timesteps = 0
    for data_dict in bbi.buffered_batch_iter(batch_size=test_batch_size, num_epochs=1):
        num_timesteps +=1 #len(data_dict['obs']['pov'])

    print(f"{num_timesteps} found for env {env} using batch_iter")
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")
    