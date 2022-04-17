import torch as th
import numpy as np
def minerlEncodeSequence(trajectory,device,discrete_rewards=False,vae_model=None,convolutional_head = False):
            sequence_actions = []
            sequence_pov_obs = []
            sequence_rewards = []
            sequence_rtg = []
            sequence_timesteps = []
            sequence_dones = []
            
            for dataset_observation, dataset_action, dataset_reward,_,done,timesteps,rtg in trajectory:
                    sequence_actions.append(dataset_action["vector"])

                    data_obs= dataset_observation["pov"].transpose(2, 0, 1)
                    sequence_pov_obs.append(data_obs)
                    sequence_rewards.append(dataset_reward)
                    sequence_dones.append(done)
                    sequence_rtg.append(rtg)
                    sequence_timesteps.append(timesteps)
            #vae_model.eval(iter((th.tensor(sequence_pov_obs,dtype=th.float32).div(256),)))#TODO erase test
            if vae_model:
                    sequence_pov_obs=vae_model.quantize(th.tensor(sequence_pov_obs,dtype=th.float32).to(device).div(256)).flatten(start_dim=1).cpu()
            elif convolutional_head:
                    sequence_pov_obs=th.tensor(sequence_pov_obs,dtype=th.float32).div(256)
            else:
                    sequence_pov_obs=th.tensor(sequence_pov_obs,dtype=th.float32).div(256).flatten(start_dim=1)

            sequence_rtg=th.tensor(sequence_rtg)
            sequence_rewards=th.tensor(sequence_rewards)
            sequence_actions=th.tensor(sequence_actions)
            
            if discrete_rewards is not True:#TODO implement dicrete rewards properly
                    sequence_rewards=th.unsqueeze(sequence_rewards,1)
                    sequence_rtg=th.unsqueeze(sequence_rtg,1)
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

def minerlEncodeSequence_performance_test(trajectory,device,discrete_rewards=False,vae_model=None,convolutional_head = False):
            sequence_actions = []
            sequence_pov_obs = []
            sequence_rewards = []
            sequence_rtg = []
            sequence_timesteps = []
            sequence_dones = []
            
            for dataset_observation, dataset_action, dataset_reward,_,done,timesteps,rtg in trajectory:
                    sequence_actions.append(dataset_action["vector"])

                    data_obs= dataset_observation["pov"].transpose(2, 0, 1)
                    sequence_pov_obs.append(data_obs)
                    sequence_rewards.append(dataset_reward)
                    sequence_dones.append(done)
                    sequence_rtg.append(rtg)
                    sequence_timesteps.append(timesteps)
            #vae_model.eval(iter((th.tensor(sequence_pov_obs,dtype=th.float32).div(256),)))#TODO erase test
            if vae_model:
                    sequence_pov_obs=vae_model.quantize(th.tensor(sequence_pov_obs,dtype=th.float32).to(device).div(256)).flatten(start_dim=1).cpu()
            elif convolutional_head:
                    sequence_pov_obs=th.tensor(sequence_pov_obs,dtype=th.float32).div(256)
            else:
                    sequence_pov_obs=th.tensor(sequence_pov_obs,dtype=th.float32).div(256).flatten(start_dim=1)

            sequence_rtg=th.tensor(sequence_rtg)
            sequence_rewards=th.tensor(sequence_rewards)
            sequence_actions=th.tensor(sequence_actions)
            
            if discrete_rewards is not True:#TODO implement dicrete rewards properly
                    sequence_rewards=th.unsqueeze(sequence_rewards,1)
                    sequence_rtg=th.unsqueeze(sequence_rtg,1)
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