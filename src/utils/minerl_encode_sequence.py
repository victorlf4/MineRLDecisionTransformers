import torch as th
import numpy as np
def minerlEncodeSequence(trajectory,device,discrete_rewards=False,vae_model=None,convolutional_head = False,state_vector=False,vectorize_actions=False,data_enviroment=None):
            sequence_actions = []
            sequence_pov_obs = []
            sequence_vector_obs=[]
            sequence_rewards = []
            sequence_rtg = []
            sequence_timesteps = []
            sequence_dones = []
            
            for dataset_observation, dataset_action, dataset_reward,_,done,timesteps,rtg in trajectory:
                    if vectorize_actions:
                        dataset_action=data_enviroment._wrap_action(dataset_action)
                    if(state_vector):
                        sequence_vector_obs.append(dataset_observation["vector"])

                    sequence_actions.append(dataset_action["vector"])
                    sequence_pov_obs.append(dataset_observation["pov"])
                    sequence_rewards.append(dataset_reward)
                    sequence_dones.append(done)
                    sequence_rtg.append(rtg)
                    sequence_timesteps.append(timesteps)
            #vae_model.eval(iter((th.tensor(sequence_pov_obs,dtype=th.float32).div(256),)))#TODO erase test
            if vae_model:
                    sequence_pov_obs=np.array(sequence_pov_obs).transpose(0,3,1,2)
                    sequence_pov_obs=th.tensor(sequence_pov_obs,dtype=th.float32).to(device).div(256)
                    sequence_pov_obs=vae_model.quantize(sequence_pov_obs).flatten(start_dim=1).cpu()
            elif convolutional_head:
                    sequence_pov_obs=th.tensor(sequence_pov_obs,dtype=th.float32)
            else:
                sequence_pov_obs=np.array(sequence_pov_obs)
                
            if discrete_rewards is not True:#TODO implement dicrete rewards properly
                    sequence_rewards=np.expand_dims(sequence_rewards,1)#TODO change getbatch so this is not necesary
                    sequence_rtg=np.expand_dims(sequence_rtg,1)
            sequence_dones = np.array(sequence_dones)
            trajectory ={#TODO check if we really need rewards for anything
            "observations_pov": sequence_pov_obs,
            "actions": sequence_actions,
            "rewards": sequence_rewards,
            "done": sequence_dones,
            "rewards2go": sequence_rtg,
            "timesteps":sequence_timesteps,
            }
            if state_vector:
                    trajectory["observations_vector"]=sequence_vector_obs
            return trajectory


