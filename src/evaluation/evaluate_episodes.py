
import numpy as np
import torch
def tokenize_image(tokenizerModel,obs,device):
    obs= obs.transpose(2, 0, 1)
    return tokenizerModel.quantizeSingle(torch.tensor(obs,dtype=torch.float32).to(device).div(256))


def evaluate_episode_rtg(
        env,
        state_pov_dim,
        act_dim,
        model,
        vq_vae,
        state_vector_dim=None,
        action_centroids=None,#if we are using kmeans
        max_ep_len=1000,
        scale=1000.,
        pov_mean=np.zeros(1),#TODO send the actual mean here when not using vq_vae
        pov_std=np.zeros(1),#TODO send the actual std here when not using vq_vae
        device='cuda',
        target_return=None,
        mode='normal',
        visualize=False
    ):

    model.eval()
    model.to(device=device)

    pov_mean = torch.from_numpy(pov_mean).to(device=device)
    pov_std = torch.from_numpy(pov_std).to(device=device)

    state = env.reset()
    if vq_vae is not None:
        state_pov=tokenize_image(vq_vae,state["pov"],device)
    else:
        state_pov=torch.tensor(state["pov"]).to(device).div(256)
    if mode == 'noise':
        state_pov = state_pov + np.random.normal(0, 0.1, size=state.shape)
    if state_vector_dim is not None:
        state_vector=torch.tensor(state["vector"]).to(device)
    else:
        state_vectors=None
    
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states_pov = state_pov.reshape((1,)+ state_pov_dim).to(device=device, dtype=torch.float32)
    if state_vector_dim is not None:
        state_vectors = state_vector.reshape((1,)+ state_vector_dim).to(device=device, dtype=torch.float32)
    
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        '''action = model.get_action(
            (states.to(dtype=torch.float32) - pov_mean) / state_std,#TODO make this only happen when not using vq:vae
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )'''

        if state_vectors is not None:
            state_vectors=state_vectors.to(dtype=torch.float32)
    
        
        action = model.get_action(
            (states_pov.to(dtype=torch.float32)),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            states_vector=state_vectors,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        action = {"vector": action}
        state, reward, done, _ = env.step(action)
        if visualize:
            env.render(mode='human')
        if vq_vae is not None:#TODO make a function
            state_pov=tokenize_image(vq_vae,state["pov"],device)#tokenize observation whith vq_vae
        else:
            state_pov=torch.tensor(state["pov"]).to(device).div(256)
        if state_vector_dim is not None:
            state_vector=torch.tensor(state["vector"]).to(device)
        
            
        cur_state = state_pov.reshape((1,)+ state_pov_dim)
        states_pov = torch.cat([states_pov, cur_state], dim=0)
        rewards[-1] = reward
        if state_vector_dim is not None:
            cur_state_vector = state_vector.reshape((1,)+ state_vector_dim)
            state_vectors=torch.cat([state_vectors, cur_state_vector], dim=0)
        

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        if reward != 0:
            print(reward)
        
        if done:
            break
    return episode_return, episode_length