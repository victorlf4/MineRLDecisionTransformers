
import numpy as np
import torch
def tokenize_image(tokenizerModel,obs,device):
    obs= obs.transpose(2, 0, 1)
    return tokenizerModel.encodeSingle(torch.tensor(obs,dtype=torch.float32).to(device).div(256))


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        vq_vae,
        action_centroids=None,#if we are using kmeans
        max_ep_len=1000,
        scale=1000.,
        state_mean=np.zeros(1),#TODO send the actual mean here when not using vq_vae
        state_std=np.zeros(1),#TODO send the actual std here when not using vq_vae
        device='cuda',
        target_return=None,
        mode='normal',
        visualize=False
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if vq_vae:
        state=tokenize_image(vq_vae,state["pov"],device)
    else:
        state=torch.tensor(state["pov"]).to(device).div(256)
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)
    
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = state.reshape((1,)+ state_dim).to(device=device, dtype=torch.float32)
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
            (states.to(dtype=torch.float32) - state_mean) / state_std,#TODO make this only happen when not using vq:vae
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )'''
        action = model.get_action(
            (states.to(dtype=torch.float32)),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        #vae_model.eval(iter((th.tensor(states,dtype=th.float32).div(256),)))#TODO erase test
        actions[-1] = action
        action = action.detach().cpu().numpy()
        if action_centroids is not None:#if we are using kmeans
            action = action_centroids[np.around(action).astype(np.uint64)][0]#[0] because otherwise its a 2d array for some reason
        action = {"vector": action}
        state, reward, done, _ = env.step(action)
        if visualize:
            env.render(mode='human')
        if vq_vae:
            state=tokenize_image(vq_vae,state["pov"],device)#tokenize observation whith vq_vae
        else:
            state=torch.tensor(state["pov"]).to(device).div(256)
            
        cur_state = state.reshape((1,)+ state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

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