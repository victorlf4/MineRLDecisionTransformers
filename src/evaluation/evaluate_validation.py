
import numpy as np
import torch
def tokenize_image(tokenizerModel,obs):
    obs= obs.transpose(2, 0, 1)
    return tokenizerModel.encodeSingle(torch.tensor(obs,dtype=torch.float32))


def evaluate_validation_rtg(
        env,
        state_dim,
        act_dim,
        model,
        get_batch,
        batch_size,
        loss_fn,
        action_centroids=None,#if we are using kmeans
        validation_batches=10,
        scale=1000.,
        device='cuda',
        mode='normal',
        
    ):
    model.eval()
    model.to(device=device)
    validation_losses = []
    for batch in range(validation_batches):
        povs,state, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(batch_size,validation=True)
        action_target = torch.clone(actions)
        state_preds,pov_preds, action_preds, reward_preds = model.forward(
                povs, actions, rewards, rtg, timesteps, attention_mask=attention_mask,state_vector=state #rtg[:,:-1]??? in original
            )
           
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = loss_fn(
                None, action_preds, None,
                None, action_target, None,
            )#TODO add state loss option
        validation_losses.append(loss.detach().cpu().item()) 
    return validation_losses
