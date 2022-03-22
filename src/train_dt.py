import random
import numpy as np
import torch as th
from torch import nn
import gym
import minerl
from sklearn.cluster import KMeans
from torch.utils.data.dataset import IterableDataset
from  Model.vq_vae import VectorQuantizerVAE as vaeq
import os
import wandb
import argparse
from Model.decision_transformer import DecisionTransformer
from training.seq_trainer import SequenceTrainer
from evaluation.evaluate_episodes import evaluate_episode_rtg
from copy import deepcopy

BATCH_SIZE = 32
NUM_ACTION_CENTROIDS = 100  # Number of KMeans centroids used to cluster the data.


#TODO maybe try things like t-SNE and UMAP or other way of reemplacing vae?
#TODO add some way to run an unmodified dt for baseline

def reward2go(rewards):
                rewards2go=[]
                remainingR=sum(rewards)

                for r in rewards:
                        reward2go=remainingR-r
                        rewards2go.append(reward2go)
                        remainingR=reward2go
                return rewards2go

#we precalculate the rewards to go to avoid loading the entire dataset every time we train
#TODO maybe precalculate this only once
def calculate_rewards2go(trajectories):
        dataset_rewards2go=[]
        for trajectory in trajectories:
                rewards=[]
                for dataset_observation, dataset_action, dataset_reward, _, done in trajectory:
                        rewards.append(dataset_reward)
                dataset_rewards2go.append(reward2go(rewards))
        return dataset_rewards2go





def main(parameters
):              
        #loading parameters
        use_checkpoint=parameters['checkpoint']
        batch_size = parameters['batch_size']#TODO check if it needs to be get 
        num_eval_episodes = parameters['num_eval_episodes']
        discrete_actions = parameters['kmeans_actions']
        discrete_rewards =parameters['discrete_rewards']
        log_to_wandb=parameters['log_to_wandb']
        max_length=parameters['K']
        max_ep_len=parameters['max_ep_len']
        max_ep_len_dataset=parameters['max_ep_len_dataset']
        
        
        num_steps_per_iter=parameters['num_steps_per_iter']
        max_iters=parameters['max_iters']
        checkpoint_file ="./models/"+parameters['checkpoint_name']
        mode = parameters['checkpoint_name']
        env_targets = [1571,547]#target rewards for #TODO make them params
        if parameters['device']=="cuda":
                device = th.device("cuda" if th.cuda.is_available() else "cpu")#TODO send a message when cuda is not avaliable
        else:#TODO send a warning instead and ask for explicitly device name
                device = th.device("cpu")#TODO fix not using cuda breaking things in dt   
        vae_model=vaeq(parameters["vae_model"],embedding_dim = parameters["vae_embedding_dim"],num_embeddings =parameters["vae_embedings"],batch_size=16)
        #Load minecraft env and dataset
        minerl.data.download(directory='data', environment=parameters["dataset"])#TODO codigo de baselines competition
        data = minerl.data.make(parameters["dataset"],  data_dir='data', num_workers=1)
        env = gym.make(parameters["env"])
        #Load trajectories
        trajectory_names = data.get_trajectory_names()
        random.shuffle(trajectory_names)
        trajectories_dataset= []
        for name in trajectory_names:
                trajectories_dataset.append(data.load_data(name))#TODO, check and ask whether these do anything , skip_interval=0, include_metadata=False
        dataset_rewards2go=calculate_rewards2go(trajectories_dataset)#We precalculate all the rewards2go for the dataset
        print("rewards to go mean:")
        print(np.mean(np.max(dataset_rewards2go),axis=0))
        print(np.max(np.max(dataset_rewards2go),axis=0))
        #print(np.max(dataset_rewards2go,axis=0))#TODO get results to make sense
        state_dim=256*parameters["vae_embedding_dim"] #TODO actually figure out why its 256 
        
        
        def save_kmeans():#TODO maybe move small functions like this to an utils file
                th.save({
                        'action_centroids': action_centroids,
                        },checkpoint_file+"_kmeans")

        def load_kmeans():
                checkpoint = th.load(checkpoint_file+"_kmeans")
                action_centroids=checkpoint['action_centroids']
                return action_centroids

        if discrete_actions:#TODO refactor this somehow
                act_dim=1
                num_action_centroids=NUM_ACTION_CENTROIDS
                if os.path.exists(checkpoint_file+"_kmeans"):
                        action_centroids=load_kmeans()
                else:
                        print("Running KMeans on the action vectors")
                        kmeans = KMeans(n_clusters=NUM_ACTION_CENTROIDS)
                        kmeans.fit(all_actions)
                        action_centroids = kmeans.cluster_centers_
                        print("KMeans done")
                        save_kmeans()
        else:
                act_dim = env.action_space["vector"].shape[0]
                num_action_centroids=None
                action_centroids=None

        #TODO check whether time embeddings are really necesary as oposed to some in sequence position embedding.
        def minerlEncode(trajectory,index,traj_slice):#TODO maybe do this in a diferent file or do it separately and store in a file?
                sequence_actions = []
                sequence_pov_obs = []
                sequence_rewards = []
                sequence_dones = []
                
                for dataset_observation, dataset_action, dataset_reward, _, done in trajectory:
                        sequence_actions.append(dataset_action["vector"])

                        data_obs= dataset_observation["pov"].transpose(2, 0, 1)
                        obs_encoding_tensor=vae_model.quantizeSingle(th.tensor(data_obs,dtype=th.float32)).flatten()#encode or quantize?
                        sequence_pov_obs.append(obs_encoding_tensor.cpu().t())

                        sequence_rewards.append(dataset_reward)
                        sequence_dones.append(done)

                sequence_actions = np.array(sequence_actions)
                sequence_pov_obs = th.stack(sequence_pov_obs)
                sequence_rewards = np.array(sequence_rewards)
                sequence_rewards2go=dataset_rewards2go[index][traj_slice]#TODO convertir en tensor directamente en el original

                sequence_rewards=th.from_numpy(sequence_rewards)#TODO use convert_to_tensor directamente instead
                sequence_rewards2go=th.tensor(sequence_rewards2go)
                sequence_actions=th.from_numpy(sequence_actions)

                if discrete_rewards is not True:
                        sequence_rewards=th.unsqueeze(sequence_rewards,1)
                        sequence_rewards2go=th.unsqueeze(sequence_rewards2go,1)
                sequence_dones = np.array(sequence_dones)
                trajectory ={
                "observations": sequence_pov_obs,
                "actions": sequence_actions,
                "rewards": sequence_rewards,
                "done": sequence_dones,
                "rewards2go": sequence_rewards2go,
                }
                return trajectory

                
        
        '''
        def test_load_trajectories(n):
                loadedTrajectories=[]
                for i in range(n):
                        trajectory_name=trajectory_names[i] 
                        trajectory=  data.load_data(trajectory_name, skip_interval=0, include_metadata=False)     
                        loadedTrajectories.append(list(trajectory))
                return loadedTrajectories
        #loadedTrajectories=test_load_trajectories(5)#TODO! erase after test
        ''''''
        def get_batch_test_load_memory(batch_size=64, max_len=max_length):#original version where we donload certain trajectories to memoryh
                obsBatch=[]
                actionBatch=[]
                rewardsBatch=[]
                doneBatch=[]
                rtgBatch=[]
                timesteps=[]
                mask =[]

                for i in range(batch_size):
                        trajectory_index =random.randint(0,len(loadedTrajectories)-1)#TODO maybe do this whith a generator(using yield)
                        trajectory=loadedTrajectories[trajectory_index]
                        inicio = random.randint(0,min(len(trajectory)-2,max_ep_len_dataset))

                        sequence=minerlEncode(trajectory[inicio:inicio+max_length],trajectory_index,slice(inicio,inicio+max_length))
                        #step =trajectory[inicio:max_length]
                        #We feed  max_length timesteps into Decision Transformer, for a total of 3*max_length tokens 
                        sequence_lenght=sequence["observations"].shape[0]
                        obsBatch.append(sequence["observations"])
                        actionBatch.append(sequence["actions"])
                        rewardsBatch.append(sequence["rewards"])
                        doneBatch.append(sequence["done"])
                        rtgBatch.append(sequence["rewards2go"])
                        timesteps.append(range(inicio,min(inicio+max_length,len(trajectory))))
                        state_dim=obsBatch[-1].shape[1]#shape of obs encoding
                        
                        obsBatch[-1] = np.concatenate([np.zeros((max_len - sequence_lenght, state_dim)), obsBatch[-1]], axis=0)
                        if discrete_actions:#TODo maybe move this diference to a reshape , leave this as the non discrete version and unsqueeze on the other side 
                                actionBatch[-1] = np.concatenate([np.zeros(max_len - sequence_lenght), actionBatch[-1]], axis=0)#TODO *-10 mask or =?
                        else:
                                actionBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, act_dim)) , actionBatch[-1]], axis=0)
                        if discrete_rewards:
                                rewardsBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght)), rewardsBatch[-1]], axis=0)
                        else:
                                rewardsBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, 1)), rewardsBatch[-1]], axis=0)
                        doneBatch[-1] = np.concatenate([np.ones(( max_len - sequence_lenght)) * 2, doneBatch[-1]], axis=0)#adds 2 as padding ? maybe so its not 1 or 0
                        if discrete_rewards:
                                rtgBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght)), rtgBatch[-1]], axis=0)
                        else:
                                rtgBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, 1)), rtgBatch[-1]], axis=0)
                        timesteps[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght)), timesteps[-1]], axis=0)
                        mask.append(np.concatenate([np.zeros((max_len - sequence_lenght)), np.ones((sequence_lenght))], axis=0))
                obsBatch = th.tensor(obsBatch).to(dtype=th.float32, device=device)#change to int only if discrete actions
                actionBatch = th.tensor(actionBatch).to(dtype=th.float32, device=device)
                rewardsBatch = th.tensor(rewardsBatch).to(dtype=th.float32, device=device)
                rtgBatch = th.tensor(rtgBatch).to(dtype=th.float32, device=device)   
                timesteps = th.tensor(timesteps).to(dtype=th.long, device=device)
                mask = th.tensor(mask).to(device=device)
                return obsBatch,actionBatch,rewardsBatch,doneBatch ,rtgBatch,timesteps,mask
        '''  
        def get_batch(batch_size=64, max_len=max_length):#version where we sample each batch from each trajectpry
                obsBatch=[]
                actionBatch=[]
                rewardsBatch=[]
                doneBatch=[]
                rtgBatch=[]
                timesteps=[]
                mask =[]
                trajectory_index =random.randint(0,len(trajectory_names)-1)#TODO maybe do this whith a generator(using yield)
                trajectory_name=trajectory_names[trajectory_index]
                trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
                trajectory=list(trajectory)#load the trajectory from generator
                for i in range(batch_size):     
                        inicio = random.randint(0,min(len(trajectory)-2,max_ep_len_dataset))
                        sequence=minerlEncode(trajectory[inicio:inicio+max_length],trajectory_index,slice(inicio,inicio+max_length))
                        #step =trajectory[inicio:max_length]
                        #We feed  max_length timesteps into Decision Transformer, for a total of 3*max_length tokens 
                        sequence_lenght=sequence["observations"].shape[0]
                        obsBatch.append(sequence["observations"])
                        actionBatch.append(sequence["actions"])
                        rewardsBatch.append(sequence["rewards"])
                        doneBatch.append(sequence["done"])
                        rtgBatch.append(sequence["rewards2go"])
                        timesteps.append(range(inicio,min(inicio+max_length,len(trajectory))))
                        state_dim=obsBatch[-1].shape[1]#shape of obs encoding
                        
                        obsBatch[-1] = np.concatenate([np.zeros((max_len - sequence_lenght, state_dim)), obsBatch[-1]], axis=0)
                        if discrete_actions:#TODo maybe move this diference to a reshape , leave this as the non discrete version and unsqueeze on the other side 
                                actionBatch[-1] = np.concatenate([np.zeros(max_len - sequence_lenght), actionBatch[-1]], axis=0)#TODO *-10 mask or =?
                        else:
                                actionBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, act_dim)) , actionBatch[-1]], axis=0)
                        if discrete_rewards:
                                rewardsBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght)), rewardsBatch[-1]], axis=0)
                        else:
                                rewardsBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, 1)), rewardsBatch[-1]], axis=0)
                        doneBatch[-1] = np.concatenate([np.ones(( max_len - sequence_lenght)) * 2, doneBatch[-1]], axis=0)#adds 2 as padding ? maybe so its not 1 or 0
                        if discrete_rewards:
                                rtgBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght)), rtgBatch[-1]], axis=0)
                        else:
                                rtgBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, 1)), rtgBatch[-1]], axis=0)
                        timesteps[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght)), timesteps[-1]], axis=0)
                        mask.append(np.concatenate([np.zeros((max_len - sequence_lenght)), np.ones((sequence_lenght))], axis=0))
                obsBatch = th.tensor(obsBatch).to(dtype=th.float32, device=device)#change to int only if discrete actions
                actionBatch = th.tensor(actionBatch).to(dtype=th.float32, device=device)
                rewardsBatch = th.tensor(rewardsBatch).to(dtype=th.float32, device=device)
                rtgBatch = th.tensor(rtgBatch).to(dtype=th.float32, device=device)   
                timesteps = th.tensor(timesteps).to(dtype=th.long, device=device)
                mask = th.tensor(mask).to(device=device)
                return obsBatch,actionBatch,rewardsBatch,doneBatch ,rtgBatch,timesteps,mask

        
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=max_length,
            max_ep_len=max_ep_len_dataset,#diferent from max_ep_len because theres a bug where the dataset has trajectories bigger than the maximun episode lenght of the enviroment
            discrete_rewards=None,#TODO actually add this in case of discrete rewards #1095,#1095 is the number of posible reward combinations in the minerl enviroment, though most are very unlikel
            discrete_actions=num_action_centroids,
            discrete_states=parameters["vae_embedings"],
            hidden_size=parameters['embed_dim'],
            n_layer=parameters['n_layer'],
            n_head=parameters['n_head'],
            n_inner=4*parameters['embed_dim'],
            activation_function=parameters['activation_function'],
            n_positions=1024,
            resid_pdrop=parameters['dropout'],
            attn_pdrop=parameters['dropout'],
            natureCNN=parameters['convolution_head'])
        model = model.to(device=device)

        warmup_steps = parameters['warmup_steps']

        optimizer = th.optim.AdamW(
        model.parameters(),
        lr=parameters['learning_rate'],
        weight_decay=parameters['weight_decay'])

        scheduler = th.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1))
        def eval_episodes(target_rew):
                def fn(model):
                        returns, lengths = [], []
                        for _ in range(num_eval_episodes):
                                with th.no_grad(): 
                                        ret, length = evaluate_episode_rtg(
                                                env,
                                                state_dim,
                                                act_dim,
                                                model,
                                                vae_model,
                                                action_centroids=action_centroids,
                                                max_ep_len=max_ep_len,
                                                scale=1,#TODO check if scalling would make sense
                                                target_return=target_rew/1,
                                                mode=mode,
                                                device=device,
                                                )
                                returns.append(ret)
                                lengths.append(length)
                        return {
                                f'target_{target_rew}_return_mean': np.mean(returns),
                                f'target_{target_rew}_return_std': np.std(returns),
                                f'target_{target_rew}_return_max': np.max(returns),
                                f'target_{target_rew}_length_mean': np.mean(lengths),
                                f'target_{target_rew}_length_std': np.std(lengths),
                        }
                return fn
                
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: th.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
        if log_to_wandb:#TODO add a way to add experiment group and name as parameters(maybe already in original paper?)
                wandb.init(
            name=f'Initial_test_refactor-{random.randint(int(1e5), int(1e6) - 1)}',
            group="Initial_test_refactor",
            project='decision-transformer_TFM',
            config=parameters
                )

        def save(epoch,checkpoint_file):#TODO maybe merge whith kmeans
                th.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },checkpoint_file)
        def load(validation_data=None):
                checkpoint = th.load(checkpoint_file)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        


        if os.path.exists(checkpoint_file) and use_checkpoint:
                print("loading save")
                load()


        for iter in range(max_iters):
                outputs = trainer.train_iteration(num_steps=num_steps_per_iter, iter_num=iter+1, print_logs=True)#TODO!  make it so there are 10 normal vbalidation steps folowed by a rl validation step
                if(use_checkpoint):#TODO make a separate variable for saving checkpoints to use checkpoint whithout saving
                        save(iter,checkpoint_file)
                if log_to_wandb:
                        wandb.log(outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #TODO add warnings for wrong parameters.check whether it can be done whith argparse
    parser.add_argument('--env', type=str, default='MineRLObtainDiamondVectorObf-v0')
    parser.add_argument('--dataset', type=str, default='MineRLObtainDiamondVectorObf-v0')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--max_ep_len', type=int, default=18000)#default of the diamond env
    parser.add_argument('--max_ep_len_dataset', type=int, default=65536)#nice round number thats almost
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)#TODO maybe rename or change how it works to have diferent sizes of embedings for rewards
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=256)#TODO fix this so it doest crash whenever its not 128
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--checkpoint', type=bool, default=False)#TODO add warning if checkpoint is false and there is a non default checkpoint name(maybe theres some way to make conditional parsing
    parser.add_argument('--checkpoint_name', type=str, default="decisiontransformers_8x8x1vaeq_256_noKmeans")#TODO change
    parser.add_argument('--convolution_head', type=bool, default=False)
    parser.add_argument('--minerl_samples', type=int, default=5)
    parser.add_argument('--vae_model', type=str, default="embedingdim_1")
    parser.add_argument('--vae_embedings', type=int, default=65536)
    parser.add_argument('--vae_embedding_dim', type=int, default=1)
    parser.add_argument('--kmeans_actions', type=bool, default=False)#TODO make this actually work , currently causes strange errors.
    parser.add_argument('--kmeans_action_centroids', type=int, default=128)#TODO make this actually work , currently causes strange errors.
    parser.add_argument('--discrete_rewards', type=bool, default=False)#TODO make this actually work, needs some way of mapping actions to specific indices
    
        #TODO make it so save fines are generated depending on parameters
    args = parser.parse_args()

    main(parameters=vars(args))