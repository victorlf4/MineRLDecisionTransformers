from pickle import TRUE
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
from evaluation.evaluate_validation import evaluate_validation_rtg
from gym.wrappers import Monitor
from minerl.herobraine.wrappers.video_recording_wrapper import VideoRecordingWrapper
from minerl.herobraine.wrappers.vector_wrapper  import Vectorized
from utils.verify_or_download_minerl import verify_or_download_dataset
from utils.minerl_encode_sequence import minerlEncodeSequence
from utils.minerl_iterators import MinerlActionIterator
from utils.buffer_trajectory_load import BufferedTrajectoryIter

#TODO maybe try things like t-SNE and UMAP or other way of reemplacing vae?
#TODO add some way to run an unmodified dt for baseline
#TODO add optional convnet when not quantized, + maybe vq_vae convnet option although that makes less sense)
#TODO change action dictionary to a vector in basalt envs
#TODO propagate gradients througth frozen vq_vae
def reward2go(rewards):
                rewards2go=[]
                remainingR=sum(rewards)

                for r in rewards:
                        reward2go=remainingR-r
                        rewards2go.append(reward2go)
                        remainingR=reward2go
                return rewards2go

#we precalculate the rewards to go to avoid loading the entire dataset every time we train
#TODO add storing in a file
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
        use_checkpoint=parameters['use_checkpoint']
        batch_size = parameters['batch_size']#TODO check if it needs to be get 
        num_eval_episodes = parameters['num_eval_episodes']
        discrete_actions = parameters['kmeans_actions']
        kmeans_action_centroids=parameters['kmeans_action_centroids']
        discrete_rewards =parameters['discrete_rewards']
        discrete_pov =parameters['discrete_pov']
        convolution_head=parameters['convolution_head']
        vectorize_actions=parameters['vectorize_actions']
        
        if discrete_pov:
                discrete_states=parameters["vae_embedings"]
        else:
                discrete_states=None

        log_to_wandb=parameters['log_to_wandb']
        max_length=parameters['K']
        max_ep_len=parameters['max_ep_len']
        max_ep_len_dataset=parameters['max_ep_len_dataset']
        buffer_target_size=parameters['buffer_target_size']
        buffer_target_size_validation=parameters['buffer_target_size_validation']
        warmup_steps = parameters['warmup_steps']
        visualize=parameters['visualize']
        
        num_steps_per_iter=parameters['num_steps_per_iter']
        max_iters=parameters['max_iters']
        checkpoint_file ="./models/"+parameters['checkpoint_name']
        mode = parameters['checkpoint_name']
        #env_targets = [1571,547]#target rewards for #TODO make them params
        env_targets = parameters["target_rewards"]#target rewards for 
        if discrete_pov:
                state_dim=256*parameters["vae_embedding_dim"] #TODO actually figure out why its 256 
        elif convolution_head:
                state_dim=(3,64,64) 
        else:
                state_dim=(3*64*64,)
                
        if parameters['device']=="cuda":
                device = th.device("cuda" if th.cuda.is_available() else "cpu")
                if device == "cpu":
                        print("cuda device not avaliable,using cpu")
        else:
                device = th.device(parameters['device'])
        if discrete_pov:
                vae_model=vaeq(parameters["vae_model"],embedding_dim = parameters["vae_embedding_dim"],num_embeddings =parameters["vae_embedings"],device_name=parameters["device"],batch_size=16)
                vae_model.load()
        else:
                vae_model=None
        #Load minecraft env and dataset
        verify_or_download_dataset(directory='data', environment=parameters["dataset"])#TODO codigo de baselines competition para solo intentar descarbgar si no existe
        data = minerl.data.make(parameters["dataset"],  data_dir='data', num_workers=4)
        def load_env(env_name):
                if vectorize_actions:
                        spec_pipeline =   minerl.data.make(env_name,  data_dir='data')
                        enviroment=spec_pipeline.spec
                        enviroment=Vectorized(enviroment)
                        enviroment.register()
                        enviroment=minerl.herobraine.wrappers.Vectorized(enviroment)
                        env_name=enviroment._update_name(env_name)
                enviroment = gym.make(env_name)
                if parameters['record']:
                        enviroment=Monitor(env,"./video",force=True)#TODO fix some bugs whith not doing the steps at the same time
                
                return enviroment
        env=load_env(parameters["env"])

        #Load validation data
        verify_or_download_dataset(directory='data', environment=parameters["dataset_validation"])#TODO allow to divide minerl diamond instead or use a custom dataset or minecraft run.
        data_validation = minerl.data.make(parameters["dataset_validation"],  data_dir='data', num_workers=1)

        def save_kmeans():#TODO maybe move small functions like this to an utils file
                th.save({
                        'action_centroids': action_centroids,
                        },checkpoint_file+"_kmeans")

        def load_kmeans():
                checkpoint_kmeans = th.load(checkpoint_file+"_kmeans")
                action_centroids=checkpoint_kmeans['action_centroids']
                return action_centroids
        act_dim = env.action_space["vector"].shape[0]

        if discrete_actions:
                action_centroids=kmeans_action_centroids
                if os.path.exists(checkpoint_file+"_kmeans"):
                        action_centroids=load_kmeans()
                else:
                        action_iterator =MinerlActionIterator(data)#TODO sample randomly from any trajectory instead.
                        action_samples=[]
                        for i in range(100000):
                                action_samples.append(next(action_iterator))
                        print("Running KMeans on the action vectors")
                        kmeans = KMeans(n_clusters=kmeans_action_centroids)
                        kmeans.fit(action_samples)
                        action_centroids = th.tensor(kmeans.cluster_centers_,dtype=th.float32)
                        print("KMeans done")
                        save_kmeans()
        else:
                action_centroids=None
                action_centroids=None

        trajectory_buffer=BufferedTrajectoryIter(data,buffer_target_size=buffer_target_size,sequence_size=max_length,reward_to_go=TRUE,max_ep_len_dataset=max_ep_len_dataset,store_rewards2go=parameters["store_rewards2go"])
        trajectory_buffer_iter=trajectory_buffer.buffered_batch_iter(batch_size,num_batches=(num_steps_per_iter*max_iters)+warmup_steps)
        
        trajectory_buffer_validation=BufferedTrajectoryIter(data_validation,buffer_target_size=buffer_target_size_validation,sequence_size=max_length,reward_to_go=TRUE,max_ep_len_dataset=max_ep_len_dataset,store_rewards2go=parameters["store_rewards2go"])#TODO make buffer target size parameter
        trajectory_buffer_iter_validation=trajectory_buffer_validation.buffered_batch_iter(batch_size,num_batches=(num_steps_per_iter*max_iters))#TODO make this dependent on num of validation iterations
        def get_batch(batch_size=64, max_len=max_length, validation=False):#version where we sample each batch from each trajectory
                obsBatch=[]
                actionBatch=[]
                rewardsBatch=[]
                doneBatch=[]
                rtgBatch=[]
                timesteps=[]
                mask =[]
                if validation:
                         trajectory_batch=next(trajectory_buffer_iter_validation)
                else:
                        trajectory_batch=next(trajectory_buffer_iter)
                for trajectory in trajectory_batch:
                                sequence=minerlEncodeSequence(trajectory,device,discrete_rewards=discrete_rewards,vae_model=vae_model)
                                #We feed  max_length timesteps into Decision Transformer, for a total of 3*max_length tokens 
                                sequence_lenght=sequence["observations"].shape[0]
                                obsBatch.append(sequence["observations"])
                                actionBatch.append(sequence["actions"])
                                rewardsBatch.append(sequence["rewards"])
                                doneBatch.append(sequence["done"])
                                rtgBatch.append(sequence["rewards2go"])
                                timesteps.append(sequence["timesteps"])
                                if discrete_pov:
                                        state_dim=(obsBatch[-1].shape[1],)#shape of obs encoding
                                elif convolution_head:
                                        state_dim=(obsBatch[-1].shape[1],obsBatch[-1].shape[2],obsBatch[-1].shape[3])#shape of images
                                else:
                                        state_dim=(obsBatch[-1].shape[1],)#shape of images
                                
              
                                obsBatch[-1] = np.concatenate([np.zeros((max_len - sequence_lenght,)+state_dim), obsBatch[-1]], axis=0)#+concantenates tuples here
                                '''
                                if discrete_actions:#TODO maybe move this diference to a reshape , leave this as the non discrete version and unsqueeze on the other side 
                                        actionBatch[-1] = np.concatenate([np.zeros(max_len - sequence_lenght), actionBatch[-1]], axis=0)#TODO *-10 mask or =?
                                else:
                                        actionBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, act_dim)) , actionBatch[-1]], axis=0)
                                '''

                                actionBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, act_dim)) , actionBatch[-1]], axis=0)
                                if discrete_rewards:
                                        rewardsBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght)), rewardsBatch[-1]], axis=0)
                                else:
                                        rewardsBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, 1)), rewardsBatch[-1]], axis=0)
                                doneBatch[-1] = np.concatenate([np.ones(( max_len - sequence_lenght)) * 2, doneBatch[-1]], axis=0)
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
            kmeans_centroids=action_centroids,
            discrete_states=discrete_states,
            hidden_size=parameters['embed_dim'],
            n_layer=parameters['n_layer'],
            n_head=parameters['n_head'],
            n_inner=4*parameters['embed_dim'],
            activation_function=parameters['activation_function'],
            n_positions=1024,
            resid_pdrop=parameters['dropout'],
            attn_pdrop=parameters['dropout'],
            natureCNN=convolution_head)
        model = model.to(device=device)

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
                                                visualize=visualize
                                                )
                                returns.append(ret)
                                lengths.append(length)
                        return {
                                f'target_{target_rew}_return_mean': np.mean(returns),
                                f'target_{target_rew}_return_std': np.std(returns),
                                f'target_{target_rew}_return_max': np.max(returns),
                                f'target_{target_rew}_length_mean': np.mean(lengths),
                                f'target_{target_rew}_length_std': np.std(lengths),#TODO maybe erase because they arent really useful for minerl
                        }
                return fn
        def validation_fn(model):
                        returns = []
                       
                        with th.no_grad(): 
                                ret=evaluate_validation_rtg(
                                env,
                                state_dim,
                                act_dim,
                                model,
                                get_batch,
                                batch_size,
                                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: th.mean((a_hat - a)**2),
                                action_centroids=None,#if we are using kmeans
                                validation_batches=10,
                                scale=1.,
                                device='cuda',
                                mode='normal',
                                
                                )
                                returns.append(ret)

                        return {
                                f'loss_mean': np.mean(returns),
                                f'loss_std': np.std(returns)
                        }
        

        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: th.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            validation_fn=validation_fn
        )
        if log_to_wandb:
                wandb.init(
                        name=f'{parameters["group_name"]}-{random.randint(int(1e5), int(1e6) - 1)}',
                        group=parameters["group_name"],
                        project='decision-transformer_TFM',
                        config=parameters)

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
                print("Loading saved decision transformer model")#TODO make it so weight and biases continues in the same run if possible
                load()
        

        for iteration in range(max_iters):
                validate = (iteration%parameters["num_validation_iters"]) == 0 and iteration != 0
                outputs = trainer.train_iteration(num_steps=num_steps_per_iter, iter_num=iteration+1, print_logs=True,validation=validate)
                if(use_checkpoint):#TODO make a separate variable for saving checkpoints to use checkpoint whithout saving
                        save(iteration,checkpoint_file)
                if log_to_wandb:
                        wandb.log(outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #TODO make convolution_head and discrete_pov flags into a "encoder" option
    #TODO add warnings for wrong parameters.check whether it can be done whith argparse
    parser.add_argument('--env', type=str, default='MineRLObtainDiamondVectorObf-v0')
    parser.add_argument('--dataset', type=str, default='MineRLObtainDiamondVectorObf-v0')
    parser.add_argument('--dataset_validation', type=str, default='MineRLObtainIronPickaxeVectorObf-v0')
    parser.add_argument('--target_rewards', nargs='+', default=[64])#Accepts multiple imputs#TODO fix bug where it interpre
    parser.add_argument('--vectorize_actions' , type=bool, default=False)
    parser.add_argument('--visualize' , type=bool, default=False)
    parser.add_argument('--record' , type=bool, default=False)
    parser.add_argument('--max_ep_len', type=int, default=18000)#default of the diamond env
    parser.add_argument('--max_ep_len_dataset', type=int, default=65536)#nice round number thats almost
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)#TODO maybe rename or change how it works to have diferent sizes of embedings for rewards
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_target_size', type=int, default=3000)#Has to be bigger than the biggest trajectory on your dataset or it will try downloading it.TODO fix that.
    parser.add_argument('--buffer_target_size_validation', type=int, default=3000)
    parser.add_argument('--store_rewards2go', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=256)#TODO fix this so it doest crash whenever its not 128
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=100)
    parser.add_argument('--num_validation_iters', type=int, default=10)#num of validation iterations before running the minerl env.
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--group_name','-g' , type=str, default="vqvae_diamond_fixed_embed_dim_debugged_vq_vae")
    parser.add_argument('--use_checkpoint', type=bool, default=False)#TODO add warning if checkpoint is false and there is a non default checkpoint name(maybe theres some way to make conditional parsing
    parser.add_argument('--checkpoint_name', type=str, default="decisiontransformers_convolution")#TODO change
    parser.add_argument('--convolution_head', type=bool, default=False)
    #VQ_VAE configurations
    parser.add_argument('--discrete_pov', type=bool, default=False)
    parser.add_argument('--vae_model', type=str, default="embedingdim_1")
    parser.add_argument('--vae_embedings', type=int, default=65536)
    parser.add_argument('--vae_embedding_dim', type=int, default=1)
    #Dicretize actions configurations
    parser.add_argument('--kmeans_actions', type=bool, default=False)#TODO????? make this actually work , currently causes strange errors.
    parser.add_argument('--kmeans_action_centroids', type=int, default=128)#TODO make this actually work , currently causes strange errors.
    parser.add_argument('--discrete_rewards', type=bool, default=False)#TODO make this actually work, needs some way of mapping actions to specific indices
    
    #TODO make it so save fines are generated depending on parameters
    args = parser.parse_args()

    main(parameters=vars(args))