from pickle import TRUE
import random
import numpy as np
import torch as th
from torch import nn
import gym
import minerl
from sklearn.cluster import KMeans
from  Model.vq_vae import VectorQuantizerVAE as vq_vae
import os
import wandb
import argparse
from Model.multimodal_decision_transformer import DecisionTransformer
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

#TODO remake batch norm scaling
def main(parameters):              
        #loading parameters
        use_checkpoint=parameters['use_checkpoint']
        startingIter=0#Number of the first iteration of the model, used to load current epoch from checpoint.
        batch_size = parameters['batch_size']
        num_eval_episodes = parameters['num_eval_episodes']
        discrete_actions = parameters['kmeans_actions']
        kmeans_action_centroids=parameters['kmeans_action_centroids']
        discrete_rewards =False#parameters['discrete_rewards']TODO not implemented
        pov_encoder=parameters['pov_encoder']
        vectorize_actions=parameters['vectorize_actions']
        state_vector=parameters['state_vector']
        validation_steps=parameters['validation_steps']
        if pov_encoder == "vq_vae":
                discrete_states=parameters["vae_embedings"]
        else:
                discrete_states=None
        
        log_to_wandb=parameters['log_to_wandb']
        max_length=parameters['K']
        max_ep_len=parameters['max_ep_len']
        max_ep_len_dataset=parameters['max_ep_len_dataset']
        buffer_target_size=parameters['buffer_target_size']
        buffer_target_size_validation=parameters['buffer_target_size_validation']
        validation_trajectories=parameters['validation_trajectories']
        warmup_steps = parameters['warmup_steps']
        visualize=parameters['visualize']
        
        num_steps_per_iter=parameters['num_steps_per_iter']
        max_iters=parameters['max_iters']
        checkpoint_file ="./models/"+parameters['model_name']
        mode = parameters['mode']
        env_targets = parameters["target_rewards"]
        if pov_encoder == "vq_vae":
                pov_dim=(256*parameters["vae_embedding_dim"],) 
                convolution_head= False
        elif pov_encoder == "cnn":
                pov_dim=(64,64,3)
                convolution_head= True 
        else:
                pov_dim=(3*64*64,)
                convolution_head= False
        if state_vector:
                state_dim= (64,) #TODO calcular correctamente
        else:
                state_dim = None

        if parameters['device']=="cuda":
                device = th.device("cuda" if th.cuda.is_available() else "cpu")
                if device == "cpu":
                        print("cuda device not avaliable,using cpu")
        else:
                device = th.device(parameters['device'])
        if pov_encoder == "vq_vae":
                vae_model=vq_vae(parameters["vae_model"],embedding_dim = parameters["vae_embedding_dim"],num_embeddings =parameters["vae_embedings"],device_name=parameters["device"],batch_size=32)
                vae_model.load()
        else:
                vae_model=None
        #Load minecraft env and dataset
        verify_or_download_dataset(directory='data', environment=parameters["dataset"])
        data = minerl.data.make(parameters["dataset"],  data_dir='data', num_workers=4)
        
        def load_env(env_name):
                data_enviroment=None
                if vectorize_actions:
                        spec_pipeline =  minerl.data.make(env_name,  data_dir='data')
                        enviroment=spec_pipeline.spec
                        enviroment=Vectorized(enviroment)
                        enviroment.register()
                        data_enviroment=Vectorized(data.spec)
                        env_name=enviroment._update_name(env_name)
                
                
                enviroment = gym.make(env_name)
                if parameters['record']:
                        enviroment=Monitor(enviroment,"./video",force=True)#TODO fix some bugs whith not doing the steps at the same time
                
                return enviroment,data_enviroment
        env,data_enviroment=load_env(parameters["env"])

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
                return action_centroids.to(device)
        act_dim = env.action_space["vector"].shape[0]
        

        if discrete_actions:
                if os.path.exists(checkpoint_file+"_kmeans"):
                        action_centroids=load_kmeans()
                else:
                        action_iterator =MinerlActionIterator(data)
                        action_samples=[]
                        for i in range(100000):
                                action_samples.append(next(action_iterator))
                        print("Running KMeans on the action vectors")
                        kmeans = KMeans(n_clusters=kmeans_action_centroids)
                        kmeans.fit(action_samples)
                        action_centroids = th.tensor(kmeans.cluster_centers_,dtype=th.float32).to(device)
                        print("KMeans done")
                        save_kmeans()
        else:
                action_centroids=None
        if parameters["dataset_validation"]==parameters["dataset"]:
                trajectory_names = data.get_trajectory_names()
                trajectory_names_train=trajectory_names[validation_trajectories:]
                trajectory_names_validate=trajectory_names[:validation_trajectories]
                print("num_training trajectories:"+str(len(trajectory_names_train)))
                trajectory_buffer=BufferedTrajectoryIter(data,all_trajectories=trajectory_names_train,buffer_target_size=buffer_target_size,sequence_size=max_length,reward_to_go=TRUE,max_ep_len_dataset=max_ep_len_dataset,store_rewards2go= parameters["store_rewards2go"])#TODO store_rewards2go=parameters["store_rewards2go"])
                trajectory_buffer_validation=BufferedTrajectoryIter(data_validation,all_trajectories=trajectory_names_validate,buffer_target_size=buffer_target_size_validation,sequence_size=max_length,reward_to_go=TRUE,max_ep_len_dataset=max_ep_len_dataset,store_rewards2go=parameters["store_rewards2go"])
                
        else:        
                trajectory_buffer=BufferedTrajectoryIter(data,buffer_target_size=buffer_target_size,sequence_size=max_length,reward_to_go=TRUE,max_ep_len_dataset=max_ep_len_dataset,store_rewards2go=parameters["store_rewards2go"])
                trajectory_buffer_validation=BufferedTrajectoryIter(data_validation,buffer_target_size=buffer_target_size_validation,sequence_size=max_length,reward_to_go=TRUE,max_ep_len_dataset=max_ep_len_dataset,store_rewards2go=parameters["store_rewards2go"])
        trajectory_buffer_iter=trajectory_buffer.buffered_batch_iter(batch_size,num_batches=(num_steps_per_iter*max_iters)+warmup_steps)        
        trajectory_buffer_iter_validation=trajectory_buffer_validation.buffered_batch_iter(batch_size,num_batches=(validation_steps*max_iters))#TODO make this dependent on num of validation iterations
                
                

        def get_batch(batch_size=64, max_len=max_length, validation=False):
                actionBatch=[]
                obsBatch=[]
                stateBatch=[]
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
                                sequence=minerlEncodeSequence(trajectory,device,discrete_rewards=discrete_rewards,vae_model=vae_model,vectorize_actions=vectorize_actions,state_vector=state_vector,data_enviroment=data_enviroment)
                                #We feed  max_length timesteps into Decision Transformer, for a total of 3*max_length tokens 
                                sequence_lenght=sequence["observations_pov"].shape[0]
                                obsBatch.append(sequence["observations_pov"])
                                if state_vector:
                                        stateBatch.append(sequence["observations_vector"])
                                actionBatch.append(sequence["actions"])
                                rewardsBatch.append(sequence["rewards"])
                                doneBatch.append(sequence["done"])
                                rtgBatch.append(sequence["rewards2go"])
                                timesteps.append(sequence["timesteps"])
                                if pov_encoder == "vq_vae":
                                        pov_dim=(obsBatch[-1].shape[1],)#shape of obs encoding
                                else:
                                        pov_dim=(obsBatch[-1].shape[1],obsBatch[-1].shape[2],obsBatch[-1].shape[3])#shape of images
                                obsBatch[-1] = np.concatenate([np.zeros((max_len - sequence_lenght,)+pov_dim), obsBatch[-1]], axis=0)#+concantenates tuples here
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
                                if state_vector:                
                                        stateBatch[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght, state_dim[0])) , stateBatch[-1]], axis=0)
                                        
                                        
                                
                                timesteps[-1] = np.concatenate([np.zeros(( max_len - sequence_lenght)), timesteps[-1]], axis=0)
                                mask.append(np.concatenate([np.zeros((max_len - sequence_lenght)), np.ones((sequence_lenght))], axis=0))
                
                if pov_encoder == "linear":
                        obsBatch = th.tensor(np.array(obsBatch)).to(dtype=th.float32, device=device).divide(256).flatten(start_dim=2)
                elif pov_encoder == "cnn":
                         obsBatch = th.tensor(np.array(obsBatch)).to(dtype=th.float32, device=device).divide(256)
                elif pov_encoder == "vq_vae":
                        obsBatch = th.tensor(np.array(obsBatch)).to(dtype=th.float32, device=device) 
               
                if state_vector:
                        stateBatch= th.tensor(np.array(stateBatch)).to(dtype=th.float32, device=device)
                else:
                        stateBatch=None  
   
                actionBatch = th.tensor(actionBatch).to(dtype=th.float32, device=device)
                rewardsBatch = th.tensor(rewardsBatch).to(dtype=th.float32, device=device)
                rtgBatch = th.tensor(rtgBatch).to(dtype=th.float32, device=device)   
                timesteps = th.tensor(timesteps).to(dtype=th.long, device=device)
                mask = th.tensor(mask).to(device=device)
                return obsBatch,stateBatch,actionBatch,rewardsBatch,doneBatch ,rtgBatch,timesteps,mask
        model = DecisionTransformer(
            pov_dim=pov_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=max_length,
            max_ep_len=max_ep_len_dataset,#diferent from max_ep_len because theres a bug where the dataset has trajectories bigger than the maximun episode lenght of the enviroment
            discrete_rewards=None,
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
                                                pov_dim,
                                                act_dim,
                                                model,
                                                vae_model,
                                                state_vector_dim=state_dim,
                                                action_centroids=action_centroids,
                                                max_ep_len=max_ep_len,
                                                scale=1,#TODO check if reward scalling would make sense
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
                        }
                return fn
        def validation_fn(model):
                        returns = []
                       
                        with th.no_grad(): 
                                ret=evaluate_validation_rtg(
                                env,
                                pov_dim,
                                act_dim,
                                model,
                                get_batch,
                                batch_size,
                                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: th.mean((a_hat - a)**2),
                                validation_batches=validation_steps,
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
                        name=f'{parameters["group_name"]}-{parameters["model_name"]}',
                        group=parameters["group_name"],
                        project='decision-transformer_TFM',
                        config=parameters)

        def save(epoch,checkpoint_file):#TODO maybe merge whith kmeans

                th.save({
                'epoch': epoch,
                'steps': trainer.total_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },checkpoint_file)
                wandb.save(checkpoint_file)

        
        def load(validation_data=None):
                checkpoint = th.load(checkpoint_file)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                iter=checkpoint['epoch']
                trainer.total_steps=checkpoint['steps']
                return iter
                

        if os.path.exists(checkpoint_file) and use_checkpoint:
                print("Loading saved decision transformer model")#TODO make it so weight and biases continues in the same run if possible
                startingIter = load()
        
        
        for iteration in range(startingIter,startingIter+max_iters):
                validate = parameters["num_validation_iters"]!=0 and ((iteration+1)%parameters["num_validation_iters"]) != 0

                outputs = trainer.train_iteration(num_steps=num_steps_per_iter, iter_num=iteration+1, print_logs=True,validation=validate)
                if(use_checkpoint):#TODO make a separate variable for saving checkpoints to use checkpoint whithout saving
                        save(iteration,checkpoint_file)
                if log_to_wandb:
                        wandb.log(outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='normal')#not usefull currently
    #General parameters    
    parser.add_argument('--max_iters', type=int, default=100,help="Number of iterations to execute during training")
    parser.add_argument('--num_steps_per_iter', type=int, default=100,help="Number of batches in an iteration")
    parser.add_argument('--num_validation_iters', type=int, default=10,help="Number of validation iterantions before running minecraft")#num of validation iterations before running the minerl env.
    parser.add_argument('--device', type=str, default='cuda',help="Device we train on read pytorch documentation for more info")
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False,help="logging on weight and biases, requires weight and biases account")
    parser.add_argument('--group_name','-g' , type=str, default="tfm",help="wandb group name")
    parser.add_argument('--use_checkpoint', type=bool, default=False,help="If true saves the model each iteration")#TODo eliminate and replace whith just havign a checkpooint name
    parser.add_argument('--model_name', type=str, default="decisiontransformers_convolution",help="Name of the model used in the checkpoint file and in wandb")
    #Enviroment and Data parameters
    parser.add_argument('--env', type=str, default='MineRLObtainDiamondVectorObf-v0', help="MineRl enviroment the model will be evaluated in")
    parser.add_argument('--vectorize_actions' , type=bool, default=False,help="Necesary to train and evaluate on Basalt envs and datasets")#TODO make this automatic
    parser.add_argument('--visualize' , type=bool, default=False,help="MineRl enviroment the model will be evaluated in")
    parser.add_argument('--record' , type=bool, default=False,help="records video, currently only working correctly on evaluate_model")
    parser.add_argument('--max_ep_len', type=int, default=18000,help="max lenght of an evaluation episode in frames")#default of the diamond env
    parser.add_argument('--max_ep_len_dataset', type=int, default=65536,help="Maximun lenght of a trajectory in the dataset in frames,affects temporal encoder size")
    parser.add_argument('--dataset', type=str, default='MineRLObtainDiamondVectorObf-v0',help="Dataset used for training")
    parser.add_argument('--dataset_validation', type=str, default='MineRLObtainIronPickaxeVectorObf-v0',help="Dataset used for validation")
    parser.add_argument('--buffer_target_size', type=int, default=3000,help="size of the buffer used for loading from the training dataset")
    parser.add_argument('--buffer_target_size_validation', type=int, default=3000,help="size of the buffer used for loading from the  validation dataset")
    parser.add_argument('--store_rewards2go', type=bool, default=False,help="stores calculated reward to go on ram")
    #Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.02)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=100)
    #Evaluation parameters
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--validation_steps', type=int, default=10)
    parser.add_argument('--validation_trajectories', type=int, default=5)
    parser.add_argument('--target_rewards',type=int, nargs='+', default=[1571])#Accepts multiple imputs     
    #Model parameters
    parser.add_argument('--embed_dim', type=int, default=128,help="dimension of the mbedding of each token")
    parser.add_argument('--n_layer', type=int, default=3,help="number of layers of the transformer model")
    parser.add_argument('--n_head', type=int, default=1,help="number of attention heads of the transformer model")
    parser.add_argument('--K', type=int, default=20 ,help="context window of the model in frames actual size in tokens is x3 or x4 dependin on whether you use state tokens")
    parser.add_argument('--pov_encoder', type=str, default="linear",choices=["linear","cnn","vq_vae"])
    parser.add_argument('--state_vector', type=bool, default=False,help="if true adds encoder for state vector to the model")
    parser.add_argument('--activation_function', type=str, default='relu',help="activation function used in the model")
    #VQ_VAE parameters
    parser.add_argument('--vae_model', type=str, default="embedingdim_1")
    parser.add_argument('--vae_embedings', type=int, default=65536)
    parser.add_argument('--vae_embedding_dim', type=int, default=1)
    #Dicretize actions parameters
    parser.add_argument('--kmeans_actions', type=bool, default=False,help="if true runs kmeans the actions to discretize the action space")
    parser.add_argument('--kmeans_action_centroids', type=int, default=128,help="number of action centroids on the if kmeans actions is used")
    #parser.add_argument('--discrete_rewards', type=bool, default=False)#TODO Not implemented
    
    args = parser.parse_args()

    main(parameters=vars(args))