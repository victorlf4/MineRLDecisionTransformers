import gym
import numpy as np
import minerl
from regex import F
from utils.enviroment_utils import load_env
import torch as th
import argparse
from evaluation.evaluate_episodes import evaluate_episode_rtg
import os
from  Model.vq_vae import VectorQuantizerVAE as vq_vae

def main(parameters):
        checkpoint_file ="./models/"+parameters['model_name']
        num_eval_episodes=parameters["num_eval_episodes"]
        dataset_name=parameters["env"]
        env,_=load_env(dataset_name,record=parameters["record"],vectorize_actions=parameters["vectorize_actions"],collab=parameters["using_collab"])
        using_collab=parameters['using_collab'] and parameters['record']
        device=parameters["device"]
        act_dim=env.action_space["vector"].shape[0]
       

        def load_kmeans():
                checkpoint_kmeans = th.load(checkpoint_file+"_kmeans")
                action_centroids=checkpoint_kmeans['action_centroids']
                return action_centroids.to(device)
        if (parameters["kmeans_actions"]):
                action_centroids=load_kmeans()
        else:
                action_centroids=None
        #if os.path.exists(checkpoint_file+"_kmeans"):
       
  

        max_ep_len=parameters["max_ep_len"]
        mode="normal"
        env_targets=parameters["target_rewards"]
        visualize=True
        #model_params
        max_length=parameters["K"]
        max_ep_len_dataset=parameters["max_ep_len_dataset"]
        discrete_states=None
        pov_encoder =parameters["pov_encoder"]
        natureCNN= pov_encoder == "cnn"
        if pov_encoder == "vq_vae":
                vae_model=vq_vae(parameters["vae_model"],embedding_dim = parameters["vae_embedding_dim"],num_embeddings =parameters["vae_embedings"],device_name=parameters["device"],batch_size=32)
                vae_model.load()
        else:
                vae_model=None

        if pov_encoder == "vq_vae":
                pov_dim=(256*parameters["vae_embedding_dim"],) #TODO actually figure out why its 256 
                convolution_head= False
        elif pov_encoder == "cnn":
                pov_dim=(64,64,3)
                convolution_head= True 
        else:
                pov_dim=(3*64*64,)
                convolution_head= False
        if parameters["state_vector"]:
                state_dim=(64,)
        else:
                state_dim=None


        from Model.multimodal_decision_transformer import DecisionTransformer
        model = DecisionTransformer(
                pov_dim=pov_dim,
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=max_length,
                max_ep_len=max_ep_len_dataset,#diferent from max_ep_len because theres a bug where the dataset has trajectories bigger than the maximun episode lenght of the enviroment
                discrete_rewards=None,#TODO not implemented
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

        def load():
                checkpoint = th.load(checkpoint_file)
                model.load_state_dict(checkpoint['model_state_dict'])
        load()
        from utils.calculate_parameters import count_parameters
        print(count_parameters(model))  
             
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
                                                visualize=visualize,
                                                collab=using_collab
                                                )
                                returns.append(ret)
                                lengths.append(length)
                        return {
                                f'target_{target_rew}_return_mean': np.mean(returns),
                                f'target_{target_rew}_return_std': np.std(returns),
                                f'target_{target_rew}_return_max': np.max(returns),
                        }
                return fn
        eval_fns=[eval_episodes(tar) for tar in env_targets]

        for eval_fn in eval_fns:
                outputs = eval_fn(model)
                for k, v in outputs.items():
                        print(f'evaluation/{k}:{v}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    #General parameters
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default="linear_k20_l3_h1_diamond_tfm")
    parser.add_argument('--using_collab','-c', type=bool, default=False,help="Uses the collab gym renderer to render the evaluation")
    #Enviroment parameters
    parser.add_argument('--env', type=str, default='MineRLObtainDiamondVectorObf-v0')
    parser.add_argument('--vectorize_actions' , type=bool, default=False)#TODO make this automatic
    parser.add_argument('--visualize' , type=bool, default=False)
    parser.add_argument('--record' , type=bool, default=False)
    parser.add_argument('--max_ep_len', type=int, default=18000)#default of the diamond env
    parser.add_argument('--max_ep_len_dataset', type=int, default=65536)#nice round number thats almost
    #Training parameters
    parser.add_argument('--dropout', type=float, default=0.1)
    #Evaluation parameters
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--target_rewards',type=int, nargs='+', default=[1571,547,67,0])#Accepts multiple imputs#TODO!!! fix bug where it interprets this as an int       
    #Model parameters
    parser.add_argument('--embed_dim', type=int, default=128)#TODO fix this so it doest crash whenever its not 128
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pov_encoder', type=str, default="linear")
    parser.add_argument('--state_vector', type=bool, default=False)
    parser.add_argument('--activation_function', type=str, default='relu')
    #VQ_VAE parameters
    parser.add_argument('--vae_model', type=str, default="embedingdim_1")
    parser.add_argument('--vae_embedings', type=int, default=65536)
    parser.add_argument('--vae_embedding_dim', type=int, default=1)
    #Dicretize actions parameters
    parser.add_argument('--kmeans_actions', type=bool, default=False)
    #parser.add_argument('--discrete_rewards', type=bool, default=False)#TODO not implemented
    args = parser.parse_args()
    parameters=vars(args)
    main(parameters)
