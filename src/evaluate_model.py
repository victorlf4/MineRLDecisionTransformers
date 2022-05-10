import gym
import numpy as np
import minerl
from utils.enviroment_utils import load_env
import torch as th
import argparse
from evaluation.evaluate_episodes import evaluate_episode_rtg

def main(parameters):
        checkpoint_file ="./models/"+parameters['checkpoint_name']
        num_eval_episodes=10
        dataset_name=parameters["env"]
        env,_=load_env(dataset_name,record=parameters["record"],vectorize_actions=parameters["vectorize_actions"])
        device="cuda"
        act_dim=env.action_space["vector"].shape[0]
        vae_model=None
        action_centroids=None
        max_ep_len=4500
        mode="normal"
        env_targets=[64]
        visualize=True
        #model_params
        max_length=20
        max_ep_len_dataset=4096
        discrete_states=None
        pov_encoder =parameters["pov_encoder"]
        natureCNN= pov_encoder == "cnn"
        if pov_encoder == "vq_vae":
                state_dim=(256*parameters["vae_embedding_dim"],) #TODO actually figure out why its 256 
                convolution_head= False
        elif pov_encoder == "cnn":
                state_dim=(64,64,3)
                convolution_head= True 
        else:
                state_dim=(3*64*64,)
                convolution_head= False



        from Model.decision_transformer import DecisionTransformer
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

        def load():
                checkpoint = th.load(checkpoint_file)
                model.load_state_dict(checkpoint['model_state_dict'])
        load()
        print(list(model.parameters()))  
        eval_fns=[eval_episodes(tar) for tar in env_targets]     
        def eval_episodes(target_rew):
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
        #eval_fns=[eval_episodes(tar) for tar in env_targets]

        for eval_fn in eval_fns:
                outputs = eval_fn(model)
                for k, v in outputs.items():
                        print(f'evaluation/{k}:{v}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MineRLObtainDiamondVectorObf-v0')
    parser.add_argument('--dataset', type=str, default='MineRLObtainDiamondVectorObf-v0')
    parser.add_argument('--dataset_validation', type=str, default='MineRLObtainIronPickaxeVectorObf-v0')
    parser.add_argument('--validation_steps', type=int, default=10)
    #parser.add_argument('--validation_trajectories', type=str, default=5)#if the validation dataset is the same as the train dataset separate a fraction of it?TODO(maybe do this at the dataset level instead manually)
    parser.add_argument('--target_rewards', nargs='+', default=[64])#Accepts multiple imputs#TODO!!! fix bug where it interpres this as an int
    parser.add_argument('--vectorize_actions' , type=bool, default=False)#TODO make this automatic
    parser.add_argument('--visualize' , type=bool, default=False)
    parser.add_argument('--record' , type=bool, default=False)
    parser.add_argument('--max_ep_len', type=int, default=18000)#default of the diamond env
    parser.add_argument('--max_ep_len_dataset', type=int, default=65536)#nice round number thats almost
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)#TODO maybe rename or change how it works to have diferent sizes of embedings for rewards
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
    parser.add_argument('--use_checkpoint', type=bool, default=False)#TODo eliminate and replace whith just havign a checkpooint name
    parser.add_argument('--checkpoint_name', type=str, default="decisiontransformers_convolution")
    parser.add_argument('--pov_encoder', type=str, default="linear")
    parser.add_argument('--state_vector', type=bool, default=False)
    #VQ_VAE configurations
    parser.add_argument('--vae_model', type=str, default="embedingdim_1")
    parser.add_argument('--vae_embedings', type=int, default=65536)
    parser.add_argument('--vae_embedding_dim', type=int, default=1)
    #Dicretize actions configurations
    parser.add_argument('--kmeans_actions', type=bool, default=False)#TODO????? make this actually work , currently causes strange errors.
    parser.add_argument('--kmeans_action_centroids', type=int, default=128)#TODO make this actually work , currently causes strange errors.
    parser.add_argument('--discrete_rewards', type=bool, default=False)#TODO make this actually work, ne
    #TODO make it so save fines are generated depending on parameters
    args = parser.parse_args()
    parameters=vars(args)
    main(parameters)
