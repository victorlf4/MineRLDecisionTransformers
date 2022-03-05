from cgi import test
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

from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import Dataset
from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader
from minerl.data import BufferedBatchIter

import torchvision.datasets as datasets
# Parameters:
EPOCHS = 2  # how many times we train over dataset.
LEARNING_RATE = 0.0001  # Learning rate for the neural network.
BATCH_SIZE = 32
NUM_ACTION_CENTROIDS = 100  # Number of KMeans centroids used to cluster the data.

DATA_SAMPLES = 4000  # how many samples to use from the dataset. Impacts RAM usage

VALIDATION_SAMPLES = 400  # how many samples to use from the dataset. Impacts RAM usage

#TRAIN_MODEL_NAME = 'decisiontransformers_original_Kmeans_8x8x1vaeq'  # name to use when saving the trained agent.
TRAIN_MODEL_NAME = 'decisiontransformers_modified_Kmeans_8x8x1vaeq_128'  # name to use when saving the trained agent.
TEST_MODEL_NAME = 'research_potato.pth'  # name to use when loading the trained agent.
TRAIN_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when saving the KMeans model.
TEST_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when loading the KMeans model.



TEST_EPISODES = 10  # number of episodes to test the agent for.
MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamondVectorObf.

TRAIN=False
EVAL=False
log_to_wandb=False
DISCRETE_ACTIONS=True
DISCRETE_REWARDS=False

"""# Download the data"""
class MinerlDatasetSamples(IterableDataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self,transform):
        self.iterator = BufferedBatchIter(data)
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.iterator.buffered_batch_iter(batch_size=32, num_epochs=1)[index])
        else:
            return iter(self.iterator.buffered_batch_iter(batch_size=32, num_epochs=1)[index])

class MinerlImageIterator:
    def __init__(self,data):
        self.iterator = BufferedBatchIter(data)

    def __next__(self):
        dataset_observation, _, _, _, _ = next(self.iterator.buffered_batch_iter(batch_size=1, num_epochs=1))
        return dataset_observation["pov"]
    def __iter__(self):
        return self




class MinerlDatasetSamplesImages(IterableDataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self,data,transform):
        self.iterator = BufferedBatchIter(data)
        self.transform = transform

    def __iter__(self):
        dataset_observation, _, _, _, _ = next(self.iterator.buffered_batch_iter(batch_size=1, num_epochs=1))
        if self.transform:
            return self.transform(dataset_observation["pov"])
        else:
            return dataset_observation["pov"]



class MinerlDatasetImages(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    def __init__(self,data,numSamples,transform):
        self.data=data
        self.observations = []
        self.numSamples=numSamples
        self.transform = transform
        trajectory_names = self.data.get_trajectory_names()
        random.shuffle(trajectory_names)

        #get trajectories to train VAE
        # Add trajectories to the data until we reach the required DATA_SAMPLES.
        for trajectory_name in trajectory_names:
            trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
            for dataset_observation, dataset_action, _, _, _ in trajectory:
              self.observations.append(dataset_observation["pov"])
            if len(self.observations) >= self.numSamples:
                break

        random.shuffle(self.observations)
        self.observations = np.array(self.observations)
       
 
    

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.observations[index])
        else:
            return self.observations[index]

    def __len__(self):
        return len(self.observations)

    def getVariance(self):
        data_variance = np.var(self.observations/255.0)
        return data_variance


def getSamples():
    all_actions = []
    all_pov_obs = []
    all_rewards = []



    print("Loading data")
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)

    #get trajectories to train VAE
    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        for dataset_observation, dataset_action, dataset_reward, _, _ in trajectory:
            all_actions.append(dataset_action["vector"])
            all_pov_obs.append(dataset_observation["pov"])
            all_rewards.append(dataset_reward)
        if len(all_actions) >= DATA_SAMPLES:
            break

    all_actions = np.array(all_actions)
    random.shuffle(all_pov_obs)
    all_pov_obs = np.array(all_pov_obs)

    #train vq_vae
    #Separamos 
    #Cambiamos las imagenes de formato BHWC a BCHW (que espera pythorch ??? maybe no)  Batchsize Height Width Channels
    training_data= all_pov_obs[VALIDATION_SAMPLES:]#.transpose(0, 3, 1, 2)
    validation_data= all_pov_obs[:VALIDATION_SAMPLES]#.transpose(0, 3, 1, 2) 
    #data_variance = np.var(training_data/ 255.0)  
    data_variance = np.var(training_data/255.0)
    print(data_variance)
    return (all_actions,all_pov_obs,all_rewards)

def reward2go(rewards):
    rewards2go=[]
    remainingR=sum(rewards)

    for r in rewards:
        reward2go=remainingR-r
        rewards2go.append(reward2go)
        remainingR=reward2go
    return rewards2go

if log_to_wandb:
        wandb.init(
            name="test",
            group="test 2",
            project='decision-transformer'
        )
minerl.data.download(directory='data', environment='MineRLObtainDiamondVectorObf-v0')
data = minerl.data.make("MineRLObtainDiamondVectorObf-v0",  data_dir='data', num_workers=1)
#minerl.data.download(directory='data', environment='MineRLObtainIronPickaxeVectorObf-v0')
#data = minerl.data.make("MineRLObtainIronPickaxeVectorObf-v0",  data_dir='data', num_workers=1)


env = gym.make('MineRLObtainDiamondVectorObf-v0')

vae_model=vaeq("embedingdim_1",embedding_dim = 1,num_embeddings =65536,batch_size=16)
#vae_model=vaeq("test")
if TRAIN :
   
    observation_dataset=MinerlDatasetImages(data,DATA_SAMPLES-VALIDATION_SAMPLES,transforms.ToTensor())
    validation_dataset=MinerlDatasetImages(data,VALIDATION_SAMPLES,transforms.ToTensor())
    data_variance = observation_dataset.getVariance()
    training_loader = DataLoader(observation_dataset, #maybe self?
                                    batch_size=16, 
                                    shuffle=True,
                                    pin_memory=True)
    validation_loader = DataLoader(validation_dataset, #maybe self?
                                    batch_size=32, 
                                    shuffle=True,
                                    pin_memory=True)

    vae_model.train(training_loader,data_variance)
    vae_model.plot()
    vae_model.eval(validation_loader)
    
else :
    vae_model.load()

if EVAL:
    validation_dataset=MinerlDatasetImages(data,VALIDATION_SAMPLES,transforms.ToTensor())
    validation_loader = DataLoader(validation_dataset, #maybe self?
                                    batch_size=16, 
                                    shuffle=True,
                                    pin_memory=True)
    vae_model.eval(validation_loader)   
    vae_model.plot()

#vae_model.showEmbedding()

all_actions = []
all_pov_obs = []
all_rewards = []
all_done = []
#probando primero con 1 trayectoria
trajectory_names = data.get_trajectory_names()
random.shuffle(trajectory_names)
trajectory = data.load_data(trajectory_names[0], skip_interval=0, include_metadata=False)
for dataset_observation, dataset_action, dataset_reward, _, done in trajectory:
    all_actions.append(dataset_action["vector"])
    data_obs= dataset_observation["pov"].transpose(2, 0, 1)
    obs_encoding_tensor=vae_model.quantizeSingle(th.tensor(data_obs,dtype=th.float32)).flatten()#encode or quantize?
    all_pov_obs.append(obs_encoding_tensor.cpu().t())
    all_rewards.append(dataset_reward)
    all_done.append(done)
    
all_actions = np.array(all_actions)
all_pov_obs=th.stack(all_pov_obs)
all_rewards = np.array(all_rewards)
all_done = np.array(all_done)
print("Running KMeans on the action vectors")
kmeans = KMeans(n_clusters=NUM_ACTION_CENTROIDS)
kmeans.fit(all_actions)
action_centroids = kmeans.cluster_centers_
print("KMeans done")#TODO necesito guardar el resultado de kmeans para solo hacerlo 1 vez sobre muchas acciones maybe junto al vqvae
#convirtiendo una trayectoria a una sequencia(rewardsFuturos,estado,Accion,rewardsFuturos2,estado2.accion2...) 

#all_pov_obs=th.cat(all_pov_obs)
#Prepocesamos como en la baseline
# calculamos las distancias a los centroides
# "None" in indexing adds a new dimension that allows the broadcasting 
distancias = np.sum((all_actions - action_centroids[:, None]) ** 2, axis=2)
trajectory_actions = np.argmin(distancias, axis=0)


#maybe usar nn.embeding para los embedings de los rewards y acciones
#Pasamos a tensores todos los 
all_rewards=th.from_numpy(all_rewards)
rewardsTogo =reward2go(all_rewards)
trajectory_actions=th.from_numpy(trajectory_actions)

rewardsTogo=th.tensor(rewardsTogo)
if DISCRETE_REWARDS is not True:
    all_rewards=th.unsqueeze(all_rewards,1)
    rewardsTogo=th.unsqueeze(rewardsTogo,1)
#trajectory_actions=th.unsqueeze(trajectory_actions,1)#???maybe doest even make sense here

'''
#all_pov_obs=th.unsqueeze(all_pov_obs,1)
print(trajectory_actions.shape)
print(rewardsTogo.shape)
print(all_pov_obs.shape)
#print([rewardsTogo,all_pov_obs,trajectory_actions])
print(rewardsTogo)
print(trajectory_actions)
'''
print("data augmented")
#decision transformer part
from Model import decision_transformer,decision_transformer_modified ,seq_trainer
if DISCRETE_ACTIONS:
    act_dim=1
else:
    act_dim=trajectory_actions.shape[1]
obs_dim=all_pov_obs.shape[1]

max_ep_len=10000
max_length=20
max_iters=20
num_steps_per_iter=1
batch_size=64
#TODO use gpu again once it works
#device = th.device("cuda" if th.cuda.is_available() else "cpu")
device=th.device("cpu")

if DISCRETE_REWARDS :
    discrete_rewards=1095
else:
    discrete_rewards=None


state_mean, state_std = th.mean(all_pov_obs.to(dtype=th.float32), axis=0).numpy(), th.std(all_pov_obs.to(dtype=th.float32), axis=0).numpy() + 1e-6
'''
model = decision_transformer.DecisionTransformer(
            state_dim=obs_dim,
            act_dim=act_dim,
            max_length=max_length,
            max_ep_len=max_ep_len,
            hidden_size=128,
            n_layer=3,
            n_head=1,
            n_inner=4*128,
            activation_function="relu",
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,)

'''

model = decision_transformer_modified.DecisionTransformer(
            state_dim=obs_dim,
            act_dim=act_dim,
            max_length=max_length,
            max_ep_len=max_ep_len,
            discrete_rewards=None,#1095,#1095 is the number of posible reward combinations in the minerl enviroment, though most are very unlikely
            discrete_actions=NUM_ACTION_CENTROIDS,
            discrete_states=65536,
            hidden_size=256,
            n_layer=3,
            n_head=1,
            n_inner=4*256,
            activation_function="relu",
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,)


model = model.to(device=device)           
optimizer = th.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

#all_pov_obs=list(all_pov_obs)
#trajectory_actions=list(trajectory_actions)
#rewardsTogo=list(rewardsTogo)
timestamped_steps =	{
  "observations": all_pov_obs,
  "trajectory_actions": trajectory_actions,
  "rewards": all_rewards,
  "done": all_done,
  "rewards2go": rewardsTogo,
  "timesteps": range(all_pov_obs.shape[0]),
}
#timestamped_steps =np.concatenate([all_pov_obs,trajectory_actions,rewardsTogo,np.expand_dims(all_done,1),np.expand_dims(range(all_pov_obs.shape[0]),1)], axis=1, out=None, dtype=None)#añadimos el range como tiemstamp del step
#print("test")
#print(timestamped_steps.shape)
print("teast action")
state_dim=all_pov_obs.shape[1]

def get_batch(batch_size=64, max_len=max_length):
    obsBatch=[]
    actionBatch=[]
    rewardsBatch=[]
    doneBatch=[]
    rtgBatch=[]
    timesteps=[]
    mask =[]

    for i in range(batch_size):
      inicio = random.randint(0, timestamped_steps["observations"].shape[0] - 1)
      #step =timestamped_steps[inicio:max_length]
      #We feed  max_length timesteps into Decision Transformer, for a total of 3*max_length tokens 
      trajectory_lenght=timestamped_steps["observations"][inicio:inicio+max_length].shape[0]
      obsBatch.append(timestamped_steps["observations"][inicio:inicio+max_length])
      actionBatch.append(timestamped_steps["trajectory_actions"][inicio:inicio+max_length])
      rewardsBatch.append(timestamped_steps["rewards"][inicio:inicio+max_length])
      doneBatch.append(timestamped_steps["done"][inicio:inicio+max_length])
      rtgBatch.append(timestamped_steps["rewards2go"][inicio:inicio+max_length])
      timesteps.append(timestamped_steps["timesteps"][inicio:inicio+max_length])
      #mask.append(np.concatenate([np.zeros((1, max_len - trajectory_lenght)), np.ones((1, trajectory_lenght))], axis=1))#create mask 0 in the padding
      # add padding to the right
      state_dim=obsBatch[-1].shape[1]#shape of obs encoding
     
      obsBatch[-1] = np.concatenate([np.zeros((max_len - trajectory_lenght, state_dim)), obsBatch[-1]], axis=0)
      if DISCRETE_ACTIONS:
            actionBatch[-1] = np.concatenate([np.ones(max_len - trajectory_lenght) * -10., actionBatch[-1]], axis=0)
      else:
            actionBatch[-1] = np.concatenate([np.ones(( max_len - trajectory_lenght, act_dim)) * -10., actionBatch[-1]], axis=0)
      if DISCRETE_REWARDS:
            rewardsBatch[-1] = np.concatenate([np.zeros(( max_len - trajectory_lenght)), rewardsBatch[-1]], axis=0)
      else:
            rewardsBatch[-1] = np.concatenate([np.zeros(( max_len - trajectory_lenght, 1)), rewardsBatch[-1]], axis=0)
      doneBatch[-1] = np.concatenate([np.ones(( max_len - trajectory_lenght)) * 2, doneBatch[-1]], axis=0)#adds 2 as padding ? maybe so its not 1 or 0
      if DISCRETE_REWARDS:
            rtgBatch[-1] = np.concatenate([np.zeros(( max_len - trajectory_lenght)), rtgBatch[-1]], axis=0)
      else:
            rtgBatch[-1] = np.concatenate([np.zeros(( max_len - trajectory_lenght, 1)), rtgBatch[-1]], axis=0)
      #rtg[-1] = np.concatenate([np.zeros((1, max_len - timestamped_steps, 1)), rtg[-1]], axis=1) 
      timesteps[-1] = np.concatenate([np.zeros(( max_len - trajectory_lenght)), timesteps[-1]], axis=0)
      mask.append(np.concatenate([np.zeros((max_len - trajectory_lenght)), np.ones(( trajectory_lenght))], axis=0))
    '''
    obsBatch = th.from_numpy(np.concatenate(obsBatch, axis=0)).to(dtype=th.float32, device=device)
    actionBatch = th.from_numpy(np.concatenate(actionBatch, axis=0)).to(dtype=th.float32, device=device)
    rewardsBatch = th.from_numpy(np.concatenate(rewardsBatch, axis=0)).to(dtype=th.float32, device=device)
    doneBatch = th.from_numpy(np.concatenate(doneBatch, axis=0)).to(dtype=th.long, device=device)
    rtgBatch = th.from_numpy(np.concatenate(rtgBatch, axis=0)).to(dtype=th.float32, device=device)
    timesteps = th.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=th.long, device=device)
    mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
    #mask=th.tensor(mask).to(device=device)
    '''
    '''
        obsBatch = th.tensor(obsBatch).to(dtype=th.float32, device=device)
        actionBatch = th.tensor(actionBatch).to(dtype=th.float32, device=device)
        rewardsBatch = th.tensor(rewardsBatch).to(dtype=th.float32, device=device)
        doneBatch = th.tensor(doneBatch).to(dtype=th.long, device=device)
        rtgBatch = th.tensor(rtgBatch).to(dtype=th.float32, device=device)
        timesteps = th.tensor(timesteps).to(dtype=th.long, device=device)
        mask = th.tensor(mask).to(device=device)
        #mask=th.tensor(mask).to(device=device)
        return obsBatch,actionBatch,rewardsBatch,doneBatch ,rtgBatch,timesteps,mask
    '''
    obsBatch = th.tensor(obsBatch).to(dtype=th.float32, device=device)#change to int only if discrete actions
    actionBatch = th.tensor(actionBatch).to(dtype=th.int32, device=device)
    if DISCRETE_REWARDS:#TODO maybe change to chooseing dtype for better redeablity
        rewardsBatch = th.tensor(rewardsBatch).to(dtype=th.int32, device=device)
    else:
        rewardsBatch = th.tensor(rewardsBatch).to(dtype=th.float32, device=device)
    
    doneBatch = th.tensor(doneBatch).to(dtype=th.long, device=device)
    rtgBatch = th.tensor(rtgBatch).to(dtype=th.int32, device=device)
    timesteps = th.tensor(timesteps).to(dtype=th.long, device=device)
    mask = th.tensor(mask).to(device=device)
    #mask=th.tensor(mask).to(device=device)
    return obsBatch,actionBatch,rewardsBatch,doneBatch ,rtgBatch,timesteps,mask
print("batch")
#print(get_batch())
#print(get_batch()[0].shape)
#timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
#timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
#????????

warmup_steps=1
num_eval_episodes=1
env_targets = [547]#target rewards for

scheduler = th.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )




from Model.evaluate_episodes import evaluate_episode_rtg
mode="delayed"#?? says normal for standard setting, delayed for sparse sso i guess delayed cause diamond is sparse kind of
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
                            scale=1,#1 cause i dont want scale , amybe even remove any scalling directly and treat reward as tokens
                            target_return=target_rew/1,
                            mode=mode,
                            state_mean=state_mean,#reemplazar en codigo? dado que no paso estados
                            state_std=state_std,
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

trainer = seq_trainer.SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: th.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],#eval model for each target reward
        )
def save(epoch,checkpoint_file):
        th.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            },checkpoint_file)
def load(validation_data=None):
            checkpoint = th.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
checkpoint_file ="./models/"+TRAIN_MODEL_NAME
if os.path.exists(checkpoint_file):
    print("loading save")
    load()


for iter in range(max_iters):
        outputs = trainer.train_iteration(num_steps=num_steps_per_iter, iter_num=iter+1, print_logs=True)
        save(iter,checkpoint_file)
        if log_to_wandb:
            wandb.log(outputs)