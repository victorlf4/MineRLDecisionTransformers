import random
import numpy as np
import torch as th
from torch import nn
import gym
import minerl
from sklearn.cluster import KMeans
from torch.utils.data.dataset import IterableDataset
from transformers import pipeline
from transformers import AutoTokenizer
from  Model.vq_vae import VectorQuantizerVAE as vaeq
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

DATA_SAMPLES = 40000  # how many samples to use from the dataset. Impacts RAM usage

VALIDATION_SAMPLES = 400  # how many samples to use from the dataset. Impacts RAM usage

TRAIN_MODEL_NAME = 'research_potato.pth'  # name to use when saving the trained agent.
TEST_MODEL_NAME = 'research_potato.pth'  # name to use when loading the trained agent.
TRAIN_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when saving the KMeans model.
TEST_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when loading the KMeans model.

TEST_EPISODES = 10  # number of episodes to test the agent for.
MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamondVectorObf.

TRAIN=False
EVAL=False


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



minerl.data.download(directory='data', environment='MineRLObtainIronPickaxeVectorObf-v0')
data = minerl.data.make("MineRLObtainIronPickaxeVectorObf-v0",  data_dir='data', num_workers=1)

env = gym.make('MineRLObtainDiamondVectorObf-v0')

vae_model=vaeq("embedingdim_3",embedding_dim = 3,num_embeddings =65536,batch_size=16)

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
#probando primero con 1 trayectoria
trajectory_names = data.get_trajectory_names()
random.shuffle(trajectory_names)
trajectory = data.load_data(trajectory_names[0], skip_interval=0, include_metadata=False)
for dataset_observation, dataset_action, dataset_reward, _, _ in trajectory:
    all_actions.append(dataset_action["vector"])
    data_obs= dataset_observation["pov"].transpose(2, 0, 1)
    obs_encoding_tensor=vae_model.encodeSingle(th.tensor(data_obs,dtype=th.float32))
    all_pov_obs.append(obs_encoding_tensor.cpu())
    all_rewards.append(dataset_reward)
    
all_actions = np.array(all_actions)
#all_pov_obs = np.array(all_pov_obs)
all_rewards = np.array(all_rewards)

print("Running KMeans on the action vectors")
kmeans = KMeans(n_clusters=NUM_ACTION_CENTROIDS)
kmeans.fit(all_actions)
action_centroids = kmeans.cluster_centers_

print("KMeans done")
#convirtiendo una trayectoria a una sequencia(rewardsFuturos,estado,Accion,rewardsFuturos2,estado2.accion2...) 

all_pov_obs=th.tensor(all_pov_obs)
#Prepocesamos como en la baseline
print(all_pov_obs.shape)

# calculamos las distancias a los centroides
# "None" in indexing adds a new dimension that allows the broadcasting
distancias = np.sum((all_actions - action_centroids[:, None]) ** 2, axis=2)
trajectory_actions = np.argmin(distancias, axis=0)


#maybe usar nn.embeding para los embedings de los rewards y acciones
#Pasamos a tensores todos los 
all_rewards=th.from_numpy(all_rewards)
rewardsTogo =reward2go(all_rewards)
trajectory_actions=th.from_numpy(trajectory_actions)
#Añadimos dimesiones para staquear
rewardsTogo=th.tensor(rewardsTogo)
rewardsTogo=th.unsqueeze(rewardsTogo,1)
trajectory_actions=th.unsqueeze(trajectory_actions,1)

#all_pov_obs=th.unsqueeze(all_pov_obs,1)
print(trajectory_actions.shape)
print(rewardsTogo.shape)
print(all_pov_obs.shape)
#print([rewardsTogo,all_pov_obs,trajectory_actions])
print(rewardsTogo)
print(trajectory_actions)

embeding = th.hstack([rewardsTogo,all_pov_obs,trajectory_actions])#MYBE UN FLATEN ORDER F DESPUES?
print(embeding)




