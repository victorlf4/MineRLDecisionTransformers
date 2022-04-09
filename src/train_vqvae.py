
from dataclasses import dataclass
import random
from  Model.vq_vae import VectorQuantizerVAE as vaeq
import os
import wandb
import minerl
import gym
import numpy as np
from torch.utils.data import DataLoader
import torch as th
from minerl.data import BufferedBatchIter
import torchvision.transforms as transforms
from utils.verify_or_download_minerl import verify_or_download_dataset
from utils.minerl_iterators import MinerlImageIterator

#TODO add weight and biases logging and parameters
train=True

verify_or_download_dataset(directory='data', environment='MineRLObtainDiamondVectorObf-v0')
data = minerl.data.make("MineRLObtainDiamondVectorObf-v0",  data_dir='data', num_workers=1)

verify_or_download_dataset(directory='data', environment='MineRLObtainIronPickaxeVectorObf-v0')
dataValidation = minerl.data.make("MineRLObtainIronPickaxeVectorObf-v0",  data_dir='data', num_workers=1)#we validate using the smaller iron pickaxeDataset

variace_estimation_dataset= MinerlImageIterator(data,num_epochs=None,num_batches=100)#maybe calculate variance on the entire dataset instead though slow
observation_dataset= MinerlImageIterator(data,transform=transforms.ToTensor())
validation_dataset= MinerlImageIterator(dataValidation,transform=transforms.ToTensor())

env = gym.make('MineRLObtainDiamondVectorObf-v0')

vae_model=vaeq("embedingdim_1_numembedings_131072",embedding_dim = 1,num_embeddings =131072,batch_size=32)

observations=[]
for image in variace_estimation_dataset:
    observations.append(image)
observations=np.array(observations)
data_variance = np.var(observations/255.0)
#observation_dataset=MinerlImageIterator(data,DATA_SAMPLES-VALIDATION_SAMPLES,transforms.ToTensor())
#validation_dataset=MinerlDatasetImages(data,VALIDATION_SAMPLES,transforms.ToTensor())

#data_variance = observation_dataset.getVariance()


training_loader = DataLoader(observation_dataset, #maybe self?
                                batch_size=32, 
                                shuffle=False,
                                pin_memory=True)
validation_loader = DataLoader(validation_dataset, #maybe self?
                                batch_size=32, 
                                shuffle=False,
                                pin_memory=True)

if(train):
    vae_model.train(training_loader,data_variance)
    vae_model.plot()
else:
    vae_model.load()
#vae_model.evalImage(th.unsqueeze(next(iter(validation_loader))[0],dim=0))
vae_model.eval(validation_loader)
    

    