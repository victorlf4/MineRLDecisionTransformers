import torchvision.datasets as datasets
import torchvision.transforms as transforms
from  Model.vq_vae import VectorQuantizerVAE as vaeq
import numpy as np
from torch.utils.data import DataLoader

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                     transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

training_loader = DataLoader(training_data, #maybe self?
                                    batch_size=32, 
                                    shuffle=True,
                                    pin_memory=True)

validation_loader = DataLoader(validation_data,
                                    batch_size=32,
                                    shuffle=True,
                                    pin_memory=True)
data_variance = np.var(training_data.data / 255.0)                                  
vae_eq_model = vaeq("cifar")
#vae_eq_model.train(training_data,data_variance)
vae_eq_model.load()
vae_eq_model.eval(validation_loader)