import torchvision.datasets as datasets
import torchvision.transforms as transforms
from  Model.vq_vae import VectorQuantizerVAE as vaeq
import numpy as np
from torch.utils.data import DataLoader
import torch as th

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
#vae_eq_model.train(training_loader,data_variance)
vae_eq_model.load()
#vae_eq_model.eval(validation_loader)
dataset=[]
for i in range(100):
   #dataset.append([vae_eq_model.encode(next(iter(validation_loader))),i,1])
   #dataset.append([vae_eq_model.encode(next(iter(validation_loader))),-i,0])
   encoding_string=' '.join([str(x.item()) for x in vae_eq_model.encodeSingle(next(iter(validation_data)))])
   dataset.append(""+encoding_string+" "+ str(i) )
   encoding_string=' '.join([str(x.item()) for x in vae_eq_model.encodeSingle(next(iter(validation_data)))])
   dataset.append(""+encoding_string+" "+str(-i))


from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
from torch import nn

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

sequence = ' Step '.join(dataset[0:4])
sequence= sequence[:-70]
print(sequence)
print("/n")

inputs = tokenizer(sequence, return_tensors="pt")
input_ids = inputs["input_ids"]
print(inputs)

# get logits of last hidden state
next_token_logits = model(**inputs).logits[:, -1, :]

# filter
filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

# sample
probs = nn.functional.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

generated = torch.cat([input_ids, next_token], dim=-1)

resulting_string = tokenizer.decode(generated.tolist()[0])
print(resulting_string)
