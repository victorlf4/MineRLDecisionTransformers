import torch as th
import numpy as np
from minerl.data import BufferedBatchIter

class MinerlImageIterator(th.utils.data.IterableDataset):
    def __init__(self,data,transform=None,num_epochs=10,num_batches=None):
        self.buffer = BufferedBatchIter(data)
        self.transform=transform
        self.iterator=self.buffer.buffered_batch_iter(batch_size=1,num_epochs=num_epochs, num_batches=num_batches)

    def __next__(self):
        dataset_observation, _, _, _, _ = next(self.iterator)
        if self.transform:
            return self.transform(np.squeeze(dataset_observation["pov"]))
        else:
            return dataset_observation["pov"]
        
        
    def __iter__(self):
        return self
        
class MinerlActionIterator(th.utils.data.IterableDataset):
    def __init__(self,data,transform=None,num_epochs=10,num_batches=None):
        self.buffer = BufferedBatchIter(data)
        self.transform=transform
        self.iterator=self.buffer.buffered_batch_iter(batch_size=1,num_epochs=num_epochs, num_batches=num_batches)

    def __next__(self):
        _, dataset_action, _, _, _ = next(self.iterator)
        if self.transform:
            return self.transform(np.squeeze(dataset_action["vector"]))
        else:
            return np.squeeze(dataset_action["vector"])
        
        
    def __iter__(self):
        return self
import time      
import minerl
if __name__ == "__main__":

    env = "MineRLTreechopVectorObf-v0"
    #env ="MineRLObtainDiamondVectorObf-v0"
    test_batch_size = 32
    num_timesteps=0
    #minerl.data.download(directory='data', environment=env)
    data_pipeline =   minerl.data.make(env,  data_dir='data')
    action_iterator =MinerlActionIterator(data_pipeline)
    start_time = time.time()
    
    action_samples=[]
    for i in range(100000):
        action_samples.append(next(action_iterator))
        num_timesteps += 1
        #print(next(action_iterator))
    unique_data = [list(x) for x in set(tuple(x) for x in action_samples)]
    print(unique_data)
    print(f"{len(unique_data)} unique arrays found for env {env} using batch_iter")
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")        