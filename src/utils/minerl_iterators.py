import torch as th
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
        
