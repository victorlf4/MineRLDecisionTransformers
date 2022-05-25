import minerl
from utils.verify_or_download_minerl import verify_or_download_dataset
from utils.minerl_iterators import MinerlImageIterator
import numpy as np
def calculate_mean_std(dataset_name):
    verify_or_download_dataset(directory='data', environment='MineRLObtainDiamondVectorObf-v0')
    data = minerl.data.make("MineRLObtainDiamondVectorObf-v0",  data_dir='data', num_workers=1)
    dataset_iterator= MinerlImageIterator(data,num_epochs=1)#maybe calculate variance on the entire dataset instead though slow
    all_obs=[]
    for image in dataset_iterator:
        all_obs.append(image)
    return np.mean(all_obs) ,np.std(all_obs)

