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

if __name__ == "__main__":

    #env = "MineRLBasaltMakeWaterfall-v0"
    env = "MineRLTreechopVectorObf-v0"
    test_batch_size = 32

    start_time = time.time()
    minerl.data.download(directory='data', environment=env)#TODO codigo de baselines competition
    data_pipeline =   minerl.data.make(env,  data_dir='data')
    bbi = BufferedTrajectoryIterTest(data_pipeline,sequence_size=20,reward_to_go=True, buffer_target_size=10000)
    num_timesteps = 0
    for data_dict in bbi.buffered_batch_iter(batch_size=test_batch_size, num_epochs=1):
        num_timesteps +=1 #len(data_dict['obs']['pov'])

    print(f"{num_timesteps} found for env {env} using batch_iter")
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")