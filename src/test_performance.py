import cProfile
import minerl
import random
import torch as th
import os
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT')
from utils.buffer_trajectory_load import BufferedTrajectoryIter
from utils.buffer_trajectory_test_performance import BufferedTrajectoryIterTest
from utils.minerl_encode_sequence import minerlEncodeSequence_performance_test
def test_minerl():
    #data_pipeline = minerl.data.make("MineRLObtainDiamondVectorObf-v0",  data_dir='data', num_workers=1)
    data_pipeline = minerl.data.make("MineRLTreechopVectorObf-v0",  data_dir='data', num_workers=1)
    bbi = BufferedTrajectoryIter(data_pipeline, buffer_target_size=2000,sequence_size=20,reward_to_go=True,store_rewards2go=False)
    #bbi = minerl.data.BufferedBatchIter(data_pipeline, buffer_target_size=20000)
    num_timesteps = 0
    #for batch in bbi.buffered_batch_iter(batch_size=64, num_epochs=1):
    for batch in bbi.buffered_batch_iter(batch_size=64,num_batches=100):
            for step in batch:
                num_timesteps += 1
                sequence=minerlEncodeSequence_performance_test(step,"cuda",discrete_rewards=False,vae_model=None,convolutional_head=False)
    print(num_timesteps)
import time
  
start = time.time()
test_minerl()
print("Time Consumed")
print("% s seconds" % (time.time() - start))