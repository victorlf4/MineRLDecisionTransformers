# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

from argparse import Action
import numpy as np
from functools import reduce
from collections import OrderedDict
import torch as th

from minerl.herobraine.wrappers.vector_wrapper  import Vectorized


if __name__ == "__main__":
    import gym
    import minerl
    from minerl.herobraine.env_specs import basalt_specs
    #enviroment = gym.make("MineRLTreechop-v0")
  
    data_pipeline =   minerl.data.make("MineRLBasaltMakeWaterfall-v0",  data_dir='data')
    env=Vectorized(data_pipeline.spec)
    bbi = minerl.data.BufferedBatchIter(data_pipeline, buffer_target_size=3000)
    act=next(bbi.buffered_batch_iter(batch_size=1, num_epochs=1))[1]

    obs=next(bbi.buffered_batch_iter(batch_size=1, num_epochs=1))[0]
    vectorized_act = env._wrap_action(act)
    print(vectorized_act)
    print(env.action_space.sample())
    print(obs["pov"])
    print(th.tensor(obs["pov"],dtype=th.int).div(256))
