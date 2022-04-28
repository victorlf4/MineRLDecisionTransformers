from gym.wrappers import Monitor
import gym
from minerl.herobraine.wrappers.vector_wrapper  import Vectorized
from minerl.herobraine.wrappers.obfuscation_wrapper  import Obfuscated
import minerl
def load_env(env_name,vectorize_actions=False,obfuscate_actions=False,record=False):
                data_enviroment=None
                if vectorize_actions:
                        spec_pipeline =  minerl.data.make(env_name,  data_dir='data')
                        enviroment=spec_pipeline.spec
                        enviroment=Vectorized(enviroment)
                        if obfuscate_actions:
                                enviroment=Obfuscated(enviroment)        
                        enviroment.register()
                        env_name=enviroment._update_name(env_name)
                elif obfuscate_actions:
                        print("can't obfuscate actions whithout vectorizing")

                
                enviroment = gym.make(env_name)
                if record:
                        enviroment=Monitor(enviroment,"./video",force=True)#TODO fix some bugs whith not doing the steps at the same time
                
                return enviroment,data_enviroment
def vectorize_dataset(data):
    data_enviroment=Vectorized(data.spec)
    return data_enviroment
