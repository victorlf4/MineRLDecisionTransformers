# MineRLDecisionTransformers

# Overview
Code for a Decision trasnformers for the enviromnents of the minerl competition based on the [official decision transformers codebase](https://github.com/kzl/decision-transformer).   

# Installation
Install the minerl [requirements](https://minerl.readthedocs.io/en/latest/tutorials/index.html) and then install the required python libraries whith: 
```
pip install -r requirements.text
```

# Training and testing the model
To train a decision transfomer model whith the default config run: 
```
python train_dt.py 
```

To test the model on the mineRl enviroment run:
```
python evaluate_model.py 
```

To train a vq_vae model use:
```
python evaluate_model.py 
```

# Parameters
Apart from the basic configuration you can use diferet optional arguments listed below:
```
usage: train_dt.py [-h] [--mode MODE] [--max_iters MAX_ITERS] [--num_steps_per_iter NUM_STEPS_PER_ITER]
                   [--num_validation_iters NUM_VALIDATION_ITERS] [--device DEVICE]
                   [--log_to_wandb LOG_TO_WANDB] [--group_name GROUP_NAME]
                   [--use_checkpoint USE_CHECKPOINT] [--model_name MODEL_NAME] [--env ENV]
                   [--vectorize_actions VECTORIZE_ACTIONS] [--visualize VISUALIZE] [--record RECORD]
                   [--max_ep_len MAX_EP_LEN] [--max_ep_len_dataset MAX_EP_LEN_DATASET] [--dataset DATASET]
                   [--dataset_validation DATASET_VALIDATION] [--buffer_target_size BUFFER_TARGET_SIZE]
                   [--buffer_target_size_validation BUFFER_TARGET_SIZE_VALIDATION]
                   [--store_rewards2go STORE_REWARDS2GO] [--batch_size BATCH_SIZE] [--dropout DROPOUT]
                   [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                   [--warmup_steps WARMUP_STEPS] [--num_eval_episodes NUM_EVAL_EPISODES]
                   [--validation_steps VALIDATION_STEPS]
                   [--validation_trajectories VALIDATION_TRAJECTORIES]
                   [--target_rewards TARGET_REWARDS [TARGET_REWARDS ...]] [--embed_dim EMBED_DIM]
                   [--n_layer N_LAYER] [--n_head N_HEAD] [--K K] [--pov_encoder {linear,cnn,vq_vae}]
                   [--state_vector STATE_VECTOR] [--activation_function ACTIVATION_FUNCTION]
                   [--vae_model VAE_MODEL] [--vae_embedings VAE_EMBEDINGS]
                   [--vae_embedding_dim VAE_EMBEDDING_DIM] [--kmeans_actions KMEANS_ACTIONS]
                   [--kmeans_action_centroids KMEANS_ACTION_CENTROIDS]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE
  --max_iters MAX_ITERS Number of iterations to execute during training
  --num_steps_per_iter NUM_STEPS_PER_ITER Number of batches in an iteration
  --num_validation_iters NUM_VALIDATION_ITERS Number of validation iterantions before running minecraft
  --device DEVICE Device we train on read pytorch documentation for more info
  --log_to_wandb LOG_TO_WANDB, -w LOG_TO_WANDB logging on weight and biases, requires weight and biases account
  --group_name GROUP_NAME, -g GROUP_NAME  wandb group name
  --use_checkpoint USE_CHECKPOINT If true saves the model each iteration
  --model_name MODEL_NAME Name of the model used in the checkpoint file and in wandb
  --env ENV MineRl enviroment the model will be evaluated in
  --vectorize_actions VECTORIZE_ACTIONS Necesary to train and evaluate on Basalt envs and datasets
  --visualize VISUALIZE MineRl enviroment the model will be evaluated in
  --record RECORD records video, currently only working correctly on evaluate_model
  --max_ep_len MAX_EP_LEN max lenght of an evaluation episode in frames
  --max_ep_len_dataset MAX_EP_LEN_DATASET Maximun lenght of a trajectory in the dataset in frames,affects temporal encoder
                        size
  --dataset DATASET Dataset used for training
  --dataset_validation DATASET_VALIDATION Dataset used for validation
  --buffer_target_size BUFFER_TARGET_SIZE size of the buffer used for loading from the training dataset
  --buffer_target_size_validation BUFFER_TARGET_SIZE_VALIDATION size of the buffer used for loading from the validation dataset
  --store_rewards2go STORE_REWARDS2GO stores calculated reward to go on ram
  --batch_size BATCH_SIZE
  --dropout DROPOUT
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --weight_decay WEIGHT_DECAY, -wd WEIGHT_DECAY
  --warmup_steps WARMUP_STEPS
  --num_eval_episodes NUM_EVAL_EPISODES
  --validation_steps VALIDATION_STEPS
  --validation_trajectories VALIDATION_TRAJECTORIES
  --target_rewards TARGET_REWARDS [TARGET_REWARDS ...]
  --embed_dim EMBED_DIM dimension of the mbedding of each token
  --n_layer N_LAYER number of layers of the transformer model
  --n_head N_HEAD number of attention heads of the transformer model
  --K K context window of the model in frames actual size in tokens is x3 or x4 depending on whether you use state tokens
  --pov_encoder {linear,cnn,vq_vae}
  --state_vector STATE_VECTOR if true adds encoder for state vector to the model
  --activation_function ACTIVATION_FUNCTION
  --vae_model VAE_MODEL
  --vae_embedings VAE_EMBEDINGS
  --vae_embedding_dim VAE_EMBEDDING_DIM
  --kmeans_actions KMEANS_ACTIONS if true runs kmeans the actions to discretize the action space
  --kmeans_action_centroids KMEANS_ACTION_CENTROIDS number of action centroids on the if kmeans actions is used
  ```
