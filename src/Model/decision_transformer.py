import numpy as np
import torch
import torch.nn as nn

import transformers

from Model.trajectory_gpt2 import GPT2Model

class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    Nicked from stable-baselines3:
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):#TODO try images as channels or 3d NATURE cnn?
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class DecisionTransformer(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=18000,
            action_tanh=True,
            discrete_rewards=1095,
            kmeans_centroids=None,#number of actions the agent can take, none to use continuous actions
            discrete_states=65536,
            natureCNN=False,
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size#reemplazando a Trajectory model , posibles bugs causados por eso
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.discrete_rewards=discrete_rewards
        self.kmeans_centroids=kmeans_centroids
        self.discrete_states=discrete_states
        self.natureCNN=natureCNN
        
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        if discrete_rewards is not None:#TODO add some form of reward tokenizer for the embedding.
            self.embed_return = nn.Embedding(discrete_rewards, hidden_size)
        else:
            self.embed_return = torch.nn.Linear(1, hidden_size)

        if kmeans_centroids is not None:
            self.num_centroids=len(kmeans_centroids)
            self.embed_action =  nn.Embedding(self.num_centroids, hidden_size)
        else:
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        if discrete_states is not None:
            self.embed_state = lambda x: x #just returns the state unchanged

        elif natureCNN:
            self.embed_state  = NatureCNN((3, 64, 64), hidden_size)
        else:
            self.embed_state = torch.nn.Linear(self.state_dim[0], hidden_size)
    
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # note: we don't predict states or returns for the paper
        
        if discrete_states is not None:
            self.predict_state = torch.nn.Linear(hidden_size,self.state_dim[0])#todo maybe change to the vq_vae embedding
        elif natureCNN:#TODO make this work probably doest make sense currently
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim[0])
        else:
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim[0])
            
        self.predict_action = nn.Sequential(#TODO try generating centroid directly
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
   
        
        
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):#TODO maybe call diferent methods for the diferetn options from this forward for redability

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        if self.discrete_rewards is not None:    
            returns_to_go=returns_to_go.to(dtype=torch.float32)
        if self.kmeans_centroids is not None:
            actions = torch.argmin(torch.cdist(actions,self.kmeans_centroids),2)#maybe add detatch cause otherwise gradient doest flow though   
            #actions = actions + (actions_kmeans - actions).detach()

            #actions=actions.to(dtype=torch.in)
        if self.natureCNN:
            states = states.reshape(-1, 3, states.shape[2], states.shape[3]).type(torch.float32).contiguous() #reshapes the steps as batches (batch * block_size, n_embd)
        #print(states.shape)  
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        if self.natureCNN:
                state_embeddings = state_embeddings.reshape(batch_size, seq_length, self.hidden_size) 
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        

        
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state
        if self.kmeans_centroids is not None:
            action_index = torch.argmin(torch.cdist(action_preds,self.kmeans_centroids),2)#maybe add detatch cause otherwise gradient doest flow though   
            action_preds_kmeans=self.kmeans_centroids[action_index]
            action_preds = action_preds + (action_preds_kmeans - action_preds).detach()

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        


        #states = states.reshape(1, -1, self.state_dim) TODO make it work when no vq_vqvae
        #states = states.reshape(1, -1)

        states = states.reshape((1, -1,)+self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]
            
            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                    [torch.zeros((states.shape[0], self.max_length-states.shape[1], )+self.state_dim, device=states.device), states],
                    dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                    [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                                device=actions.device), actions],
                    dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]



