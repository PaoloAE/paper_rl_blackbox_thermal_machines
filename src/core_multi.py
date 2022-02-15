import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

"""
This module defines the NNs used to parameterize the value and policy function when we have
a single continuous action and two multi-objectives.
"""

#Constants used by the NNs to prevent numerical instabilities
LOG_STD_MAX = 2
LOG_STD_MIN = -20
UNIFORM_SIGMA =3.46

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    return a sequential net of fully connected layers with custom
    activations

    Args:
        sizes(tuple(int)): sizes of all the layers
        activation: activation function for all layers except for the output layer
        output_activation: activation to use for output layer only

    Returns:
        stacked fully connected layers
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    """counts the parameters of a module """
    return sum([np.prod(p.shape) for p in module.parameters()])

def create_pi_fc_layers(channel_sizes, fc_sizes, act_dim):
    """
    Create the fully connected layers from the last convolutional block, to
    the last hidden layer in the policy architecture
    """
    if len(fc_sizes) == 0:
        return nn.Identity()
    else:
        layers = []
        all_fc_sizes = [channel_sizes[-1] + act_dim] + list(fc_sizes)
        for i in range(len(fc_sizes)):
            layers += [nn.Linear(all_fc_sizes[i], all_fc_sizes[i+1]), nn.ReLU()]
        return nn.Sequential(*layers)

def create_q_fc_layers(channel_sizes, act_dim, fc_sizes):
    """
    Create the fully connected layers from the last convolutional block, to
    the output which is the q-value 
    """
    layers = []
    all_fc_sizes = [channel_sizes[-1] + act_dim] + list(fc_sizes) 
    for i in range(len(fc_sizes)):
        layers += [nn.Linear(all_fc_sizes[i], all_fc_sizes[i+1]), nn.ReLU()]
    #add a final linear layer to output a single value
    layers += [ nn.Linear(all_fc_sizes[-1],1) ]
    return nn.Sequential(*layers)

def create_halving_layers( obs_dim, channel_sizes, perform_norm):
    """
    creates a series of convolution blocks that half the input time-series down to
    a single element (with multiple possible channels)
    """
    layers = [] 
    t_len = obs_dim[1]
    all_channel_sizes = [obs_dim[0]] + list(channel_sizes)
    for i in range(len(channel_sizes)):
        layers.append(ReduceBlock(all_channel_sizes[i], all_channel_sizes[i+1],
                                    t_len,perform_norm=perform_norm))
        t_len = t_len//2 + t_len%2
    return nn.Sequential(*layers)

def scale_input(obs, a, b):
    """
    scales the input from an interval [a,b] to an interval such that,
    if the input was a uniform disstribution between [a,b], now it's a
    uniform distribution with variance = 1
    """
    return UNIFORM_SIGMA*((obs - a)/(b-a) - 0.5) 

class ActorCritic(nn.Module):
    """
    Module that contains 4 value functions, self.qa1, self.qa2, self.qb1, 
    self.qb2, the policy self.pi, and self.alpha used to store and optimize the temperature
    parameter alpha. The value functions and policies are computed with a series of 
    convolutionan NNs, and then fully connected layers (see QFunction and SquashedGaussianActor 
    for details)
    WARNING: it only works for 1 continuous action. Should be easy to generalize to more.

    Args:
        observation_space: observation space of the environment
        action_space: action space of sac_envs environments
        channel_sizes (tuple(int)): number of channels of each convolution block halving
            the input size. The size of the tuple must be such that, halving the input each time 
            with the addition of padding before dividng if the number is odd - the final output
            has a single element series (and eventually multiple channels)
        pi_fc_sizes (tuple(int)): size of the fully connected hidden layers applied to the output of
            the last convolution block stacked together with the last action 
        q_fc_sizes (tuple(int)): size of the fully connected hidden layers applied to the output of
            the last convolution block stacked together with the action  
        alpha_reset_val (float): value to which alpha is reset if it becomes negative after an update
        perform_norm (bool): whether batch norm should be applied to each convolution block. Not 
            recommended.
        
    """
    def __init__(self, observation_space, action_space, channel_sizes, pi_fc_sizes, q_fc_sizes,
                alpha_reset_val = 1.e-5, perform_norm=False):
        super().__init__()

        #load size of observation and action space
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_lower_limit = action_space.low[0]
        act_upper_limit = action_space.high[0]
    
        #I check if the number of channels is correct with the length of the timeseries that
        #is halved every time
        assert len(channel_sizes) == int(np.ceil(np.log2(obs_dim[1]))) 

        # build policy, value functions, and alpha that stores and updates the alpha parameter
        self.pi = SquashedGaussianActor(obs_dim, act_dim, channel_sizes, pi_fc_sizes, act_lower_limit,
                                         act_upper_limit, perform_norm=perform_norm)

        self.qa1 = QFunction(obs_dim, act_dim, channel_sizes, q_fc_sizes, act_lower_limit, act_upper_limit,
                            perform_norm=perform_norm)
        self.qa2 = QFunction(obs_dim, act_dim, channel_sizes, q_fc_sizes, act_lower_limit, act_upper_limit,
                            perform_norm=perform_norm)
        self.qb1 = QFunction(obs_dim, act_dim, channel_sizes, q_fc_sizes, act_lower_limit, act_upper_limit,
                            perform_norm=perform_norm)
        self.qb2 = QFunction(obs_dim, act_dim, channel_sizes, q_fc_sizes, act_lower_limit, act_upper_limit,
                            perform_norm=perform_norm)
        self.alpha = Alpha(alpha_reset_val=alpha_reset_val)

    def act(self, obs, deterministic=False):
        """ return the action, chosen according to deterministic, given an obs """
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

    def alpha_no_grad(self):
        """ returns the current value of alpha without gradients """
        with torch.no_grad():
            return self.alpha()

class QFunction(nn.Module):
    """
    Class representing a q-value function. It is computed with a series of 
    convolutionan NNs applied to the state, and then fully connected layers applied to 
    this output, and to the action.

    Args:
        obs_dim(tuple(int)): shape of the observation space (channel, timeseries)
        act_dim(int): number of continuous actions
        channel_sizes (tuple(int)): number of channels of each convolution block halving
            the input size. The size of the tuple must be such that, halving the input each time 
            with the addition of padding before dividng if the number is odd - the final output
            has a single element series (and eventually multiple channels)
        fc_sizes (tuple(int)): size of the fully connected hidden layers applied to the output of
            the last convolution block stacked together with the action 
        act_lower_limit (float): the lower bound of the action
        act_upper_limit (float): the upper bound of the action
        perform_norm (bool): whether batch norm should be applied to each convolution block. Not 
            recommended.
        
    """
    def __init__(self, obs_dim, act_dim, channel_sizes, fc_sizes, act_lower_limit, act_upper_limit,
                perform_norm=False):
        super().__init__()
        #save the bounds for rescaling
        self.register_buffer("act_lower_limit", torch.as_tensor(act_lower_limit, dtype=torch.float32) )
        self.register_buffer("act_upper_limit", torch.as_tensor(act_upper_limit, dtype=torch.float32) )

        #this goes from input state, to a single timestep with channel_sizes[-1] channels 
        self.halving_layers = create_halving_layers(obs_dim,channel_sizes,perform_norm)
        
        #these are fc layer that take a stacking of conv layer output and action as input, and 
        #output a single value
        self.fc_layers = create_q_fc_layers(channel_sizes, act_dim, fc_sizes)
        
    def forward(self, obs, act):
        """
        Args:
            obs(torch.Tensor): batch of observations
            act(torch.Tensor): batch of continuous actions

        Returns:
            (torch.Tensor): 1D tensor with value of each state-action in the batch
        """
        #scale observations and actions
        obs_out = scale_input(obs, self.act_lower_limit, self.act_upper_limit)
        act_out = scale_input(act, self.act_lower_limit, self.act_upper_limit)

        #feed the observation into the conv layers, and flatten them
        obs_out = self.halving_layers(obs_out)
        obs_out = torch.flatten(obs_out,1)
        
        #concatenate observations and actions
        out = torch.cat([obs_out, act_out], dim=-1)

        #pass the concatenation to the fc layers
        out = self.fc_layers(out)

        return torch.squeeze(out, -1)

class SquashedGaussianActor(nn.Module):
    """
    Class representing the policy. It is computed with a series of 
    convolutionan NNs applied to the state, and then fully connected layers applied to 
    this output, and to the last action. Then the average and sigma of the
    probability density of chosing a continuous action is outputted. We use a squashed
    gaussian policy.

    Args:
        obs_dim(tuple(int)): shape of the observation space (channel, timeseries)
        act_dim(int): number of continuous actions
        channel_sizes (tuple(int)): number of channels of each convolution block halving
            the input size. The size of the tuple must be such that, halving the input each time 
            with the addition of padding before dividng if the number is odd - the final output
            has a single element series (and eventually multiple channels)
        fc_sizes (tuple(int)): size of the fully connected hidden layers applied to the output of
            the last convolution block stacked together with the last action 
        act_lower_limit (float): the lower bound of the action
        act_upper_limit (float): the upper bound of the action
        perform_norm (bool): whether batch norm should be applied to each convolution block. Not 
            recommended.
    """
    def __init__(self, obs_dim, act_dim, channel_sizes, fc_sizes, act_lower_limit, act_upper_limit,
                perform_norm=False):
        super().__init__()
        
        #save the action limits
        self.register_buffer("act_lower_limit", torch.as_tensor(act_lower_limit, dtype=torch.float32) )
        self.register_buffer("act_upper_limit", torch.as_tensor(act_upper_limit, dtype=torch.float32) )

        #this goes from the state, to a single timestep with channel_sizes[-1] channels 
        self.halving_layers = create_halving_layers(obs_dim,channel_sizes,perform_norm)
        
        #this is a series of fully-connected layers to put right after that also takes
        # the last action. If fc_sizes = (), it will just be the identity
        self.fc_layers = create_pi_fc_layers(channel_sizes, fc_sizes, act_dim)
        
        #size of output after the last two layers
        final_size = channel_sizes[-1] + act_dim if len(fc_sizes) == 0 else fc_sizes[-1]

        #layer producing the average of the gaussian mu used to sample the action
        self.mu_layer = nn.Linear(final_size, act_dim)
        #layer producing the log of the std of the gaussian used to sample the action
        self.log_std_layer = nn.Linear(final_size, act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        Args:
            obs(torch.Tensor): batch of observations
            deterministic(bool): if the actions should be chosen deterministally or not
            with_logprob(bool): if the log of the probability should be computed and returned

        Returns:
            pi_action(torch.Tensor): the chosen continuous action
            logp_pi(torch.Tensor): the log probability of such continuous action
        """
        #rescaling the state time-series
        scaled_obs = scale_input(obs,self.act_lower_limit,self.act_upper_limit)

        #feed the rescaled input into the halving layers and flatten
        out = self.halving_layers(scaled_obs)
        out = torch.flatten(out,1)

        #append also the last action, since it is in a priviledged position
        out = torch.cat([out, scaled_obs[:,:,-1]], dim=-1)

        #apply some fully connected layers
        out = self.fc_layers(out)

        #output the heads (average and standard deviation of the gaussian)
        mu = self.mu_layer(out)
        log_std = self.log_std_layer(out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash (tanh) distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
            
        #if necessary, compute the log of the probabilities
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            #change of distribution when going from gaussian to Tanh
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            #change of distribution when going from Tanh to the interval [act_lower_limit,act_upper_limit]
            logp_pi -= torch.log(0.5*(self.act_upper_limit-self.act_lower_limit)) 
        else:
            logp_pi = None

        #Apply Tanh to the sampled gaussian    
        pi_action = torch.tanh(pi_action)
        #Apply the shift of the action to the correct interval
        pi_action = self.act_lower_limit + 0.5*(pi_action + 1.)*(self.act_upper_limit-self.act_lower_limit)

        return pi_action, logp_pi

class Alpha(nn.Module):
    """
    Class containing only the alpha val, and used to update it with some optimizer
   
    Args:
        alpha_reset_val (float): value to which alpha is set if it goes negative during an
            optimization
    """
    def __init__(self, alpha_reset_val = 1.e-5):
        super().__init__()
        #initialize it to some "large" initial value
        self.alpha_val = nn.Parameter(torch.tensor(5., dtype=torch.float32))
        #this is the value of alpha, such that if alpha is negative, it gets set to this
        self.register_buffer("alpha_reset_val", torch.tensor(alpha_reset_val, dtype=torch.float32))

    def forward(self):
        """
        simply returns the value of alpha, which is stored as a proper trainable parameter
        """
        return 1.*self.alpha_val

    def reset_alpha(self):
        """
        sets the value of alpha to self.alpha_reset_val. It is called if alpha becomes negative
        """
        with torch.no_grad():
            self.alpha_val.copy_(self.alpha_reset_val)

class ReduceBlock(nn.Module):
    """
    Class that takes as input (N,C,T), where N is batch size, C is number of channels, T is time-series length,
    does a 1D convolution with kernel=2 and stride=2, eventually padding if odd, eventually batch norm, relu,
    and then it sums it with the imnut (skip connect) which is first passed through a average pooling kernel=2
    stride=2 and a 1x1 convolution to match the size.
    
    Args:
        c_in (int): number of input channels
        c_out (int): number of output channels
        t_len (int): input length of the timeseries
        perform_norm (bool): whether we do or not the batch norm. Not recommended
    
    """
    def __init__(self, c_in, c_out, t_len, perform_norm=True):
        super().__init__()

        #convolutional layer that halves the input. If odd, adds padding on the old time side
        self.conv = nn.Conv1d(c_in, c_out, 2, stride=2, padding=t_len % 2, padding_mode="replicate")
        #normalization layer
        if perform_norm:
            #the average is computed over the batch, and over the timeseries dimension 
            self.norm = nn.BatchNorm1d(c_out)
        else:
            self.norm = nn.Identity()
        #relu
        self.relu = nn.ReLU()
        #average pooling 
        self.pool = nn.AvgPool1d(2, stride=2, padding=t_len % 2, count_include_pad=False)
        #1x1 convolution only if the number of channel changes
        if c_in == c_out:
            self.conv_1x1 = nn.Identity()
        else:
            self.conv_1x1 = nn.Conv1d(c_in, c_out, 1, stride=1)

    def forward(self, x):
        """
        Args:
            x(torch.Tensor): input as described in the class doc-string.

        Returns:
            (torch.Tensor): output as described in the class doc-string, i.e. 
                halving the timeseries lenght, and eventually changing the
                number of channels.
        """
        # The nonlinear branch:
        # convolution
        out = self.conv(x)
        # batchnorm
        out = self.norm(out)
        # ReLU
        out = self.relu(out)

        #the skip branch:
        #pooling to match t_len dimension
        skip = self.pool(x)
        #1x1 convolution to match channel number
        skip = self.conv_1x1(skip)

        #sum the two branches together
        return out + skip







