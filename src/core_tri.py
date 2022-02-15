import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

"""
This module defines the NNs used to parameterize the value and policy function when we have
a discrete action with 3 choices, and a single continuous action.
"""

#Constants used by the NNs to prevent numerical instabilities
LOG_STD_MAX = 2
LOG_STD_MIN = -20
PROBS_MIN = 2e-5
PROBS_MAX = 1.
UNIFORM_SIGMA =3.46

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    return a sequential net of fully connected layers with custom
    activations

    Args:
        sizes(tuple(int)): sizes of all the layers
        activation: activation function for all layers except for the output layer
        output_activation: activation to use for output layer only. if a string
            "soft_max" is passed, it will apply a softmax.

    Returns:
        stacked fully connected layers
    """
    layers = []
    for j in range(len(sizes)-1):
        if j < len(sizes)-2:
            act = activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        else:
            if output_activation == "soft_max":
                layers += [nn.Linear(sizes[j], sizes[j+1]), nn.Softmax(dim=-1)]
            else:
                act = output_activation
                layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    """counts the parameters of a module """
    return sum([np.prod(p.shape) for p in module.parameters()])

def create_pi_fc_layers(channel_sizes, fc_sizes, full_act_dim):
    """
    Create the fully connected layers from the last convolutional block, to
    the last hidden layer in the policy architecture
    """
    if len(fc_sizes) == 0:
        return nn.Identity()
    else:
        layers = []
        all_fc_sizes = [channel_sizes[-1] + full_act_dim] + list(fc_sizes)
        for i in range(len(fc_sizes)):
            layers += [nn.Linear(all_fc_sizes[i], all_fc_sizes[i+1]), nn.ReLU()]
        return nn.Sequential(*layers)

def create_q_fc_layers(channel_sizes, act_dim, fc_sizes):
    """
    Create the fully connected layers from the last convolutional block, to
    the output which is are the 3 q-values associated to each discrete action 
    """
    layers = []
    all_fc_sizes = [channel_sizes[-1] + act_dim] + list(fc_sizes) 
    for i in range(len(fc_sizes)):
        layers += [nn.Linear(all_fc_sizes[i], all_fc_sizes[i+1]), nn.ReLU()]
    #add a final linear layer to output the 3 values (for each discrete action)
    layers += [ nn.Linear(all_fc_sizes[-1],3) ]
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
    Since the input is an observation, there are 4 channels per time step.
    The first 3 are just one-hot encoding, in [0, 1], whereas the
    last channel is in [a,b]. This scales these inputs, living in different ranges,
    such that, if the input was a uniform disstribution in its inserval,
    the output would be a uniform distribution with variance = 1
    """
    a_mat = torch.zeros_like(obs)-0.5
    b_mat = torch.ones_like(obs) -0.5
    sigma_mat = torch.ones_like(obs)
    a_mat[:,-1:,:] = a 
    b_mat[:,-1:,:] = b
    sigma_mat[:,-1:,:] = UNIFORM_SIGMA
    return sigma_mat*((obs - a_mat)/(b_mat-a_mat) - 0.5) 

def scale_all_input(obs, a, b):
    """
    scales the whole input from an interval [a,b] to an interval such that,
    if the input was a uniform disstribution between [a,b], now it's a
    uniform distribution with variance = 1
    """
    return UNIFORM_SIGMA*((obs - a)/(b-a) - 0.5) 
    
class ActorCritic(nn.Module):
    """
    Module that contains 2 value functions, self.q1, self.q2, the policy self.pi, and 
    self.alpha used to store and optimize the two temperature parameters alpha_d and 
    alpha_c associated to the discrete and continuous part of the policy. The value functions
    and policies are computed with a series of convolutionan NNs, and then fully connected
    layers (see QFunction and SquashedGaussianActor for details)
    WARNING: it only works for 1 continuous action and 1 discrete action with 3 options.
    Should be easy to generalize to more.

    Args:
        observation_space: observation space of the sac_tri_envs environment.
        action_space: action space of sac_tri_envs environment.
        channel_sizes (tuple(int)): number of channels of each convolution block halving
            the input size. The size of the tuple must be such that, halving the input each time 
            with the addition of padding before dividng if the number is odd - the final output
            has a single element series (and eventually multiple channels)
        pi_fc_sizes (tuple(int)): size of the fully connected hidden layers applied to the output of
            the last convolution block stacked together with the last action 
        q_fc_sizes (tuple(int)): size of the fully connected hidden layers applied to the output of
            the last convolution block stacked together with the action.
        alpha_reset_val (float): value to which both alpha are reset if they become negative after an update
        perform_norm (bool): whether batch norm should be applied to each convolution block. Not 
            recommended.
        
    """

    def __init__(self, observation_space, action_space, channel_sizes, pi_fc_sizes, q_fc_sizes,
                alpha_reset_val=1.e-5, perform_norm=False):
        super().__init__()

        #load size of observation and action space
        obs_dim = observation_space.shape
        act_dim = action_space[1].shape[0]
        act_lower_limit = action_space[1].low[0]
        act_upper_limit = action_space[1].high[0] 

        #check if there is only 1 continuous action
        assert act_dim == 1
        
        #I check if the number of channels is correct with the length of the timeseries that
        #is halved every time
        assert len(channel_sizes) == int(np.ceil(np.log2(obs_dim[1]))) 

        # build policy, value functions, and alpha that stores and updates the alpha parameters
        self.pi = SquashedGaussianActor(obs_dim, act_dim, channel_sizes, pi_fc_sizes, act_lower_limit,
                                         act_upper_limit, perform_norm=perform_norm)
        
        self.q1 = QFunction(obs_dim, act_dim, channel_sizes, q_fc_sizes, act_lower_limit, act_upper_limit,
                            perform_norm=perform_norm)
        self.q2 = QFunction(obs_dim, act_dim, channel_sizes, q_fc_sizes, act_lower_limit, act_upper_limit,
                            perform_norm=perform_norm)
        self.alpha_d = Alpha(alpha_reset_val=alpha_reset_val)
        self.alpha_c = Alpha(alpha_reset_val=alpha_reset_val)

        #create a zero and one for later reference
        self.zero_float = torch.tensor(0., dtype=torch.float32)
        self.one_float = torch.tensor(1., dtype=torch.float32)

    def act(self, obs, deterministic=False):
        """ return the action, chosen according to deterministic, given an obs """
        with torch.no_grad():
            b_action, pi0_action, pi1_action, pi2_action, _, _, _, _, _ = self.pi(
                obs.view((-1,obs.shape[-2],obs.shape[-1])), deterministic, False)         
            if torch.isclose(b_action[0],self.zero_float):
                return (b_action[0], pi0_action[0])
            elif torch.isclose(b_action[0],self.one_float):
                return (b_action[0], pi1_action[0])
            else:
                return (b_action[0], pi2_action[0])
    
    def alpha_d_no_grad(self):
        """
        returns the current value of alpha_d, i.e. the temperature associated to the discrete
        part of the policy, without gradients
        """
        with torch.no_grad():
            return self.alpha_d()

    def alpha_c_no_grad(self):
        """
        returns the current value of alpha_c, i.e. the temperature associated to the continuous
        part of the policy, without gradients
        """
        with torch.no_grad():
            return self.alpha_c()

class QFunction(nn.Module):
    """
    Class representing a q-value function. It is computed with a series of 
    convolutionan NNs applied to the state, and then fully connected layers applied to 
    this output, and to the action. The output is 3 values, i.e. the value of each discrete
    action.

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
        
        #these are fc layer that take a stacking of state and action as input, and 
        #output 3 values, i.e. the value function associate to each discrete action
        self.fc_layers = create_q_fc_layers(channel_sizes, act_dim, fc_sizes)

    def forward(self, obs, act):
        """
        Args:
            obs(torch.Tensor): batch of observations
            act(torch.Tensor): batch of continuous actions

        Returns:
            (torch.Tensor): 2D tensor with batch on first dimension, and 3 values 
                on second dimension, corresponding to the value function of each 
                discrete action
        """
        #scale observations and actions
        obs_out = scale_input(obs, self.act_lower_limit, self.act_upper_limit)
        act_out = scale_all_input(act, self.act_lower_limit, self.act_upper_limit)

        #feed the observation into the conv layers, and flatten them
        obs_out = self.halving_layers(obs_out)
        obs_out = torch.flatten(obs_out,1)
        
        #concatenate observations and actions
        out = torch.cat([obs_out, act_out], dim=-1)

        #pass the concatenation to the fc layers
        out = self.fc_layers(out)

        return out

class SquashedGaussianActor(nn.Module):
    """
    Class representing the policy. It is computed with a series of 
    convolutionan NNs applied to the state, and then fully connected layers applied to 
    this output, and to the last action. The the output is the marginal probability of each
    discrete action, and the average and sigma of the conditional density probability of chosing
    a continuous action, given each of the 3 discrete actions. The density probability for
    the conditional continuous action is a squashed gaussian policy.
    
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
        self.fc_layers = create_pi_fc_layers(channel_sizes, fc_sizes, obs_dim[0])

        #size of output after the last two layers
        final_size = channel_sizes[-1] + obs_dim[0] if len(fc_sizes) == 0 else fc_sizes[-1]

        #final layers producing the average of the gaussian mu used to sample the continuous
        #action corresponding to each conditional probability
        self.mu0_layer = nn.Linear(final_size, act_dim)
        self.mu1_layer = nn.Linear(final_size, act_dim)
        self.mu2_layer = nn.Linear(final_size, act_dim)

        #final layers producing the log of the std of the gaussian used to sample the continuous
        #action corresponding to each conditional probability
        self.log0_std_layer = nn.Linear(final_size, act_dim)
        self.log1_std_layer = nn.Linear(final_size, act_dim)
        self.log2_std_layer = nn.Linear(final_size, act_dim)
        
        #final laver to produce the marginal probabilities of the discrete action
        p_activation = "soft_max"
        self.p_layer = mlp(sizes=[final_size, 3], activation=p_activation, output_activation=p_activation)

    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        Args:
            obs(torch.Tensor): batch of observations
            deterministic(bool): if the actions should be chosen deterministally or not
            with_logprob(bool): if the log of the probability should be computed and returned

        Returns:
            b_action(torch.Tensor): the chosen discrete action (0,1,2)
            pi0_action(torch.Tensor): the chosen continuous action if discrete action 0 is chosen
            pi1_action(torch.Tensor): the chosen continuous action if discrete action 1 is chosen
            pi2_action(torch.Tensor): the chosen continuous action if discrete action 2 is chosen
            p(torch.Tensor): the probability of chosing each of the 3 discrete actions
            p_entropy(torch.Tensor): the entropy associated with the discrete action
            logp_pi0(torch.Tensor): log probability of chosing the continuous action assuming that
                discrete action 0 was chosen (it does not include the probability of the discrete action)
            logp_pi1(torch.Tensor): log probability of chosing the continuous action assuming that
                discrete action 1 was chosen (it does not include the probability of the discrete action)
            logp_pi2(torch.Tensor): log probability of chosing the continuous action assuming that
                discrete action 2 was chosen (it does not include the probability of the discrete action)
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

        #output the heads (average and standard deviation of the gaussian and the 
        # discrete probabilities)
        mu0 = self.mu0_layer(out)
        mu1 = self.mu1_layer(out)
        mu2 = self.mu2_layer(out)
        log0_std = self.log0_std_layer(out)
        log1_std = self.log1_std_layer(out)
        log2_std = self.log2_std_layer(out)
        log0_std = torch.clamp(log0_std, LOG_STD_MIN, LOG_STD_MAX)
        log1_std = torch.clamp(log1_std, LOG_STD_MIN, LOG_STD_MAX)
        log2_std = torch.clamp(log2_std, LOG_STD_MIN, LOG_STD_MAX)
        std0 = torch.exp(log0_std)
        std1 = torch.exp(log1_std)
        std2 = torch.exp(log2_std)
        p = self.p_layer(out)
        #added this clamp to prevent the argument of softmax from exploding. Might help
        #by giving zero gradient when the probability want to go lower than PROBS_MIN
        p = torch.clamp(p, PROBS_MIN,PROBS_MAX)
        #renormalize the probabilities after clamping
        p = p / p.sum(-1, keepdim=True)

        #Sample the distribution
        b_distribution = Categorical(probs=p)
        pi0_distribution = Normal(mu0, std0)
        pi1_distribution = Normal(mu1, std1)
        pi2_distribution = Normal(mu2, std2)
        if deterministic:
            # Only used for evaluating policy at test time.
            b_action = torch.argmax(p, dim=-1).type(torch.float32)
            pi0_action = mu0
            pi1_action = mu1
            pi2_action = mu2
        else:
            b_action = b_distribution.sample().type(torch.float32)
            pi0_action = pi0_distribution.rsample()
            pi1_action = pi1_distribution.rsample()
            pi2_action = pi2_distribution.rsample()
            
        #if necessary, compute the log of the probabilities
        if with_logprob:
            logp_pi0 = pi0_distribution.log_prob(pi0_action).sum(axis=-1)
            logp_pi1 = pi1_distribution.log_prob(pi1_action).sum(axis=-1)
            logp_pi2 = pi2_distribution.log_prob(pi2_action).sum(axis=-1)
            #change of distribution when going from gaussian to Tanh
            logp_pi0 -= (2*(np.log(2) - pi0_action - F.softplus(-2*pi0_action))).sum(axis=1)
            logp_pi1 -= (2*(np.log(2) - pi1_action - F.softplus(-2*pi1_action))).sum(axis=1)
            logp_pi2 -= (2*(np.log(2) - pi2_action - F.softplus(-2*pi2_action))).sum(axis=1)
            #change of distribution when going from Tanh to the interval [act_lower_limit,act_upper_limit]
            logp_pi0 -= torch.log(0.5*(self.act_upper_limit-self.act_lower_limit)) 
            logp_pi1 -= torch.log(0.5*(self.act_upper_limit-self.act_lower_limit)) 
            logp_pi2 -= torch.log(0.5*(self.act_upper_limit-self.act_lower_limit)) 
            #compute entropy of the discrete action alone
            p_entropy = b_distribution.entropy()
        else:
            logp_pi0 = None
            logp_pi1 = None
            logp_pi2 = None
            p_entropy = None

        #Apply Tanh to the sampled gaussian    
        pi0_action = torch.tanh(pi0_action)
        pi1_action = torch.tanh(pi1_action)
        pi2_action = torch.tanh(pi2_action)
        #Apply the shift of the action to the correct interval
        pi0_action = self.act_lower_limit + 0.5*(pi0_action + 1.)*(self.act_upper_limit-self.act_lower_limit)
        pi1_action = self.act_lower_limit + 0.5*(pi1_action + 1.)*(self.act_upper_limit-self.act_lower_limit)
        pi2_action = self.act_lower_limit + 0.5*(pi2_action + 1.)*(self.act_upper_limit-self.act_lower_limit)

        return b_action, pi0_action, pi1_action, pi2_action, p, p_entropy, logp_pi0, logp_pi1, logp_pi2

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
