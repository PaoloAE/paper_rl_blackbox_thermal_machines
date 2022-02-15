from __future__ import print_function
import sys
import os
sys.path.append(os.path.join('..','src'))
import numpy as np
from gym import spaces
import sac_tri_phys_envs

"""
This module contains gym.Env environments that can be trained using sac_tri.SacTrain. These envs have 1
discrete action with 3 options, and 1 continuous action. Tri stands for the 3 possible discrete choices.

In these environments, the state is a numpy array of floats, containing the sequence of the last N actions.
Its shape is (act_dim, N), where act_dim=4 is the size of the continuous action (1) + 3, since the discrete
action is one-hot encoded.  N is the number of actions defining the state.

These environments, besides being proper gym.Env, MUST satisfy these additional requirements:
    1) __init__ must accept a single dict with all the parameters necessary to define the environment.
    2) implement set_current_state(state). Functions that takes a state as input, and sets the environment to that 
    3) the step() function must provide in the info_dict the key "multi_obj" corresponding to a np.array with the
     value of the two objectives being optimized.
"""

class HarmonicEnginePowEntropy(sac_tri_phys_envs.HarmonicEngineLogStatePowEntropyNonBlackbox):
    """
        Gym.Env representing a quantum harmonic oscillator based heat engine. See jupyter/harmonic_engine.ipynb
        and the results section of the manuscript for additional info. The equation for the evolution of the state are 
        taken from  https://doi.org/10.1088/1367-2630/8/5/083.

        Args:
            env_params is a dictionary that must contain the following: 
            "g0" (float): \Gamma of bath 0
            "g1" (float): \Gamma of bath 1
            "b0" (float): inverse temperature \beta of bath 0
            "b1" (float): inverse temperature \beta of bath 1
            "min_u" (float): minimum value of action u
            "max_u" (float): maximum value of action u
            "w0" (float): \omega_0
            "dt" (float): timestep \Delta t
            "a" (float): value of the weight c
            "p_coeff" (float): the cooling power is multiplied by this factor
            "entropy_coeff" (float): the entropy production is multiplied by this factor
        """
    def __init__(self, env_params):
        super().__init__(env_params)

        #load the "state_steps" parameter
        self.state_steps = env_params["state_steps"]

        #load parameters to discourage not using both baths
        self.min_temp_steps = env_params["min_temp_steps"]
        self.discourage_coeff = env_params["discourage_coeff"]
 
        #get the dimension of the action space
        self.disc_act_n = self.action_space[0].n
        self.cont_act_dim = self.action_space[1].shape[0]
        self.act_dim = self.disc_act_n + self.cont_act_dim

        #create the empty state variable
        self.seq_state = np.zeros((self.act_dim, self.state_steps))

        #override the observation space with the new specifications
        zeros = np.zeros((self.disc_act_n, self.state_steps), dtype=np.float32)
        ones = np.ones((self.disc_act_n, self.state_steps), dtype=np.float32)
        self.observation_space = spaces.Box( 
            low = np.concatenate([zeros ,np.transpose(np.broadcast_to(self.action_space[1].low, (self.state_steps, self.cont_act_dim)))],axis=0 ),
            high = np.concatenate([ones, np.transpose(np.broadcast_to(self.action_space[1].high, (self.state_steps,self.cont_act_dim)))], axis=0)  )
        
        #initialize the state with a sequence of random actions
        self.set_current_state(self.return_random_state())
    
    def current_state(self):
        """
        Returns the current state as the type specificed by self.observation_space.
        By overriding it, super().step() returns the new state.
        """

        return self.seq_state

    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac_tri.SacTrain.load_train() to properly load a saved training session.

        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        self.apply_actions_array(state)
    
    def step(self, action):
        """ Evolves the state for a timestep depending on the chosen action

        Args:
            action (type specificed by self.action_space): the action to perform on the environment
            
        Raises:
            Exception: action out of bound

        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, i.e. the weighed average between power and
                negative entropy production during the current timestep
            end(bool): whether the episode ended (these environments never end)
            info_dict: dictionary with the following key:
                "multi_obj"(np.array): array with power and negative entropy production
        """
        #append the latest action to the state
        self.add_action_to_state(action)

        #call the step from inherited environment
        s, r, end, info_dict = super().step(action)
         
        #Add a penalty if too little steps are spent with each bath
        if self.seq_state[0,:].sum() < self.min_temp_steps:
            r -= self.discourage_coeff*(self.min_temp_steps-self.seq_state[0,:].sum())/self.min_temp_steps
        if self.seq_state[1,:].sum() < self.min_temp_steps:
            r -= self.discourage_coeff*(self.min_temp_steps-self.seq_state[1,:].sum())/self.min_temp_steps
    
        return s, r, end, info_dict

    def apply_actions_array(self, actions_array):
        """
        Applies a sequence of actions to the environment. It returns the same data 
        as step(), but summed over all applied actions (so the cumulative quantities)

        Args:
            actions_array (np.Array): an array of actions. First elements are applied first
        """
        tot_r = 0.
        tot_multi_obj = 0.
        for i in range(actions_array.shape[1]):
            disc_action = np.argmax( actions_array[:self.disc_act_n,i] )
            cont_action = actions_array[self.disc_act_n:,i]
            s,r,_,info_dict = self.step((disc_action, cont_action ))
            tot_r += r
            tot_multi_obj += info_dict["multi_obj"]
        return s, tot_r, False, {"multi_obj": tot_multi_obj}

    def add_action_to_state(self, action):
        """
        Adds an action to the state array that is the sequence of actions.

        Args:
            action (type specificed by self.action_space): the action to add 
        """
        disc_action = np.zeros(self.disc_act_n, dtype=np.float32)
        disc_action[action[0]] = 1.
        action_to_insert = np.concatenate([disc_action, action[1]] )

        self.seq_state = np.roll(self.seq_state,-1,axis=1)
        self.seq_state[:,-1] = action_to_insert

    def return_random_state(self):
        """generates and returns a random state (uniformly distributed)"""
        
        #generate discrete actions
        disc_actions = np.array([ self.action_space[0].sample() for i in range(self.state_steps) ])
        
        #convert to one hot encoding
        disc_actions_hot = np.zeros((self.disc_act_n,self.state_steps), dtype=np.float32)
        disc_actions_hot[disc_actions, np.arange(self.state_steps)] = 1.

        #generate continuous actions
        cont_actions = np.transpose(np.array([ self.action_space[1].sample() for i in range(self.state_steps) ], dtype=np.float32))

        #concatenate the actions
        return np.concatenate([disc_actions_hot, cont_actions], axis=0)


    


