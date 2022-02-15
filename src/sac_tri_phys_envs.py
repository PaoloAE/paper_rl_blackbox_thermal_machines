from __future__ import print_function
import gym
from gym import spaces
import numpy as np
import dataclasses

"""
This module contains gym.Env environments that handle 1 discrete action (with 3 choices) and 1 continuous actions, and that
use as state space the actual density matrix of the physical (phys) system. Therefore, they cannot be used with the blackbox
model. However, this module is used by src/sac_tri_envs.py to define environments that can be trained using sac_tri.SacTrain.
Tri stands for the fact that it supports a discrete action with three choices (hot, cold, or unitaty). These environments,
besides being proper gym.Env, MUST satisfy these additional requirements:
    1) __init__ must accept a single dict with all the parameters necessary to define the environment.
    2) implement set_current_state(state). Functions that takes a state as input, and sets the environment to that 
    3) the step() function must provide in the info_dict the key "multi_obj" corresponding to a np.array with the
        value of the two objectives being optimized.
"""

class HarmonicEnginePowEntropyNonBlackbox(gym.Env):
    """
    Gym.Env representing a quantum harmonic oscillator based heat engine. See jupyter/harmonic_engine.ipynb
    and the results section of the manuscript for additional info. The equation for the evolution of the state are 
    taken from  https://doi.org/10.1088/1367-2630/8/5/083.
    WARNING: we recommend using HarmonicEngineLogStatePowEntropyNonBlackbox instead, since it is numerically more stable.

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
    @dataclasses.dataclass
    class State:
        """
        Data object representing the state of the environment. We use as state 
        (h,l,d,u), where (see https://doi.org/10.1088/1367-2630/8/5/083 for details):
            h: expectation value of the Hamiltonian
            l: expectation value of the Lagrangian
            d: expectation value of the position momentum correlation
            u: last chosen continuous action
        """
        h: float = 0.
        l: float = 0.
        d: float = 0.
        u: float = 0.

    def __init__(self, env_params):
        super().__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        self.g0 = env_params["g0"]
        self.g1 = env_params["g1"]
        self.b0 = env_params["b0"]
        self.b1 = env_params["b1"]
        self.min_u = env_params["min_u"] if "min_u" in env_params else env_params["min_q"]
        self.max_u = env_params["max_u"] if "max_u" in env_params else env_params["max_q"]
        self.w0 = env_params["w0"]
        self.dt = env_params["dt"]
        self.a = env_params["a"]
        self.p_coeff = env_params["p_coeff"]
        self.entropy_coeff = env_params["entropy_coeff"]
        self.state = self.State()

        #set the observation and action spaces.
        #WARNING: the low and high values are not accurate, however we never use them in this code. 
        self.observation_space = (spaces.Box( low=np.array([-100., -100., -100., self.min_u],dtype=np.float32),
                                            high=np.array([100., 100., 100., self.max_u],dtype=np.float32), dtype=np.float32)
                                    )
        self.action_space = (spaces.Discrete(3), spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                                            high=np.array([self.max_u],dtype=np.float32), dtype=np.float32))
 
        #reset the state of the environment
        self.reset_internal_state_variables()
        
    def reset(self):
        """ resets the state of the environment """
        self.reset_internal_state_variables()
        return self.current_state()
    
    def step(self, action):
        """ Evolves the state for a timestep depending on the chosen action

        Args:
            action (type specificed by self.action_space): the action to perform on the environment

        Raises:
            Exception: action out of bound

        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, i.e. the weighed average between the power and
                negative entropy production during the current timestep
            end(bool): whether the episode ended (these environments never end)
            info_dict: dictionary with the following key:
                "multi_obj"(np.array): array with power and negative entropy production
        """

        #check if action in range
        if not self.action_space[0].contains(action[0]) or not self.action_space[1].contains(action[1]):
            raise Exception(f"Action {action} out of bound")

        #load action
        b_act = action[0]
        u_act = action[1][0]
        if b_act == 0:
            b_act = self.b0
            g_act = self.g0
        elif b_act == 1:
            b_act = self.b1
            g_act = self.g1
        else:
            b_act = 1.
            g_act = 0.

        #Compute the effect of the quench in omega (section 4.2 of the ref)
        #d doesn't change, while l and h do.
        #The frequency before the quench
        w0 = self.w(self.state.u)
        #the frequency after the quench
        w1 = self.w(u_act)
        a = 0.5*(1. + (w1/w0)**2)
        b = 0.5*(1. - (w1/w0)**2)
        mat = np.array([[a,b],[b,a]])
        vec = np.array([self.state.h,self.state.l])
        self.state.h, self.state.l = (mat @ vec).reshape(-1)

        #I save h before doing the constant part (where heat is exchanged)
        h_prev = self.state.h

        #I do the constant part (adiabat in ref language, section 4.1)
        w_temp = self.w(u_act)
        heq_temp = self.heq(w_temp,b_act)
        self.state.h = np.exp(-g_act*self.dt) * ( self.state.h - heq_temp ) + heq_temp
        a = np.cos(2.*w_temp*self.dt)
        b = np.sin(2.*w_temp*self.dt)
        mat = np.exp(-g_act*self.dt) * np.array([[a, -0.5*w_temp*b],[2.*b/w_temp, a]])
        vec = np.array([self.state.l,self.state.d])
        self.state.l, self.state.d = (mat @ vec).reshape(-1)
        
        #compute the power as the heat absorbed from the baths, which is the energy 
        #difference during the constant part
        power = self.p_coeff*(self.state.h - h_prev)/self.dt

        #compute the entropy production
        entropy = self.entropy_coeff*(power/self.p_coeff)*b_act

        #compute the reward 
        reward = self.a*power + (1-self.a)*entropy

        #finish updating the state
        self.state.u = u_act

        return self.current_state(), reward, False, {"multi_obj": 
            np.array([power,entropy], dtype=np.float32)}   

    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.state)
    
    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac_tri.SacTrain.load_full_state() to properly load a saved training session.

        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        self.state.h, self.state.l, self.state.d, self.state.u = state

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        return np.array([self.state.h, self.state.l, self.state.d, self.state.u] , dtype=np.float32)
           
    def reset_internal_state_variables(self):
        """ Sets the initial values for the state """        

        #set initial population to average temperature and choose random action b
        avg_b = 2./(1./self.b0 + 1./self.b1 )
        random_u =  self.action_space[1].sample()[0]
        
        #set the state to an equilibrium state
        self.state.h = self.heq(self.w(random_u) ,avg_b)
        self.state.l = 0.
        self.state.d = 0.
        self.state.u = random_u

    def heq(self, w, b):
        """
        Thermal equilibrium value of h at frequency w and inverse temperature b

        Args:
            w (float): frequency of the oscillator
            b (float): inverse temperature

        Returns:
            heq (float): thermal equilibrium value of h
        """
        return 0.5 * w / np.tanh(0.5*w*b)

    def w(self, u):
        """
        Frequency of the oscillator

        Args:
            u (float): value of the control
        
        Returns:
            w (float): frequency of the oscillator
        """
        return self.w0 * u

class HarmonicEngineLogStatePowEntropyNonBlackbox(HarmonicEnginePowEntropyNonBlackbox):
    """
    This is a wrapper for the HarmonicEnginePowEntropyNonBlackbox environment to change the state representation
    from (h,l,d,u), used in HarmonicEnginePowEntropyNonBlackbox, to a more numerically stable representation that
    works better when fed into the neural networks. Since h, l and d can take very large values during
    the quenches in the control, we use the log of the absolute value of h,l,d instead (plus a small number
    to prevent divergences). We further use the sign of l and of d as an additional state. The state of h is
    not used since h is always positive. We recommend using this instead of HarmonicEnginePowEntropyNonBlackbox.

    Args:
        env_params: same as HarmonicEnginePowEntropyNonBlackbox
    """

    def __init__(self, env_params):
        super(HarmonicEngineLogStatePowEntropyNonBlackbox,self).__init__(env_params)

        #i specify the new observation space. 100 is wrong, but never used.
        self.observation_space = (spaces.Box( low=np.array([-100., -100., -100., -100., -100., self.min_u],dtype=np.float32),
                                            high=np.array([100., 100., 100.,100., 100., self.max_u],dtype=np.float32), dtype=np.float32))
        self.safety_param =  1e-20

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        log_h = self.safe_log(self.state.h)
        log_abs_l = self.safe_log(np.abs(self.state.l))
        log_abs_d = self.safe_log(np.abs(self.state.d))
        sign_l = np.sign(self.state.l)
        sign_d = np.sign(self.state.d)
        return np.array([log_h, log_abs_l, log_abs_d,sign_l,sign_d, self.state.u] , dtype=np.float32) 

    def set_current_state(self, state):
        """ Sets the current state of the environment """
        log_h, log_abs_l, log_abs_d,sign_l,sign_d, self.state.u = state
        self.state.h = self.safe_exp(log_h)
        self.state.l = sign_l * self.safe_exp(log_abs_l)
        self.state.d = sign_d * self.safe_exp(log_abs_d)

    def safe_log(self, x):
        """ Safe way to take the log of a non-negative quantity """
        return np.log(x + self.safety_param)

    def safe_exp(self,x):
        """ The inverse of _safe_log """
        return np.exp(x)-self.safety_param
        
    def trapezoid_policy(self, state, u0, u1, inter_steps=10, wait_steps=10):
        """
        WARNING: this function is only used for testing. It should not be trusted and it is quite hacky.
        It represents a policy that applies a trapezoidal policy, i.e. an Otto cycle.

        Args:
            state: environment state
            u0 (float): smallest value of u in the trapezoidal cycle
            u1 (float): largerst value of u in the trapezoidal cycle
            inter_steps (int): how many steps necessary to ramp from u0 to u1
            wait_steps (int): number of steps to spend at u0 and u1

        Returns:
            action: the action to perform on the environment
        """
        assert u0<u1
        inter_steps += 2
        if not "_trapez_up" in dir(self):
            self._trapez_up = True
            self._trapez_waited_steps = 0
        last_q = state[5]
        q_vals = np.linspace(u0,u1,inter_steps)
        nearest_index = np.argmin(np.abs(q_vals-last_q))
        #if at bottom
        if nearest_index == 0:
            self._trapez_waited_steps += 1
            #if should start going  up again
            if self._trapez_waited_steps >= wait_steps:
                self._trapez_waited_steps = 0
                self._trapez_up = True
                q_ind = 1
            else:
                #if I should wait
                q_ind = 0
        #if at top
        elif nearest_index == inter_steps -1:
            self._trapez_waited_steps += 1
            #if should start going  up again
            if self._trapez_waited_steps >= wait_steps:
                self._trapez_waited_steps = 0
                self._trapez_up = False
                q_ind = inter_steps-2
            else:
                #if I should wait
                q_ind = inter_steps-1
        else:
            #if i'm doing the ramp
            if self._trapez_up:
                q_ind =nearest_index +1
            else:
                q_ind =nearest_index -1
        if q_ind == 0:
            d_action = 1
        elif q_ind == len(q_vals)-1:
            d_action = 0
        else:
            d_action = 2
        return (d_action, np.array([q_vals[q_ind]]))

