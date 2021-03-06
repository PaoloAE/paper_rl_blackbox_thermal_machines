from __future__ import print_function
import gym
from gym import spaces
import numpy as np
import dataclasses

"""
This module contains gym.Env environments that only handle continuous actions, and that use as state space the actual density
matrix of the physical (phys) system. Therefore, they cannot be used with the blackbox model. However, this module is used
by src/sac_multi_envs.py to define environments that can be trained using sac_multi.SacTrain. Multi stands for "multiobjective",
meaning that these enviroments provide two objectives to be optimized, and allow for a variable weight c. These environments,
besides being proper gym.Env, MUST satisfy these additional requirements:
    1) __init__ must accept a single dict with all the parameters necessary to define the environment.
    2) implement set_current_state(state). Functions that takes a state as input, and sets the environment to that 
    3) the step() function must take as input (action, a_weight), where action is the action, and a_weight is the
     value of the weight c (that indeed can vary during training). 
    4) the step() function must provide in the info_dict the key "multi_obj" corresponding to a np.array with the
     value of the two objectives being optimized.
    5) Optional: if we want to evaluate the average coherence generated by a cycle, the step() function should proving
    in the info_dict the key "coherence" corresponding to a float with the average coherence during the timestep.
            np.array([power,entropy], dtype=np.float32), "coherence": coherence}
"""

class CoherentQubitFridgePowEntropyNonBlackbox(gym.Env):
    """
    
    Gym.Env representing a refrigerator based on a qubit where the sigma_x component is fixed, and the
    sigma_z prefactor is the only continuous controllable parameter. See 
    jupyter/superconducting_qubit_refrigerator.ipynb and the "Results" section of the manuscript
    for additional info. The equation for the evolution are derived from the Lindblad equation given in
    https://doi.org/10.1103/PhysRevB.100.035407
    The reward is a weighed average between the cooling power out of bath 1, and the negative entropy
    production. One must specify inverse temperatures such that b0 <= b1.
    This environment cannot be trained by sac_multi.SacTrain.
    It is used to define the environment CoherentQubitFridgePowEntropy that instead is compatible with sac_multi.SacTrain.

    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): g of bath 0
        "g1" (float): g of bath 1
        "b0" (float): inverse temperature \beta of bath 0
        "b1" (float): inverse temperature \beta of bath 1
        "q0" (float): quality factor of bath 0
        "q1" (float): quality factor of bath 1
        "e0" (float): E_0
        "delta" (float): \Delta
        "w0" (float): resonance frequency of bath 0
        "w1" (float): resonance frequency of bath 1
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "dt" (float): timestep \Delta t
        "p_coeff" (float): the cooling power is multiplied by this factor
        "entropy_coeff" (float): the entropy production is multiplied by this factor
    """

    @dataclasses.dataclass
    class State:
        """
        Data object representing the state of the environment. We use as state a full 
        description of the density matrix, given by rh0_ee, Re[rho_ge], Im[rho_ge], and the 
        last chosen action. In Python, we denote these with (p, re_p, im_p, u).
        """
        p: float = 0.
        re_p: float = 0.
        im_p: float = 0.
        u: float = 0.

    #constant variables used for the computation in step()
    c_mat = np.array([[0.,0.,1.],
                    [-1j,1j,0.],
                    [1.,1.,0.]])
    c_mat_inv = np.linalg.inv(c_mat)

    def __init__(self, env_params):
        super().__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()

        Raises:
            assert: temperature of bath 0 must be greater or equal than bath 1
        """
        self.g0 = env_params["g0"]
        self.g1 = env_params["g1"]
        self.b0 = env_params["b0"]
        self.b1 = env_params["b1"]
        self.q0 = env_params["q0"]
        self.q1 = env_params["q1"]
        self.e0 = env_params["e0"]
        self.delta = env_params["delta"]        
        self.w0 = env_params["w0"]
        self.w1 = env_params["w1"]
        self.min_u = env_params["min_u"] if "min_u" in env_params else env_params["min_q"]
        self.max_u = env_params["max_u"] if "max_u" in env_params else env_params["max_q"]
        self.dt = env_params["dt"]
        self.p_coeff = env_params["p_coeff"]
        self.entropy_coeff = env_params["entropy_coeff"]
        self.state = self.State()

        #check if the temperatures are such that bath 1 is colder than bath 0
        assert self.b0 <= self.b1

        #set the observation and action spaces
        self.observation_space = spaces.Box( low=np.array([0.,-0.5,-0.5, self.min_u], dtype=np.float32),
                                            high=np.array([1.,0.5,0.5,self.max_u], dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                                            high=np.array([self.max_u],dtype=np.float32), dtype=np.float32)
 
        #reset the state of the environment
        self.reset_internal_state_variables()

    def reset(self):
        """resets the state of the environment"""
        self.reset_internal_state_variables()
        return self.current_state()
    
    def step(self, action, a_weight):
        """ Evolves the state for a timestep depending on the chosen action

        Args:
            action (type specificed by self.action_space): the action to perform on the environment
            a_weight (float): the value of the weight c

        Raises:
            Exception: action out of bound

        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, i.e. the weighed average between cooling power and
                negative entropy production during the current timestep
            end(bool): whether the episode ended (these environments never end)
            info_dict: dictionary with the following two keys:
                "multi_obj"(np.array): array with power and negative entropy production
                "coherence"(float): coherence in instantaneous eigenstates in current timestep
        """

        #check if action in range
        if not self.action_space.contains(action):
            raise Exception(f"Action {action} out of bound")

        #read the action
        new_u = action[0]
        
        #compute the effect of the quench in the control
        dtheta = self.theta(new_u) - self.theta(self.state.u)
        rotation = np.array([ [np.cos(2*dtheta), -np.sin(2*dtheta)],
                         [np.sin(2*dtheta), np.cos(2*dtheta)  ]])
        vec1 = np.array([ self.state.p-0.5, self.state.re_p ])
        vec2 = np.array([0.5, 0.])
        temp_p, temp_re_p  = rotation @ vec1 + vec2
        
        #compute the effect of the constant part of duration dt
        de = self.de(new_u)
        init_state = np.array([temp_p, temp_re_p,self.state.im_p])
        b_element = self.s0(-de) + self.s1(-de)
        s_tot = self.s0(de) + self.s1(de) + b_element
        a = np.array([[ -s_tot, 0., 0.  ],
                    [0., -0.5*s_tot, -de],
                    [0., de, -0.5*s_tot  ]  ])
        b = np.array([b_element, 0., 0.])
        a_eigen = np.array([0.5*(-2.*1j*de-s_tot),0.5*(2*1j*de-s_tot),-s_tot ])
        exp_mat = np.diag(np.exp(a_eigen*self.dt))
        a_inv_b = np.linalg.inv(a) @ b
        sol = np.real( self.c_mat @ exp_mat @ self.c_mat_inv @ ( init_state + a_inv_b ) - a_inv_b )

        #i compute the integral of p between 0 and dt
        exp_mat_integrated = np.diag(  (np.expm1(a_eigen*self.dt))/a_eigen   )
        sol_integrated = np.real( self.c_mat @ exp_mat_integrated @ self.c_mat_inv @ ( init_state + a_inv_b ) - a_inv_b*self.dt )
        p_integrated = sol_integrated[0]

        #compute the heat exchanged with the two baths
        qc = de*self.s1(-de) -de*p_integrated*(self.s1(de)+self.s1(-de))/self.dt
        qh = de*self.s0(-de) -de*p_integrated*(self.s0(de)+self.s0(-de))/self.dt
        
        #compute power, negative entropy production, and reward
        power = self.p_coeff*qc
        entropy = self.entropy_coeff*(qc*self.b1 + qh*self.b0)
        reward = a_weight*power + (1-a_weight)*entropy
      
        #update the state
        self.state.p, self.state.re_p, self.state.im_p = sol
        self.state.u = new_u

        #compute the coherence
        #this is the entropy of the diagonal state
        coherence = -self.state.p*np.log(self.state.p)-(1.-self.state.p)*np.log(1.-self.state.p)
        #i subtract the entropy of the full state
        p1 = 0.5*( 1 - np.sqrt( 1. + 4.*(self.state.im_p**2 + self.state.re_p**2+self.state.p**2-self.state.p  )))
        p2 = 0.5*( 1 + np.sqrt( 1. + 4.*(self.state.im_p**2 + self.state.re_p**2+self.state.p**2-self.state.p  )))
        coherence -= -p1*np.log(p1)-p2*np.log(p2)

        return self.current_state(), reward, False, {"multi_obj": 
            np.array([power,entropy], dtype=np.float32), "coherence": coherence}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.current_state())

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        return np.array([self.state.p, self.state.re_p, self.state.im_p, self.state.u] , dtype=np.float32)

    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        to properly load a saved training session.

        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        self.state.p, self.state.re_p, self.state.im_p, self.state.u = state

    def reset_internal_state_variables(self):
        """ Sets the initial values for the state """        
        
        #set initial population to average temperature and choose random action b
        avg_b = 2./(1./self.b0 + 1./self.b1 )
        random_u =  self.action_space.sample()[0]

        #set the 4 state variables
        self.state.p = self.peq( self.de(random_u) ,avg_b)
        self.state.re_p = 0.
        self.state.im_p = 0.
        self.state.u = random_u

    def de(self,u):
        """
        Returns the instantaneous energy gap of the qubit 

        Args:
            u (float): value of the control
            
        Returns:
            de (float): instantaneous energy gap of the qubit  
        """
        return 2. * self.e0 * np.sqrt(self.delta**2 + u**2)

    def theta(self,u):
        """
        Variable parameterizing the instantaneous eigenstates of the qubit. See
        https://doi.org/10.1103/PhysRevB.100.035407 for details.

        Args:
            u (float): value of the control

        Returns:
        theta (float): value of the angle parameterizing the instantaneous eigenstates
            of the qubit
        """
        return 0.5*np.arctan(self.delta/u)
      
    def peq(self, de, b):
        """
        Thermal equilibrium probability of being in the excited state

        Args:
            de (float): energy gap of the qubit
            b (float): inverse temperature

        Returns:
            peq (float): thermal equilibrium probability of being in the excited state
        """
        return 1. / (1. + np.exp(b*de) )

    def s0(self, de):
        """
        Noise power spectrum of bath 0. See Eq. (13) of the manuscript

        Args:
            de (float): energy gap of the qubit

        Returns:
        s0 (float): noise power spectrum of bath 0
        """
        return 0.5 * self.g0 * de / (1-np.exp(-self.b0*de)) / ( 1 + self.q0**2 * ( de/self.w0 - self.w0/de )**2 )

    def s1(self, de):
        """
        Noise power spectrum of bath 1. See Eq. (13) of the manuscript

        Args:
            de (float): energy gap of the qubit

        Returns:
        s1 (float): noise power spectrum of bath 1
        """
        return 0.5 * self.g1 * de / (1-np.exp(-self.b1*de)) / ( 1 + self.q1**2 * ( de/self.w1 - self.w1/de )**2 )

    def square_policy(self, state, u0, u1):
        """
        WARNING: this function is only used for testing.
        It represents a policy that applies a square policy alternating between u0
        and u1 at each dt step.

        Args:
            state: environment state
            u0 (float): one value of u in the square cycle
            u1 (float): other value of u in the square cycle

        Returns:
            action: the action to perform on the environment
        """
        last_u = state[3]
        d0 = np.abs(last_u-u0)
        d1 = np.abs(last_u-u1)
        if d0<d1:
            return np.array([u1], dtype=np.float32)
        else:
            return np.array([u0], dtype=np.float32)

    def trapezoid_policy(self, state, u0, u1, inter_steps=10, wait_steps=10):
        """
        WARNING: this function is only used for testing. It should not be trusted and it is quite hacky.
        It represents a trapezoidal policy.

        Args:
            state: environment state
            u0 (float): smallest value of u in the trapezoidal cycle
            u1 (float): largerst value of u in the trapezoidal cycle
            inter_steps (int): how many steps necessary to ramp from u0 to u1 (and viceversa)
            wait_steps (int): number of steps to spend at u0 and u1

        Returns:
            action: the action to perform on the environment
        """
        assert u0<u1
        inter_steps += 2
        if not "_trapez_up" in dir(self):
            self.trapez_up = True
            self.trapez_waited_steps = 0
        last_q = state[3]
        q_vals = np.linspace(u0,u1,inter_steps)
        nearest_index = np.argmin(np.abs(q_vals-last_q))
        #if at bottom
        if nearest_index == 0:
            self.trapez_waited_steps += 1
            #if should start going  up again
            if self.trapez_waited_steps >= wait_steps:
                self.trapez_waited_steps = 0
                self.trapez_up = True
                q_ind = 1
            else:
                #if I should wait
                q_ind = 0
        #if at top
        elif nearest_index == inter_steps -1:
            self.trapez_waited_steps += 1
            #if should start going  up again
            if self.trapez_waited_steps >= wait_steps:
                self.trapez_waited_steps = 0
                self.trapez_up = False
                q_ind = inter_steps-2
            else:
                #if I should wait
                q_ind = inter_steps-1
        else:
            #if i'm doing the ramp
            if self.trapez_up:
                q_ind =nearest_index +1
            else:
                q_ind =nearest_index -1
        return np.array([q_vals[q_ind]])

