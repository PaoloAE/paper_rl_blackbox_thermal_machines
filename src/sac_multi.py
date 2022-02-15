from __future__ import print_function
import os
import numpy as np
import torch
import torch.optim as optim
import pickle
import shutil
import sys
import warnings
from itertools import chain
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import logging

sys.path.append(f'../src')
import plotting
import core_multi
import sac_multi_envs
import extra

"""
This mudule contains the objects used to train quantum thermal machine environments
with 2 multi-objectives (eg power and entropy production) and with 1 continuous action.
The corresponding environments are in sac_multi_envs. All torch tensors that are not integers are torch.float32.
It was written starting from the code:
J. Achiam, Spinning Up in Deep Reinforcement Learning, https://github.com/openai/spinningup (2018).
"""


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents with continuous actions.

    Args:
        obs_dim (tuple(int)): tuple of size 2. First is number of channels (corresponding to the
            size of an action), and second one is number of timesteps defining the state.
        act_dim (int): number of continuous parameters of action space.
        size (int): size of the buffer.
        device (torch.device): which torch device to use.
    """

    def __init__(self, obs_dim, act_dim, size, device):  
        self.obs_buf = torch.zeros((size, obs_dim[0],obs_dim[1]), dtype=torch.float32, device=device)
        self.obs2_buf = torch.zeros((size, obs_dim[0],obs_dim[1]), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rewa_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.rewb_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rewa, rewb, next_obs):
        """
        stores a transition into the buffer. All args are torch.float32.

        Args:
            obs (torch.tensor): the initial state
            act (torch.tensor): the action
            rewa (torch.tensor): the reward corresponding to the first multiobjective
            rewwb (torch.tensor): the reward corresponding to the second multiobjective
            next_obs (torch.tensor): the next state        
        """
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rewa_buf[self.ptr] = rewa
        self.rewb_buf[self.ptr] = rewb
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        """
        Return a random batch of experience from the buffer.
        The batch index is the leftmost index.

        Args:
            batch_size (int): size of batch
            device (torch.device)
        """
        idxs = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rewa=self.rewa_buf[idxs],
                     rewb=self.rewb_buf[idxs])
        return batch

class SacTrainState(extra.SacTrainState):
    """ 
    This object is here only for backward compatibility when loading older trains.
    See extra.SacTrainState
    """
    def __init__():
        super().__init__()

def state_to_tensor(state, device):
    """ Coverts a numpy state to a torch.tensor """
    return  torch.as_tensor(state, device=device, dtype=torch.float32)

def action_to_tensor(state, device):
    """ Coverts a numpy state to a torch.tensor """
    return  torch.as_tensor(state, device=device, dtype=torch.float32)

class SacTrain(object):
    """
    Main class to train the RL agent on a quantum thermal machine environment
    with 1 continuous action, and with 2 multi-objectives. See environments in
    src/sac_multi_envs.
    This class can create a new training session, or load an existing one.
    It takes care of logging and of saving the training session all in one folder.

    Usage:
        After initialization either
        - call initialize_new_train() to initialize a new training session
        - call load_train() to load an existing training session
    """
        
    #define some constants defining the filestructure of the logs and saved state.
    PARAMS_FILE_NAME = "params.txt"
    S_FILE_NAME = "s.dat"
    POLICY_NET_FILE_NAME = "policy_net.dat"
    TARGET_NET_FILE_NAME = "target_net.dat"
    ERROR_LOG_FILE_NAME = "caught_errors.log"
    STATE_FOLDER_NAME = "state"
    SAVE_DATA_DIR = os.path.join("..", "data")
    SAVED_LOGS_FOLDER = "logs"
    RUNNING_REWARD_FILE_NAME = "running_reward.txt"
    RUNNING_LOSS_FILE_NAME = "running_loss.txt"
    RUNNING_MULTI_OBJ_FILE_NAME = "running_multi_obj.txt"
    ACTIONS_FILE_NAME = "actions.txt"
    SAVED_POLICY_DIR_NAME = "saved_policies"

    #methods that can be called:

    def initialize_new_train(self, env_class, env_params, training_hyperparams, log_info):
        """ Initializes a new training session. Should be called right after initialization.

        Args:
            env_class (gym.Env): class representing the quantum thermal machine environment to learn.
                compatible envoronments are in src/sac_multi_envs
            env_params (dict): parameters used to initialize env_class. See specific env requirements.
            training_hyperparameters (dict): dictionary with training hyperparameters. Must contain the following
                "BATCH_SIZE" (int): batch size
                "LR" (float): learning rate
                "H_START" (float): initial value of average entropy
                "H_END" (float): final value of the average entropy
                "H_DECAY" (float): exponential decay of average entropy
                "A_START" (float): initial value of weight c
                "A_END" (float): final value of the weight c
                "A_DECAY" (float): sigmoid decay of weight c
                "A_MEAN" (float): sigmoid average of the weight c
                "REPLAY_MEMORY_SIZE" (int): size of replay buffer
                "POLYAK" (float): polyak coefficient
                "LOG_STEPS" (int): save logs and display training every number of steps
                "GAMMA" (float): RL discount factor
                "CHANNEL_SIZES" tuple(int): number of channels in each convolution block. Since each block
                    halves the input size (eventually adding padding before halving if odd), the size of 
                    the tuple must be compatible with N, the lenght of the time-series of actions defining
                    the state
                "PI_FC_SIZES" tuple(int): size of the fully connected hidden layers, applied after the 
                    convolutions blocks, producing the policy function
                "Q_FC_SIZES": tuple(int): size of the fully connected hidden layers, applied after the 
                    convolution bloacks, producing the value function Q
                "SAVE_STATE_STEPS" (int): saves complete state of trainig every number of steps
                "INITIAL_RANDOM_STEPS" (int): number of initial uniformly random steps
                "UPDATE_AFTER" (int): start minimizing loss function after initial steps
                "UPDATE_EVERY" (int): performs this many updates every this many steps
                "USE_CUDA" (bool): use cuda for computation
                "ALPHA_RESET_VAL" (float): if the temperature parameter becomes negative, it is reset to this
                    value. It should be a small quantity, e.g. 1.e-6
            log_info (dict): specifies logging info. Must contain
                    "log_running_reward" (bool): whether it should log running reward 
                    "log_running_loss" (bool): whether it should log running loss
                    "log_actions" (bool): whether it should log actions
                    "log_running_multi_obj" (bool): whether it should log the multiobjectives
                    "extra_str" (str): extra string to append to training folder
        """
        
        
        #initialize a SacTrainState to store the training state 
        self.s = extra.SacTrainState()

        #save input parameters
        self.s.save_data_dir = self.SAVE_DATA_DIR
        self.s.env_params = env_params
        self.s.training_hyperparams = training_hyperparams
        self.s.log_info = log_info 

        #setup the torch device
        if self.s.training_hyperparams["USE_CUDA"]:
            if torch.cuda.is_available():
                self.s.device = torch.device("cuda")
            else:
                warnings.warn("Cuda is not available. Will use cpu instead.")
                self.s.device = torch.device("cpu")
        else:
            self.s.device = torch.device("cpu")

        #create environment
        self.env = env_class(self.s.env_params)

       #add the environment name to the env_params dictionry
        self.s.env_params["env_name"] = self.env.__class__.__name__

        #set the training steps_done to zero
        self.s.steps_done = 0

        #reset the environment and save the initial state
        self.s.state = state_to_tensor(self.env.reset(), self.s.device)

        #initialize logging session
        self.s.log_session = self.initialize_log_session()

        #setup the memory replay buffer
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]
        self.s.memory = ReplayBuffer(obs_dim, act_dim, self.s.training_hyperparams["REPLAY_MEMORY_SIZE"],self.s.device)

        #initialize the NNs
        self.initialize_nns()

        #set the optimizer
        self.create_optimizer()
        
    def load_train(self, log_folder, specific_state_folder = None, no_train=False):
        """
        Loads a training session that had been previously saved. The training
        sessions are saved as folders numbered as "0", "1",... By default, the latest
        one is loaded.
        
        Args:
            log_folder (str): folder of the training session
            specific_state_folder (str): can load a specific save. If None, loads the latest one.
            no_train (bool): if False, it creates a new logging folder where all new saves and loggings
                are located. If true, it doesn't create a new folder, but cannot train anymore.        
        """
        save_dir_path = os.path.join(log_folder, self.STATE_FOLDER_NAME)
        if specific_state_folder is not None:
            #load the folder where the save is
            save_dir_path = os.path.join(save_dir_path, specific_state_folder)
        else:
            #must find the latest folder if not specificed
            path = Path(save_dir_path)
            folders = [dir.name for dir in path.iterdir() if dir.is_dir()]
            index = int(folders[0])
            for folder in folders:
                index = max(index, int(folder))
            save_dir_path = os.path.join(save_dir_path, str(index))

        #load self.s
        with open(os.path.join(save_dir_path, self.S_FILE_NAME), 'rb') as input:
            self.s = pickle.load(input)
        
        #add an attribute to the replay buffer for back-compatibility
        if not hasattr(self.s.memory, "device"):
            self.s.memory.device = self.s.device
        
        #load the environment
        env_method = self.return_env_class_from_name()
        self.env = env_method(self.s.env_params)
        try:
            self.env.set_current_state(self.s.state.cpu().numpy())
        except:
            self.env.set_current_state(self.s.state)
        
        #create the nns
        self.initialize_nns()

        #load the policy net
        self.ac.load_state_dict(torch.load(os.path.join(save_dir_path, self.POLICY_NET_FILE_NAME)))
        
        #i stop here if i don't want to further optimize the model
        if no_train:
            self.s.log_session.log_dir = log_folder
        else:
            #load the targer net
            self.ac_targ.load_state_dict(torch.load(os.path.join(save_dir_path, self.TARGET_NET_FILE_NAME)))
            #load the optimizer
            self.create_optimizer()
            #now that everything is loaded, i create a new LogSession, and copy in the old logs
            self.s.log_session = self.initialize_log_session(reset_running_vars = False)
            for file in Path(os.path.join(save_dir_path, self.SAVED_LOGS_FOLDER)).iterdir():
                shutil.copy(str(file), os.path.join(self.s.log_session.log_dir, file.name))

    def train(self, steps, output_plots = True):
        """
        Runs "steps" number of training steps. Takes care of saving and logging. It can be called multiple
        times and it will keep training the same model.

        Args:
            steps (int): number of training steps to perform
            output_plots (bool): if true, it will output a plot with all the running logs every LOG_STEPS.
        """
        for _ in range(steps):
            
            #choose an action (random uniform for first INITIAL_RANDOM_STEPS, then according to policy )
            if self.s.steps_done > self.s.training_hyperparams["INITIAL_RANDOM_STEPS"]:
                a = self.get_action(self.s.state)
            else:
                a = action_to_tensor(self.env.action_space.sample(), self.s.device)

            #perform the action on environment
            o2_np, r, _, info_dict = self.env.step(a.cpu().numpy(), self.current_a_weight())
            o2 = state_to_tensor(o2_np,self.s.device)
            
            # Store experience to replay buffer
            self.s.memory.store(self.s.state, a, float(info_dict["multi_obj"][0]),float(info_dict["multi_obj"][1]), o2)
            
            #move to the next state
            self.s.state = o2

            #increase the counter
            self.s.steps_done += 1

            # Perform NN parameters updates
            if self.s.steps_done > self.s.training_hyperparams["UPDATE_AFTER"] and \
                self.s.steps_done % self.s.training_hyperparams["UPDATE_EVERY"] == 0:
                for _ in range(self.s.training_hyperparams["UPDATE_EVERY"]):
                    #collect a batch of experience to use for training
                    batch = self.s.memory.sample_batch(self.s.training_hyperparams["BATCH_SIZE"])
                    try:
                        #perform the update using the batch
                        q_loss, pi_loss, entropy = self.update(data=batch)
                        #update logging: running loss, value of alpha, entropy, and of the weight a
                        self.s.running_loss[0] += (1.-self.s.training_hyperparams["GAMMA"])*(q_loss - self.s.running_loss[0])
                        self.s.running_loss[1] += (1.-self.s.training_hyperparams["GAMMA"])*(pi_loss - self.s.running_loss[1])
                        self.s.running_loss[2] += (1.-self.s.training_hyperparams["GAMMA"])*(self.current_alpha() - self.s.running_loss[2])
                        self.s.running_loss[3] += (1.-self.s.training_hyperparams["GAMMA"])*(entropy - self.s.running_loss[3])
                        self.s.running_loss[4] += (1.-self.s.training_hyperparams["GAMMA"])*(self.current_a_weight() - self.s.running_loss[4])
                    except RuntimeError as e:
                        #there could be an error doing updates, e.g. covariance singular. In such case it is logged
                        logging.error(f"Exception at step {self.s.steps_done} during self.update: {e}")

            #update running logging: reward and action
            self.s.running_reward += (1.-self.s.training_hyperparams["GAMMA"])*(r - self.s.running_reward)
            self.s.actions.append([self.s.steps_done] + a.tolist() ) 
            
            #if present, update running estimate of the multiobjective environments
            if "multi_obj" in info_dict:
                if self.s.running_multi_obj is None:
                    self.s.running_multi_obj = np.zeros(len(info_dict["multi_obj"]) ,dtype=np.float32)
                self.s.running_multi_obj += (1.-self.s.training_hyperparams["GAMMA"])*(info_dict["multi_obj"]
                                                -self.s.running_multi_obj  )
            
            #if its time to save the logs
            if self.s.steps_done % self.s.training_hyperparams["LOG_STEPS"] == 0 :
                #update log files
                self.update_log_files()
                
                #plot the logs
                if output_plots:
                    self.plot_logs()

            #if it's time to save the full training state   
            if self.s.steps_done % self.s.training_hyperparams["SAVE_STATE_STEPS"] == 0:
                self.save_full_state()

    def save_full_state(self):
        """
        Saves the full state to file, so that training can continue exactly from here by loading the file.
        The saved session is placed in a folder inside STATE_FOLDER_NAME, named using an ascending index
        0, 1, ... Largest index is the most recent save.
        """
        #folder where the session is saved
        path_location = os.path.join( self.s.log_session.state_dir, str(len(list(Path(self.s.log_session.state_dir).iterdir()))) )
        #create the folder to save the state
        Path(path_location).mkdir(parents=True, exist_ok=True)
        #save self.s state object
        with open(os.path.join(path_location, self.S_FILE_NAME), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.s, output, pickle.HIGHEST_PROTOCOL)
        #save policy_net params
        torch.save(self.ac.state_dict(), os.path.join(path_location, self.POLICY_NET_FILE_NAME))
        #save target_net params
        torch.save(self.ac_targ.state_dict(), os.path.join(path_location, self.TARGET_NET_FILE_NAME))
        #copy over the logging folder 
        saved_logs_path = os.path.join(path_location, self.SAVED_LOGS_FOLDER)
        Path(saved_logs_path).mkdir(parents=True, exist_ok=True)
        for file in Path(self.s.log_session.log_dir).iterdir():
            if not file.is_dir() :
                shutil.copy(str(file), os.path.join(saved_logs_path, file.name ))

    def evaluate_current_policy(self, deterministic, steps=1000, suppress_show=False, gamma=None,actions_to_plot=400,
                                save_policy_to_file_name = None,actions_ylim=None,dont_clear_output=False):
        """
        creates a copy of the environment, and evaluates the current policy in a deterministic or probabilistic way.
        The evaluation is done with the final weight, not the current one.
        It return the final running reward, multiobjective, and generated coherence (if the environment returns it),
        and plots the running rewards and actions. It can also save the chosen actions to a file.
        See extra.test_policy for more details. 

        Args:
            deterministic (bool): if the chosen actions should be deterministic or not
            steps (int): how many steps of the environment to do to evaluate the running return
            suppress_show (bool): if True, it will not plot anything
            gamma (float): the exponential average factor to compute the return and multiobjectives.
                It doesn't have to coincide with the one used for training
            actions_to_plot (int): how many of the latest actions to show in the plots
            save_policy_to_file_name (str): if specified, it will save the chosen actions to file, so they can be plotted
            actions_ylim (tuple(float,float)): the y_lim to plot the actions
            dont_clear_output (bool): if False, it clears the previous plots.
        Returns:
            (np.array): it return the running avg of the reward, and based on the environment, also
                the running average of the multiobjectives, and of the coherence
        """
        #if gamma is not specified, it will use the one used during training
        if gamma is None:
            gamma = self.s.training_hyperparams["GAMMA"]
        #if necessary, create the path to save the actions
        if save_policy_to_file_name is not None:
            save_policy_to_file_name = os.path.join( self.s.log_session.log_dir, self.SAVED_POLICY_DIR_NAME, save_policy_to_file_name)
        #evaluates the policy
        return extra.test_policy(self.return_env_class_from_name(), self.s.env_params,
                     lambda o: self.get_action(torch.as_tensor(o,device=self.s.device,dtype=torch.float32),deterministic=deterministic).cpu().numpy(),
                     gamma, steps=steps, env_state = self.s.state.cpu().numpy(),
                     suppress_show=suppress_show,actions_to_plot=actions_to_plot,save_policy_to_file_name=save_policy_to_file_name,
                     actions_ylim=actions_ylim,dont_clear_output=dont_clear_output,a_weight=self.s.training_hyperparams["A_END"],
                     is_tri=False)

    #Methods that should only be used internally:

    def initialize_log_session(self, reset_running_vars = True):
        """
        creates a folder, named with the current time and date, for logging and saving a training session,
        and saves all the physical parameters and hyperparameters in file PARAMS_FILE_NAME. 
        
        Args:
            reset_running_vars (bool): wether to reset the logged data

        Raises:
            Exception: if the folder for logging already exists
        
        Returns:
            log_session (extra.LogSession): info used by this class to do logging and saving state in the right place
        """
        #reset the running variables
        if reset_running_vars:
            self.s.running_reward = 0.
            self.s.running_loss = np.zeros(5, dtype=np.float32)
            self.s.running_multi_obj = None
            self.s.actions =[]

        #create folder for logging
        now = datetime.now()
        log_dir = os.path.join(self.s.save_data_dir, now.strftime("%Y_%m_%d-%H_%M_%S") + self.s.log_info["extra_str"])
        Path(log_dir).mkdir(parents=True, exist_ok=False)
            
        #create a file with all the environment params and hyperparams
        param_str = ""
        for name, value in chain(self.s.env_params.items(), self.s.training_hyperparams.items()):
            param_str += f"{name}:\t{value}\n"
        param_file = open(os.path.join(log_dir, self.PARAMS_FILE_NAME),"w") 
        param_file.write(param_str)
        param_file.close()
        
        #create files for logging
        running_reward_file = os.path.join(log_dir, self.RUNNING_REWARD_FILE_NAME)
        running_loss_file = os.path.join(log_dir, self.RUNNING_LOSS_FILE_NAME)
        running_multi_obj_file = os.path.join(log_dir, self.RUNNING_MULTI_OBJ_FILE_NAME)
        actions_file = os.path.join(log_dir, self.ACTIONS_FILE_NAME)

        #create folder for saving the state
        state_dir = os.path.join(log_dir, self.STATE_FOLDER_NAME)
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        
        #initialize the logging for errors
        logging.basicConfig(filename= os.path.join(log_dir, self.ERROR_LOG_FILE_NAME), 
            level=logging.ERROR, format="%(asctime)s:\t%(message)s")

        #set default value for logging multiobjective if it's not passed in (for compatibility)
        if not "log_running_multi_obj" in self.s.log_info:
            self.s.log_info["log_running_multi_obj"] = True

        return extra.LogSession(log_dir, state_dir, self.s.log_info["log_running_reward"], self.s.log_info["log_running_loss"],
                            self.s.log_info["log_running_multi_obj"], self.s.log_info["log_actions"], running_reward_file,
                            running_loss_file, running_multi_obj_file, actions_file)

    def initialize_nns(self):
        """ Initializes the NNs for the soft actor critic method """
        #create the main NNs
        self.ac = core_multi.ActorCritic(self.env.observation_space, self.env.action_space,
                                        channel_sizes=self.s.training_hyperparams["CHANNEL_SIZES"],
                                        pi_fc_sizes = self.s.training_hyperparams["PI_FC_SIZES"],
                                        q_fc_sizes = self.s.training_hyperparams["Q_FC_SIZES"],
                                        alpha_reset_val = self.s.training_hyperparams["ALPHA_RESET_VAL"]).to(self.s.device)
        #create the target NNs
        self.ac_targ = deepcopy(self.ac)
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (saved for convenience)
        self.q_params = chain(self.ac.qa1.parameters(),self.ac.qa2.parameters(), self.ac.qb1.parameters(),self.ac.qb2.parameters())

        # Count and print number of variables 
        var_counts = tuple(core_multi.count_vars(module) for module in [self.ac.pi, self.ac.qa1,self.ac.qa2, self.ac.qb1,self.ac.qb2])
        print('\nNumber of parameters: \t pi: %d, \t qa1: %d, \t qa2: %d, \t qb1: %d, \t qb2: %d,\n'%var_counts)

    def create_optimizer(self):
        """ Setup the ADAM optimizer for pi and q"""
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.s.training_hyperparams["LR"])
        self.q_optimizer = optim.Adam(self.q_params, lr=self.s.training_hyperparams["LR"]) 
        self.alpha_optimizer = optim.SGD(self.ac.alpha.parameters(), lr=0.001)

    def current_h(self):
        """ returns the current value of the average entropy h, which decreases exponentially """
        return self.s.training_hyperparams["H_END"] + \
            (self.s.training_hyperparams["H_START"] - self.s.training_hyperparams["H_END"]) * \
            np.exp(-1. * self.s.steps_done / self.s.training_hyperparams["H_DECAY"])

    def current_a_weight(self): 
        """
        returns the current value of the weighe c, that behaves as
        as sigmoid / Fermi distribution
        """
        end = self.s.training_hyperparams["A_END"]
        start = self.s.training_hyperparams["A_START"]
        decay = self.s.training_hyperparams["A_DECAY"]
        mean = self.s.training_hyperparams["A_MEAN"]
        now = self.s.steps_done
        return end + (1./(1. + np.exp((now - mean)/decay))  )*(start - end)

    def current_alpha(self):
        """
        returns the current value of alpha, that is updated automatically by the optimizers
        """
        return self.ac.alpha_no_grad()

    def get_action(self, o, deterministic=False):
        """ Returns an on-policy action based on the state passed in.
        This computation does not compute the gradients.

        Args:
            o (torch.Tensor): state from which to compute action
            deterministic (bool): wether the action should be sampled or deterministic  
        """
        return self.ac.act(o.reshape((-1,o.shape[-2],o.shape[-1])), deterministic)[0]

    def compute_loss_q(self, data):
        """
        Compute the loss function of the q-value functions given a batch of data. This function
        is used to find the gradient of the loss using backprop.

        Args:
            data(dict): dictionary with the following
                "obs" (torch.Tensor): batch of states
                "act" (torch.Tensor): batch of continuous actions
                "rewa" (torch.Tensor): batch of rewards of first objective
                "rewb" (torch.Tensor): batch of rewards of second objective
                "obs2" (torch.Tensor): batch of next states

        Returns:
            (torch.Tensor): the sum of the loss function for both q-values
        """
        #unpack the batched data
        o, a, ra, rb, o2 = data['obs'], data['act'], data['rewa'],data['rewb'], data['obs2']

        #value Q(s,u) of both objectives (a and b)
        qa1 = self.ac.qa1(o,a)
        qa2 = self.ac.qa2(o,a)
        qb1 = self.ac.qb1(o,a)
        qb2 = self.ac.qb2(o,a)

        # Bellman backup for Q functions. The gradient is not taken respect to the target
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            qa1_pi_targ = self.ac_targ.qa1(o2, a2)
            qa2_pi_targ = self.ac_targ.qa2(o2, a2)
            qb1_pi_targ = self.ac_targ.qb1(o2, a2)
            qb2_pi_targ = self.ac_targ.qb2(o2, a2)
            qa_pi_targ = torch.min(qa1_pi_targ, qa2_pi_targ)
            qb_pi_targ = torch.min(qb1_pi_targ, qb2_pi_targ)
            
            #backup values to "fit" - one for each qa and one for qb
            backup_a = ra + self.s.training_hyperparams["GAMMA"] * (qa_pi_targ - self.current_alpha() * logp_a2)
            backup_b = rb + self.s.training_hyperparams["GAMMA"] * (qb_pi_targ - self.current_alpha() * logp_a2)

        # MSE loss against Bellman backup
        loss_qa1 = ((qa1 - backup_a)**2).mean()
        loss_qa2 = ((qa2 - backup_a)**2).mean()
        loss_qb1 = ((qb1 - backup_b)**2).mean()
        loss_qb2 = ((qb2 - backup_b)**2).mean()
        loss_q = loss_qa1 + loss_qa2 + loss_qb1 + loss_qb2

        return loss_q

    def compute_loss_pi(self, data):
        """
        Compute the loss function for the policy given a batch of data. This function
        is used to find the gradient of the loss using backprop.

        Args:
            data(dict): dictionary with the following
                "obs" (torch.Tensor): batch of states

        Returns:
            tuple(torch.Tensor): the loss function for the policy and the average entropy 
                of the policy
        """
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)  
        q1_pi = self.current_a_weight()* self.ac.qa1(o, pi) + (1.-self.current_a_weight())* self.ac.qb1(o, pi)
        q2_pi = self.current_a_weight()* self.ac.qa2(o, pi) + (1.-self.current_a_weight())* self.ac.qb2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.current_alpha() * logp_pi - q_pi).mean()

        return loss_pi, -logp_pi.mean()

    def compute_loss_alpha(self, data):
        """
        Compute the loss function to determine the "temperure" parameter alpha.

        Args:
            data(dict): dictionary with the following
                "obs" (torch.Tensor): batch of states

        Returns:
            (torch.Tensor): the loss function for alpha
        """

        o = data['obs']

        with torch.no_grad():
            _, logp_pi = self.ac.pi(o)

        # alpha loss function
        loss_alpha = -self.ac.alpha()*( logp_pi + self.current_h() ).mean()
        return loss_alpha

    def update(self, data):
        """
        Performs an update of the parameters of both Q, Pi and alpha.

        Args:
            data (dict): batch of experience drawn from replay buffer. See compute_loss_q for details
        
        Return:
             (loss_q(float), loss_pi(float), entropy(float)): the numerical value of
            the loss function for q, for pi, and the average entropy of the policy
        """

        # Update of qa1, qa2, qb1, qb2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data) 
        loss_q.backward()
        self.q_optimizer.step()

        #update of the policy function
        # Freeze Q-networks since they will not be updated
        for p in self.q_params:
            p.requires_grad = False

        #optimze the policy params
        self.pi_optimizer.zero_grad()
        loss_pi, entropy = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so they can be optimized at the next step
        for p in self.q_params:
            p.requires_grad = True

        #optimize the temperature alpha
        self.alpha_optimizer.zero_grad()
        loss_alpha = self.compute_loss_alpha(data)
        loss_alpha.backward()
        self.alpha_optimizer.step()

        #prevents alpha from being negative
        if self.current_alpha() < self.s.training_hyperparams["ALPHA_RESET_VAL"]: 
            logging.error(f"Alpha was reset at step {self.s.steps_done}")
            self.ac.alpha.reset_alpha()

        #update target networks by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # in-place operations
                p_targ.data.mul_(self.s.training_hyperparams["POLYAK"])
                p_targ.data.add_((1 - self.s.training_hyperparams["POLYAK"]) * p.data)

        return loss_q.item(), loss_pi.item(), entropy

    def update_log_files(self):
        """ updates all the log files with the current running reward, running losses, running multi objectives, and actions"""
        #update running reward
        if self.s.log_session.log_running_reward:
            self.append_log_line(f"{self.s.running_reward}", self.s.log_session.running_reward_file, self.s.steps_done)
        #update running loss
        if self.s.log_session.log_running_loss:
            self.append_log_line(np.array_str(self.s.running_loss,999999).replace("[", "").replace("]","")
                                ,self.s.log_session.running_loss_file, self.s.steps_done)
        #update running multi objective (if present)
        if self.s.log_session.log_running_multi_obj and self.s.running_multi_obj is not None:
            self.append_log_line(np.array_str(self.s.running_multi_obj,999999).replace("[", "").replace("]","")
                                ,self.s.log_session.running_multi_obj_file, self.s.steps_done)
        #update the actions
        if self.s.log_session.log_actions: 
            f=open(self.s.log_session.actions_file,'ab')
            np.savetxt(f, self.s.actions)
            f.close()
            self.s.actions = []
  
    def append_log_line(self, data, file, count):
        """appends count and data to file as plain text """
        file_object = open(file, 'a')
        file_object.write(f"{count}\t{data}\n")
        file_object.close()

    def plot_logs(self):
        """Plot the current logs regarding the training progress"""
        plotting.plot_sac_logs(self.s.log_session.log_dir, running_reward_file=self.s.log_session.running_reward_file,
            running_loss_file=self.s.log_session.running_loss_file, actions_file=self.s.log_session.actions_file,
                plot_to_file_line = None, suppress_show=False, save_plot = False, extra_str="",is_tri=False)

    def return_env_class_from_name(self):
        """
        Return the class to create a new environment, given the string
        of the environment class name in self.s.env_params['env_name'].
        Looks in sac_multi_envs for the environment class.

        Raises:
            NameError: if env_name doesn't exist

        Returns:
            Returns the class to create the environment
        """
        if hasattr(sac_multi_envs, self.s.env_params['env_name']):
            return getattr(sac_multi_envs, self.s.env_params['env_name'])
        else:
            raise NameError(f"Environment named {self.s.env_params['env_name']} not found in sac_envs")




