"""
ESSEC | CENTRALESUPELEC 
DSBA M2 
Reinforcement Learning (2024, March)
Individual Assignment

Authors: 
PETER KESZTHELYI 

Text Flappy Bird Agents implemented. 

References:
    - Richard S. Sutton and Andrew G. Barto - Reinforcement Learning
    - Class materials of "RL - Apprentissage par renforcement - CentraleSupelec (2024)"
"""

from __future__ import print_function
from abc import ABCMeta, abstractmethod
import numpy as np


"""
   ---------------------------
    Base Class
   ---------------------------
"""
class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info= {}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """


"""
    ---------------------------
    Monte Carlo Agent
    ---------------------------
"""
class FlappyMonteCarloAgent(BaseAgent):
    """Implements a Monte Carlo based agent.
    """
    def agent_init(self, agent_init_info):
        """Initialize the FlappyMonteCarloAgent.
        
        Args:
            agent_init_info (dict): Parameters for agent initialization:
                - num_states (int): Number of states.
                - num_actions (int): Number of actions.
                - epsilon (float): Exploration parameter.
                - eps_decay (float): Rate of epsilon decay per episode.
                - eps_min (float): Minimum value of epsilon.
                - step_size (float): Step-size for updating Q-values.
                - discount (float): Discount factor.
                - seed (int): Seed for random number generation.
        """
        # Store initialization parameters
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info['eps_start']
        self.eps_decay = agent_init_info['eps_decay']
        self.eps_min = agent_init_info['eps_min']
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        
        # Initialize Q-values and state dictionary
        self.q = np.zeros((self.num_states, self.num_actions))
        self.state_dict = {}
        
        # Initialize episode memory
        self.episode = []
        
    def agent_start(self, state):
        """Start a new episode.
        
        Args:
            state (int): Initial state.
        
        Returns:
            action (int): Action selected by the agent.
        """
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
        
        # Map state to index
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]
        
        # Choose action using epsilon-greedy strategy
        current_q = self.q[state_idx, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        # Store state-action pair in episode memory
        self.episode.append((state_idx, action))
        
        return action

    def agent_step(self, reward, state):
        """Take a step in the environment.
        
        Args:
            reward (float): Reward received.
            state (int): New state.
        
        Returns:
            action (int): Action selected by the agent.
        """
        # Map state to index
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]

        # Choose action using epsilon-greedy strategy
        current_q = self.q[state_idx, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        # Store state-action pair in episode memory
        self.episode.append((state_idx, action))
        
        return action

    def agent_end(self, reward):
        """Handle the end of an episode.
        
        Args:
            reward (float): Reward received upon termination.
        """
        G = 0  # Initialize return
        # Iterate over the episode in reverse order
        for t in reversed(range(len(self.episode))):
            state, action = self.episode[t]
            G = self.discount * G + reward  # Calculate return
            # Update Q-value using Monte Carlo update rule
            self.q[state, action] += self.step_size * (G - self.q[state, action])

        # Clear episode memory
        self.episode = []
    
    def argmax(self, q_values):
        """Select action with highest value, breaking ties randomly.
        
        Args:
            q_values (numpy array): Array of action-values.
        
        Returns:
            action (int): Selected action.
        """
        top = float("-inf")
        ties = []
        # Find actions with highest value
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)


"""
   ---------------------------
    Expected Sarsa Agent
   ---------------------------
"""
class FlappyESAgent(BaseAgent):
    """Implements an Expected SARSA agent.
    """
    def agent_init(self, agent_init_info):
        """Initialize the FlappyESAgent with given parameters.
        
        Args:
            agent_init_info (dict): Dictionary containing initialization parameters:
                - num_states (int): Number of states.
                - num_actions (int): Number of actions.
                - epsilon (float): Epsilon parameter for exploration.
                - step_size (float): Step-size.
                - discount (float): Discount factor.
        """
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info['eps_start']
        self.eps_decay = agent_init_info['eps_decay']
        self.eps_min = agent_init_info['eps_min']
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        
        self.q = np.zeros((self.num_states, self.num_actions)) # Action-value estimates
        self.state_dict = {} # Mapping of states to indices
        
    def agent_start(self, state):
        """Take the initial action when the episode starts.
        
        Args:
            state (int): Initial state.
        
        Returns:
            action (int): The first action the agent takes.
        """
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min) # Decay epsilon
        
        # Map state to index
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]
        
        # Choose action using epsilon-greedy strategy
        current_q = self.q[state_idx, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        self.prev_state_idx = state_idx
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):
        """Take a step in the environment.
        
        Args:
            reward (float): Reward received.
            state (int): New state.
        
        Returns:
            action (int): The action the agent takes.
        """
        # Map state to index
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]
        
        # Choose action using epsilon-greedy strategy
        current_q = self.q[state_idx, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        # Update Q-values using the expected SARSA equation
        expected_q = self.calculate_expected_q(current_q)
        self.q[self.prev_state_idx, self.prev_action] += self.step_size * (
            reward + self.discount * expected_q - self.q[self.prev_state_idx, self.prev_action]
        )
        
        self.prev_state_idx = state_idx
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """Run when the agent terminates.
        
        Args:
            reward (float): Reward received upon termination.
        """
        # Update Q-values for the last step
        self.q[self.prev_state_idx, self.prev_action] += self.step_size * (
            reward - self.q[self.prev_state_idx, self.prev_action]
        )
        
    def argmax(self, q_values):
        """argmax with random tie-breaking.
        
        Args:
            q_values (numpy array): Array of action-values.
        
        Returns:
            action (int): Action with the highest value.
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)
    
    def calculate_expected_q(self, current_q):
        """Calculate the expected Q-value.
        
        Args:
            current_q (numpy array): Array of current action-values.
        
        Returns:
            expected_q (float): Expected Q-value.
        """
        q_max = np.max(current_q)
        epsilon_prob = np.ones(self.num_actions) * self.epsilon / self.num_actions
        greedy_prob = (current_q == q_max) * (1 - self.epsilon) / np.sum(current_q == q_max)
        pi = epsilon_prob + greedy_prob
        expected_q = np.sum(pi * current_q)
        return expected_q