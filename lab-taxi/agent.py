import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy_s = np.ones(self.nA)/nA
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1
        self.episode = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA, p = self.policy_s)

    def update_Q(self,Qsa, Qsa_next, reward, alpha, gamma):
        '''update the action-value function using the most recent timestep)''' 
        return Qsa + (alpha * (reward + (gamma* Qsa_next) - Qsa))

    def epsilon_greedy_probs(self, Q_s, eps=None):
        '''obtains the action probabilities corresponding to epsilon-greedy policy'''
        if eps is not None:
            epsilon = eps
        else:
            epsilon = 0.005
        epsilon = epsilon/self.episode
        policy_s = np.ones(self.nA) * epsilon/ self.nA
        policy_s[np.argmax(Q_s)] = 1-epsilon + (epsilon / self.nA)
        return policy_s

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        alpha = self.alpha
        gamma = self.gamma
        epsilon = self.epsilon

        # create variable Q
        Q = self.Q
        # crate update policy
        self.policy_s = self.epsilon_greedy_probs(Q[next_state], 0.005)
        # forecast next action based on epsilon greedy prob
        next_action = np.random.choice(np.arange(self.nA),p=self.policy_s)
        # set the episode
        self.episode += 1
        self.Q[state][action] = self.update_Q(Q[state][action],
                                                    np.dot(Q[next_state] ,self.policy_s),
                                                    reward, alpha, gamma )