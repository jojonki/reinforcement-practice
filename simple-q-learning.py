# -*- coding: utf-8 -*-
"""
Created on Thu May  5, 2016

@author: jonki
"""

import numpy as np
import random
import sys

# sample ref
# http://mnemstudio.org/path-finding-q-learning-tutorial.htm

# Reward matrix
R = np.array([
[-1, -1, -1, -1,  0,  -1],
[-1, -1, -1,  0, -1, 100],
[-1, -1, -1,  0, -1,  -1],
[-1,  0,  0, -1,  0,  -1],
[ 0, -1, -1,  0, -1, 100],
[-1,  0, -1, -1,  0, 100]
])

# Initial Q-value
Q = np.zeros((6,6))

LEARNING_COUNT = 1000
GAMMA = 0.8
GOAL_STATE = 5

class QLearning(object):
    def __init__(self):
        print "init"
    
    def learn(self):
        # set a start state randomly
        state = self._getRandomState()
        for i in range(LEARNING_COUNT):        
            # extract possible actions in state
            possible_actions = self._getPossibleActionsFromState(state)
            
            # choise an action from possible actions randomly
            action = random.choice(possible_actions)        
            
            # Update Q-value
            # Q(s,a) = r(s,a) + Gamma * max[Q(next_s, possible_actions)]
            next_state = action # in this example, action value is same as next state
            next_possible_actions = self._getPossibleActionsFromState(next_state)
            max_Q_next_s_a = self._getMaxQvalueFromStateAndPossibleActions(next_state, next_possible_actions)
            Q[state, action] = R[state, action] + GAMMA * max_Q_next_s_a
            
            state = next_state
            
            # If an agent reached a goal state, restart an episode from a random start state
            if state == GOAL_STATE:
                state = self._getRandomState()
    
    def _getRandomState(self):
        return random.randint(0, R.shape[0] - 1)
      
    def _getPossibleActionsFromState(self, state):
        if state < 0 or state >= R.shape[0]: sys.exit("invaid state: %d" % state)
        return list(np.where(np.array(R[state] != -1)))[0]
    
    def _getMaxQvalueFromStateAndPossibleActions(self, state, possible_actions):
        return max([Q[state][i] for i in (possible_actions)])
            
    def dumpQvalue(self):
        print Q.astype(int) # convert float to int for redability

    def runGreedy(self, start_state = 0):
        print "===== START ====="
        state = start_state
        while state != GOAL_STATE:
            print "current state: %d" % state
            possible_actions = self._getPossibleActionsFromState(state)
            
            # get best action which maximaizes Q-value(s, a)
            max_Q = 0
            best_action_candidates = []
            for a in possible_actions:            
                if Q[state][a] > max_Q:
                    best_action_candidates = [a,]
                    max_Q = Q[state][a]
                elif Q[state][a] == max_Q:
                    best_action_candidates.append(a)
            
            # get a best action from candidates randomly
            best_action = random.choice(best_action_candidates)
            print "-> choose action: %d" % best_action
            state = best_action # in this example, action value is same as next state
        print "state is %d, GOAL!!" % state
            
if __name__ == "__main__":
    QL = QLearning()
    QL.learn()
    
    QL.dumpQvalue()
    
    for s in range(R.shape[0]-2):
        QL.runGreedy(s)