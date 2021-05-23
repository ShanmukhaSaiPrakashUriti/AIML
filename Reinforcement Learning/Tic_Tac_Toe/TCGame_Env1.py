from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix
        
        # List of combinations of which if sum is 15 can be considered as a win
        self.all_comb = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        # Declaring and Initialising to return the final status
        win = False; # Initialised as false and will be made true if there is a winning combination
        
        
        
        # Verifying win
        for comb in self.all_comb:
            sum = 0
            count=0
            for ind in range(0,3):
                if (np.isnan(curr_state[comb[ind]])==False):
                    sum = sum + curr_state[comb[ind]]
                else:
                    break
            if(sum==15 & count==3):
                win = True
                break;
        return win
 

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        curr_state[curr_action[0]] =curr_action[1]
        return curr_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        next_state = self.state_transition(curr_state, curr_action)
        terminal, status = self.is_terminal(next_state)
        if(terminal == True):
            if(status == 'Win'):
                reward = 10
            else:
                reward = 0
        else:
            action = self.randomAction(next_state)
            next_state = self.state_transition(next_state,action)
            terminal, status = self.is_terminal(next_state)
            if(terminal == True):
                if(status == 'Win'):
                    reward = -10
                else:
                    reward = 0
            else:
                reward = -1
        return reward , next_state, terminal

    def reset(self):
        return self.state
    
    
   
    def randomAction(self, curr_state):
        """Function to do Random action in Tic tac toe as environments choice of entering a number. The selection is a random selection from all the possible values that environment can take."""
        valid_actions = [act for act in self.action_space(curr_state)[1]]
        action = random.choice(valid_actions)
        return action;
        
        
       