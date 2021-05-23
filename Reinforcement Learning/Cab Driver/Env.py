# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
        self.action_space = [(0,0)]+[(sou,dest) for sou in range(1,m+1) for dest in range (1,m+1) if sou!=dest]
        self.state_space = [(loc, time, day) for loc in range(1, m+1) for time in range(t) for day in range(d)]
        self.state_init = random.choice(self.state_space) # initial start state
        self.time_elapsed = 0 # Intial time on rides

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        state_encod = [0 for i in range(m+t+d)] # Creating a encoded state vector of all zeros
        state_encod[state[0] -1] = 1 # Changing the value to 1 to encode location info
        state_encod[m + state[1]] = 1 # Changing the value to 1 encode time information
        state_encod[m + t + state[2]] = 1 # Changing the value to 1 encode date information
        
        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
            """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
            state_encod = [0 for _ in range(m+t+d+m+m)]
            state_encod[state[0]] = 1
            state_encod[m+state[1]] = 1
            state_encod[m+t+state[2]] = 1
            if (action[0] != 0):
                state_encod[m+t+d+action[0]] = 1
            if (action[1] != 0):
                state_encod[m+t+d+m+action[1]] = 1
            return state_encod
            
        
        
         


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        requests = 0
        if location == 1:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12)
        if location == 3:
            requests = np.random.poisson(4)
        if location == 4:
            requests = np.random.poisson(7)
        if location == 5:
            requests = np.random.poisson(8)
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        # Time taken for the cab driver to arriver to arrive at the pickup location from the current location
        t_pickup = 0
        # Actual time taken for the ride - from pickup to drop
        t_ride = 0
        # Time spent when the cab driver goes offline
        offline_time = 0    
        
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]
        
        # When option to go offline is selected
        if ((pickup_loc== 0) and (drop_loc == 0)):
            offline_time = 1
        # when driver is already at the pickup location
        elif (curr_loc == pickup_loc):
            t_ride = Time_matrix[pickup_loc-1][drop_loc-1][curr_time][curr_day]
        # Driver is at a location different from the pickup location
        else:
            t_pickup = Time_matrix[curr_loc-1][pickup_loc-1][curr_time][curr_day]
            time_at_pickup, day_at_pickup = self.update_time_day(curr_time, curr_day, t_pickup)
            t_ride = Time_matrix[pickup_loc-1][drop_loc-1][time_at_pickup][day_at_pickup]

        # Calculate total time as sum of all durations        
        reward = (R * t_ride) - (C * (t_ride + t_pickup + offline_time))
        
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        # To indicate if the 30 hours has been completed
        is_terminal = False
        
        # Initialize various times
        total_time   = 0
        # Time taken for the cab driver to arriver to get to the pickup location from the current location
        t_pickup = 0
        # Actual time taken for the ride - from pickup to drop
        t_ride = 0
        # Time spent when the cab driver goes offline
        offline_time = 0    
        
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]
        
        # When option to go offline is selected
        if ((pickup_loc== 0) and (drop_loc == 0)):
            offline_time = 1
            next_loc = curr_loc
        # when driver is already at the pickup location
        elif (curr_loc == pickup_loc):
            t2 = Time_matrix[pickup_loc-1][drop_loc-1][curr_time][curr_day]
            next_loc = drop_loc
        # Driver is at a location different from the pickup location
        else:
            t_pickup = Time_matrix[curr_loc-1][pickup_loc-1][curr_time][curr_day]
            time_at_pickup, day_at_pickup = self.update_time_day(curr_time, curr_day, t_pickup)
            t_ride = Time_matrix[pickup_loc-1][drop_loc-1][time_at_pickup][day_at_pickup]
            next_loc  = drop_loc

        # Calculate total time as sum of all durations
        total_time = offline_time + t_pickup + t_ride
        next_time, next_day = self.update_time_day(curr_time, curr_day, total_time)
        self.time_elapsed += total_time
        if self.time_elapsed >=720:
            is_terminal = True
        next_state = [next_loc, next_time, next_day]
        return next_state, is_terminal

    def update_time_day(self, time, day, ride_duration):
        """
        Takes in the pickup time and time taken for driver's journey to return
        the state post that journey.
        """
        ride_duration = int(ride_duration)

        if (time + ride_duration) < 24:
            time = time + ride_duration
            # day is unchanged
        else:
            # duration taken spreads over to subsequent days
            # Get the number of days
            num_days = (time + ride_duration) // 24
            
            # convert the time to 0-23 range
            time = (time + ride_duration) % 24             
            
            # Convert the day to 0-6 range
            day = (day + num_days ) % 7

        return time, day
    


    def reset(self):
        self.state_init = random.choice(self.state_space) # initial start state
        self.time_elapsed = 0
        return self.state_init