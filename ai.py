import copy
import random

from game import Game, states

HIT = 0
STAND = 1
DISCOUNT = 0.95 #This is the gamma value for all value calculations

class Agent:
    def __init__(self):

        # For MC values
        self.MC_values = {} # Dictionary: Store the MC value of each state
        self.S_MC = {}      # Dictionary: Store the sum of returns in each state
        self.N_MC = {}      # Dictionary: Store the number of samples of each state
        # MC_values should be equal to S_MC divided by N_MC on each state (important for passing tests)

        # For TD values
        self.TD_values = {}  # Dictionary: Store the TD value of each state
        self.N_TD = {}       # Dictionary: Store the number of samples of each state

        # For Q-learning values
        self.Q_values = {}   # Dictionary: Store the Q-Learning value of each state and action
        self.N_Q = {}        # Dictionary: Store the number of samples of each state for each action

        # Initialization of the values
        for s in states:
            self.MC_values[s] = 0
            self.S_MC[s] = 0
            self.N_MC[s] = 0
            self.TD_values[s] = 0
            self.N_TD[s] = 0
            self.Q_values[s] = [0,0] # First element is the Q value of "Hit", second element is the Q value of "Stand"
            self.N_Q[s] = [0,0] # First element is the number of visits of "Hit" at state s, second element is the Q value of "Stand" at s

        # Game simulator
        # NOTE: see the comment of `init_cards()` method in `game.py` for description of the initial game states       
        self.simulator = Game()

    # NOTE: do not modify this function
    # This is the fixed policy given to you, for which you need to perform MC and TD policy evaluation. 
    @staticmethod
    def default_policy(state):
        user_sum = state[0]
        user_A_active = state[1]
        actual_user_sum = user_sum + user_A_active * 10
        if actual_user_sum < 14:
            return 0
        else:
            return 1

    # NOTE: do not modify this function
    # This is the fixed learning rate for TD and Q learning. 
    @staticmethod
    def alpha(n):
        return 10.0/(9 + n)
   
    #TODO: Take one step of transition in the game simulator
    #Hint: Take the given action, and return the next state given by the game engine. 
    #Hint: Useful functions: self.simulator.act_hit, self.simulator.act_stand, self.simulator.state 
    #Hint: If a state is terminal ("game_over"), i.e., taking any action from it doesn't lead to any next state, then you can return None
    #Hint: You need the act_hit and act_stand functions in game.py. Note that they are already generating random next cards. 
    #Hint: You can keep track the reward of states with this function as well, e.g., as one of the return values
    #Hint: After this function, you can also define another function that simulates one full trajectory, but it's optional
    def make_one_transition(self, action):
        if self.simulator.game_over():
            return None

        if action == HIT:
            self.simulator.act_hit()
        else:
            self.simulator.act_stand()

        return self.simulator.state

    #TODO: Implement MC policy evaluation
    def MC_run(self, num_simulation, tester=False):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):

            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "MC")
            self.simulator.reset()  # The simulator is already reset for you for each new trajectory

            # TODO
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: self.simulator.state gives you the current state of the trajectory
            # Hint: Use the "make_one_transition" function to take steps in the simulator, and keep track of the states
            # Hint: Go through game.py file and figure out which functions will be useful
            # Make sure to update self.MC_values, self.S_MC, self.N_MC for the autograder
            # Don't forget the DISCOUNT

            trajectory = []

            while True:
                state = self.simulator.state
                trajectory.append(state)

                if self.simulator.game_over():
                    break

                action = self.default_policy(state)
                self.make_one_transition(action)

            G = 0

            for state in reversed(trajectory):

                if state == (0, 0, 0):
                    reward = 1
                elif state == (1, 0, 0):
                    reward = -1
                else:
                    reward = 0

                G = reward + DISCOUNT * G

                self.S_MC[state] += G
                self.N_MC[state] += 1
                self.MC_values[state] = self.S_MC[state] / self.N_MC[state]
    #TODO: Implement TD policy evaluation
    def TD_run(self, num_simulation, tester=False):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):

            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "TD")
            self.simulator.reset()

            # TODO
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: self.simulator.state gives you the current state of the trajectory
            # Hint: Use the "make_one_transition" function to take steps in the simulator, and keep track of the states
            # Hint: Go through game.py file and figure out which functions will be useful
            # Hint: The learning rate alpha is given by "self.alpha(...)"
            # Make sure to update self.TD_values and self.N_TD for the autograder
            # Don't forget the DISCOUNT

            while True:
                state = self.simulator.state

                if state == ( 0,0,0 ):
                    self.N_TD[state] += 1
                    alpha = self.alpha(self.N_TD[state])
                    self.TD_values[state] += alpha * ( 1 - self.TD_values[state] )
                    break

                if state == ( 1,0,0 ):
                    self.N_TD[state] += 1
                    alpha = self.alpha(self.N_TD[state])
                    self.TD_values[state] += alpha * ( -1 - self.TD_values[state] )
                    break

                action = self.default_policy(state)
                next_state = self.make_one_transition(action)

                if next_state == (0, 0, 0):
                    target = DISCOUNT * 1
                elif next_state == (1, 0, 0):
                    target = DISCOUNT * -1
                else:
                    target = DISCOUNT * self.TD_values[next_state]
 
                self.N_TD[state] += 1
                alpha = self.alpha(self.N_TD[state])
                self.TD_values[state] += alpha * (target - self.TD_values[state])
                
    #TODO: Implement Q-learning
    def Q_run(self, num_simulation, tester=False, epsilon=0.4):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):

            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "Q")
            self.simulator.reset()

            # TODO
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: self.simulator.state gives you the current state of the trajectory
            # Hint: Use the "make_one_transition" function to take steps in the simulator, and keep track of the states
            # Hint: Go through game.py file and figure out which functions will be useful
            # Hint: The learning rate alpha is given by "self.alpha(...)"
            # Hint: Implement epsilon-greedy method in "self.pick_action(...)"
            # Important: When calling pick_action, use the given parameter epsilon=0.4 to match the autograder
            # Make sure to update self.Q_values, self.N_Q for the autograder
            # Don't forget the DISCOUNT
            while True:
                state = self.simulator.state

                if state == (0, 0, 0):
                    for action in [HIT, STAND]:
                        self.N_Q[state][action] += 1
                        alpha = self.alpha(self.N_Q[state][action])
                        self.Q_values[state][action] += alpha * (1 - self.Q_values[state][action])
                    break

                if state == (1, 0, 0):
                    for action in [HIT, STAND]:
                        self.N_Q[state][action] += 1
                        alpha = self.alpha(self.N_Q[state][action])
                        self.Q_values[state][action] += alpha * (-1 - self.Q_values[state][action])
                    break
               
                action = self.pick_action(state, epsilon)
                next_state = self.make_one_transition(action)

                if next_state == (0, 0, 0):
                    target = DISCOUNT * 1
                elif next_state == (1, 0, 0):
                    target = DISCOUNT * -1
                else:
                    target = DISCOUNT * max(self.Q_values[next_state])

                self.N_Q[state][action] += 1
                alpha = self.alpha(self.N_Q[state][action])
                self.Q_values[state][action] += alpha * (target - self.Q_values[state][action])

    #TODO: Implement epsilon-greedy policy
    def pick_action(self, s, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)

        hit_q = self.Q_values[s][HIT]
        stand_q = self.Q_values[s][STAND]

        if hit_q > stand_q:
            return HIT
        elif stand_q > hit_q:
            return STAND
        else:
            return HIT


    ####Do not modify anything below this line####

    #Note: do not modify
    def autoplay_decision(self, state):
        hitQ, standQ = self.Q_values[state][HIT], self.Q_values[state][STAND]
        if hitQ > standQ:
            return HIT
        if standQ > hitQ:
            return STAND
        return HIT #Before Q-learning takes effect, just always HIT

    # NOTE: do not modify
    def save(self, filename):
        with open(filename, "w") as file:
            for table in [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q]:
                for key in table:
                    key_str = str(key).replace(" ", "")
                    entry_str = str(table[key]).replace(" ", "")
                    file.write(f"{key_str} {entry_str}\n")
                file.write("\n")

    # NOTE: do not modify
    def load(self, filename):
        with open(filename) as file:
            text = file.read()
            MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text, _  = text.split("\n\n")
            
            def extract_key(key_str):
                return tuple([int(x) for x in key_str[1:-1].split(",")])
            
            for table, text in zip(
                [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q], 
                [MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text]
            ):
                for line in text.split("\n"):
                    key_str, entry_str = line.split(" ")
                    key = extract_key(key_str)
                    table[key] = eval(entry_str)

    # NOTE: do not modify
    @staticmethod
    def tester_print(i, n, name):
        print(f"\r  {name} {i + 1}/{n}", end="")
        if i == n - 1:
            print()
