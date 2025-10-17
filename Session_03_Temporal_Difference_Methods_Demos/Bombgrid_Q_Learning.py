import numpy as np
from termcolor import colored
import random
import time

class BombEnv:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.state = 0
        self.forbidden_step = False
        self.reward_table = np.array([[0,-100,0,100],[0,0,-100,0],[0,0,0,0],[-100,0,0,0]])
        self.done_table = np.array([[False,True,False,True],[False,False,True,False],[False,False,False,False],[True,False,False,False]])
        self.render_table = np.array([['S','B','P','F'],['P','P','B','P'],['P','P','P','P'],['B','P','P','P']])

    def reset(self):
        self.x = 0
        self.y = 0
        return 0

    def step(self, action):
        self.action = action
        if self.action == 0:   # GO LEFT
           if (self.x-1 < 0):
               self.x = self.x
               self.y = self.y
               self.forbidden_step = True
           else:
               self.x = self.x-1
               self.y = self.y
               self.forbidden_step = False
        if self.action == 1:   # GO RIGHT
           if (self.x+1 >3):
               self.x = self.x
               self.y = self.y
               self.forbidden_step = True
           else:
               self.x = self.x +1
               self.y = self.y
               self.forbidden_step = False
        if self.action == 2:  # GO UP
           if (self.y -1 <0):
               self.x = self.x
               self.y = self.y
               self.forbidden_step = True
           else:
               self.x = self.x
               self.y = self.y -1
               self.forbidden_step = False
        if self.action == 3: # GO DOWN
           if (self.y+1 > 3):
               self.x = self.x
               self.y = self.y
               self.forbidden_step = True
           else:
               self.x = self.x
               self.y = self.y + 1
               self.forbidden_step = False
        self.state = 4*self.x +self.y
        self.done = self.done_table[self.y,self.x]
        if self.forbidden_step == True: # give negative reward for bumping into the wall
            self.reward = -1
        else:
            self.reward = self.reward_table[self.y,self.x]-1

        return self.state, self.reward, self.done

    def render(self):
        self.rendering = self.render_table.copy()
        self.rendering[self.y,self.x] ='O'
        print('\n')
        for self.r in range(self.rendering.shape[0]):
            print('')
            for self.c in range(self.rendering.shape[1]):
                if self.rendering[self.r,self.c] == 'O':
                   print(colored('O', 'red'),end = '')
                else:
                   print(self.rendering[self.r,self.c],end = '')

class Qagent:
    def __init__(self, nr_states, nr_actions, alpha, gamma, epsilon, decay):
        print('exploration rate', epsilon)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.nr_states = nr_states
        self.nr_actions = nr_actions
        self.state = 0
        self.possibleActions = [0, 1, 2, 3]
        self.q_table = np.zeros((nr_states, nr_actions))

    def compute_action(self,state):
        self.state = state
        self.epsilon = self.epsilon*self.decay
        self.r = np.random.uniform()
        if self.r <= self.epsilon:
            self.action = random.sample(self.possibleActions,1)[0]
            print('\nexploration')
        else:
            print('\nexploitation')
            self.action = np.argmax(self.q_table[self.state,:])

        return self.action

    def update_qtable(self, state, new_state, reward):
        self.reward = reward
        self.state = state
        self.new_state = new_state

        self.q_table[self.state, self.action] = (1 - self.alpha) * self.q_table[
            self.state, self.action] + self.alpha * (self.reward + self.gamma * self.q_table[
            self.new_state, np.argmax(self.q_table[self.new_state, :])])

#------------ Agent parameters -----------------
nr_of_episodes = 1000
alpha = 0.1
gamma = 0.8
decay = 0.9999
epsilon = 0.999
nr_states = 16
nr_actions = 4


env = BombEnv()
env.reset()
q_agent = Qagent(nr_states, nr_actions, alpha, gamma, epsilon, decay)

for i_episode in range(nr_of_episodes):
    #env.render()
    state = env.reset()
    for t in range(200):

        # compute and execute action
        action =  q_agent.compute_action(state)
        new_state, reward, done = env.step(action)

        # update the q-table
        q_agent.update_qtable(state, new_state, reward)

        #update the state
        state = new_state

        env.render()
        print('\n', new_state)

        if done == True:
            break

print(q_agent.q_table)

# Test final q-table
done = False
state = env.reset()
env.render()
while (done == False):
    q_agent.epsilon = 0
    action = q_agent.compute_action(state)
    new_state, reward, done = env.step(action)
    state = new_state
    env.render()









