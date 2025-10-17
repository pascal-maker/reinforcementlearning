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
    def __init__(self, nr_states, nr_actions, alpha, gamma, exploration_rate, decay):
        print('exploration rate', exploration_rate)
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.decay = decay
        self.nr_states = nr_states
        self.nr_actions = nr_actions
        self.state = 0
        self.possibleActions = [0, 1, 2, 3]
        self.q_table = np.zeros((nr_states, nr_actions))-100

    def compute_action(self,state):
        self.state = state
        self.exploration_rate = self.exploration_rate*self.decay
        self.r = random.random()
        if self.r <= self.exploration_rate:
            self.action = random.sample(self.possibleActions,1)[0]
            print('exploration')
        else:
            print(self.r, self.exploration_rate, 'exploitation')
            self.action = np.argmax(self.q_table[self.state,:])

        return self.action

    def update_qtable(self, state, new_state, reward):
        self.reward = reward
        self.state = state
        self.new_state = new_state

        self.r = random.random()
        if self.r <= self.exploration_rate:
            self.next_q_value = self.q_table[self.new_state, random.sample(self.possibleActions,1)[0]]
        else:
            self.next_q_value = self.q_table[self.new_state, np.argmax(self.q_table[self.new_state, :])]

        self.q_table[self.state, self.action] = (1 - self.alpha) * self.q_table[
            self.state, self.action] + self.alpha * (self.reward + self.gamma * self.next_q_value)

#------------ Agent parameters -----------------
alpha = 0.1
gamma = 0.8
decay = 0.9998
exploration_rate = 0.999
nr_states = 16
nr_actions = 4


env = BombEnv()
env.reset()
q_agent = Qagent(nr_states, nr_actions, alpha, gamma, exploration_rate, decay)

for i_episode in range(1000):
    #env.render()
    state = env.reset()
    print('initial state = ', state)
    for t in range(200):
        action =  q_agent.compute_action(state)
        print('initial action ', action)
        new_state, reward, done = env.step(action)
        print('old state = ', state, 'new state = ', new_state)
        if new_state != state:
           q_agent.update_qtable(state, new_state, reward)

        #print('reward = ', reward, ' in new state = ', new_state, 'done = ', done, 'from previous state ', state)

        state = new_state
        #time.sleep(0.3)
        env.render()
        print('\n', new_state)

        if done == True:
            break

print(q_agent.q_table)

# Test final q-table
done = False
state = env.reset()
env.render()
stopcounter = 0
while (done == False):
    q_agent.exploration_rate = 0
    action = q_agent.compute_action(state)
    new_state, reward, done = env.step(action)
    state = new_state
    stopcounter +=1
    if stopcounter > 25:
        done = True
    env.render()