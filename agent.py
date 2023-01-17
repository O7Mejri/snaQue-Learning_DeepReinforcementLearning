import random
from collections import deque

import numpy as np
import torch

from game import BLOCK_SIZE, Direction, Point, SnakeGame
from model import Linear_QNet, QTrain
from send_plots import plotting

import matplotlib.pyplot as plt
from IPython import display as diss

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self) -> None:
        self.gameNb = 0
        self.epsilon = 0 # randomness variable
        self.gamma = 0.9 # discout rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft if memory exceeded
        #11 states for input 3 actions for output, hidden layers dont ma"er
        self.model = Linear_QNet(11, 256, 3) 
        self.trainer = QTrain(self.model, lr=LR, gamma=self.gamma)
    
    def getState(self, game, blocks):
        head = game.snake[0]
        ptL = Point(head.x - blocks, head.y)
        ptR = Point(head.x + blocks, head.y)
        ptU = Point(head.x, head.y - blocks)
        ptD = Point(head.x, head.y + blocks)
        
        dirL = game.direction == Direction.LEFT
        dirR = game.direction == Direction.RIGHT
        dirU = game.direction == Direction.UP
        dirD = game.direction == Direction.DOWN
        
        state = [
            #danger ahead
            (dirL and game.is_collision(ptL)) or
            (dirR and game.is_collision(ptR)) or
            (dirU and game.is_collision(ptU)) or
            (dirD and game.is_collision(ptD)),
            
            #danger right
            (dirL and game.is_collision(ptU)) or
            (dirR and game.is_collision(ptD)) or
            (dirU and game.is_collision(ptR)) or
            (dirD and game.is_collision(ptL)),
            
            #danger left
            (dirL and game.is_collision(ptD)) or
            (dirR and game.is_collision(ptU)) or
            (dirU and game.is_collision(ptL)) or
            (dirD and game.is_collision(ptR)),
            
            #directions
            dirL,
            dirR,
            dirU,
            dirD,
            
            #food
            game.food.x < game.head.x, #food is at left
            game.food.x > game.head.x, #food is at right
            game.food.y < game.head.y, #food is at up
            game.food.y > game.head.y, #food is at down            
            
        ]
        
        return np.array(state, dtype=int) 
                
    
    def remember(self, state, action, reward, nextState, over):
        self.memory.append((state, action, reward, nextState, over))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory
            
        # states, actions, rewards, nextStates, overs = zip(*mini_sample)
        # self.trainer.train_step(states, actions, rewards, nextStates, overs)
        # equivalent to for states, actions, re... in mini_samples:
        #   self.train.train_st.....
        for state, action, reward, nextState, over in mini_sample:
            self.trainer.train_step(state, action, reward, nextState, over)
            
    def train_short_memory(self, state, action, reward, nextState, over):
        self.trainer.train_step(state, action, reward, nextState, over)

    def getAction(self, state, randomness=True):
        # random moves: tradeoff exploration//exploitation
        if randomness:
            self.epsilon = 80 - self.gameNb #random hardcoded number
        else:
            self.epsilon = 0
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else: 
            state0 = torch.tensor(state, dtype=torch.float)
            predic = self.model(state0)
            move = torch.argmax(predic).item()
            final_move[move] = 1
            
        return final_move
    
##### Functions ####
def GA():
    agent = Agent()
    agent.model.load()
    agent.model.eval()
    game = SnakeGame()
    while True:
        state_old = agent.getState(game, BLOCK_SIZE)
        action = agent.getAction(state_old, randomness=False)
        reward, game_over, score = game.play_step(action)
        if game_over:
            game.reset()  


def train():
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0
    
    agent = Agent()
    game = SnakeGame()
    
    while True:
        state_old = agent.getState(game, BLOCK_SIZE) #get old state
        final_move = agent.getAction(state_old) #get the move
        reward, over, score = game.play_step(final_move) #perform move
        state_new = agent.getState(game, BLOCK_SIZE) #get new state
        #train short term
        agent.train_short_memory(state_old, final_move, reward, state_new, over)
        #remember
        agent.remember(state_old, final_move, reward, state_new, over)
        
        if over:
            #train long term
            game.reset()
            agent.gameNb += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print(f'Game: {agent.gameNb} \n Score: {score} \n Record: {record} \n')
            
            # Plotting
            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.gameNb
            plot_mean_score.append(mean_score)
            plotting(plot_score, plot_mean_score)       
        
def main():
    agent = Agent()
    agent.model.load()
    agent.model.eval()
    game = SnakeGame()
    while True:
        state_old = agent.getState(game, BLOCK_SIZE)
        action = agent.getAction(state_old, randomness=False)
        reward, game_over, score = game.play_step(action)
        if game_over:
            game.reset()        
        
if __name__ == '__main__':
    
    #****   DEEP REINFORCEMENT LEARNING  ***#
    
    # if you want to train
    # train()
    
    # if you want to test current model
    # main()
    
    
    
    
    #**** Genetic Algorithm  **** #
    
    GA()


