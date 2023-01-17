import random
from collections import namedtuple
from enum import Enum

import numpy as np
import pygame

pygame.init()

font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (20,20,20)

violet = (148, 0, 211)
indigo = (75, 0, 130)
blue = (30, 30, 220)
green = (0, 255, 0)
yellow = (250, 255, 0)
orange = (255, 150, 0)
red = (255, 0, 0)

colors = [violet, indigo, blue, green, yellow, orange, red]

BLOCK_SIZE = 20
SPEED = 40

class SnakeGame:
    
    def __init__(self, w=800, h=800):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Python made with Python')
        self.clock = pygame.time.Clock()
        self.reset()
        
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                    Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_itr = 0
    
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_itr += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         self.direction = Direction.LEFT
            #     elif event.key == pygame.K_RIGHT:
            #         self.direction = Direction.RIGHT
            #     elif event.key == pygame.K_UP:
            #         self.direction = Direction.UP
            #     elif event.key == pygame.K_DOWN:
            #         self.direction = Direction.DOWN
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        reward = 0
        # 3. check if game over
        game_over = False
        if self.is_collision() or self.frame_itr>100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        # hits boundary
        if pt is None:
            pt = self.head
        if pt.x > self.w-BLOCK_SIZE or pt.x < 0 or pt.y > self.h-BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        idx = 0
        for pt in self.snake:
            
            pygame.draw.rect(self.display, colors[idx%7], pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            idx += 1
            # pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        
        img = pygame.image.load(r'X:\Code\Python\SnakeAI\apple.png')
        img = pygame.transform.scale(img, (BLOCK_SIZE,BLOCK_SIZE))
        self.display.blit(img, (self.food.x, self.food.y))
        # pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        moves = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = moves.index(self.direction)
        if np.array_equal(action, [1,0,0]):
            new_dir = moves[idx]
        elif np.array_equal(action, [0,1,0]):
            next_idx = (idx+1)%4
            new_dir = moves[next_idx]
        else:
            next_idx = (idx-1)%4
            new_dir = moves[next_idx]
            
        self.direction = new_dir            
            
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

# if __name__ == '__main__':
#     game = SnakeGame()
    
#     # game loop
#     while True:
#         game_over, score = game.play_step()
        
#         if game_over == True:
#             break
        
#     print('Final Score', score)
        
        
#     pygame.quit()
