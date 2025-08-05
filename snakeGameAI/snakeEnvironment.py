import numpy as np
import random
from enum import Enum
import pygame

class getDirection(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class snakeEnvironment:
    
    def __init__(self, width=20, height=20, render=True):
        self.width = width
        self.height = height
        self.renderGame = render
        
        if self.renderGame:
            pygame.init()
            self.window = pygame.display.set_mode((width * 20, height * 20))
            pygame.display.set_caption("Snake AI Training")
            self.clock = pygame.time.Clock()
   
    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = getDirection.RIGHT
        self.food = self.generateFood()
        self.score = 0
        self.steps = 0
        self.maxSteps = 100 * len(self.snake)
        
        return self.getState()
        
    def generateFood(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            
            if food not in self.snake:
                return food
                
    def getState(self):
        headX, headY = self.snake[0]
        foodX, foodY = self.food
        
        state = [
            self.isCollision(self.getNextPosition(self.direction)),
            self.isCollision(self.getNextPosition(self.turnRight(self.direction))),
            self.isCollision(self.getNextPosition(self.turnLeft(self.direction))),
            
            self.direction == getDirection.RIGHT,
            self.direction == getDirection.DOWN,
            self.direction == getDirection.LEFT,
            self.direction == getDirection.UP,
            
            foodX < headX,
            foodX > headX,
            foodY < headY,
            foodY > headY
        ]
            
        return np.array(state, dtype=np.float32)
        
    def getNextPosition(self, direction):
        headX, headY = self.snake[0]
        
        if direction == getDirection.UP:
            return (headX, headY - 1)
        elif direction == getDirection.DOWN:
            return (headX, headY + 1)
        elif direction == getDirection.LEFT:
            return (headX - 1, headY)
        elif direction == getDirection.RIGHT:
            return (headX + 1, headY)
            
    def turnRight(self, direction):
        turns = {
            getDirection.UP: getDirection.RIGHT,
            getDirection.RIGHT: getDirection.DOWN,
            getDirection.DOWN: getDirection.LEFT,
            getDirection.LEFT: getDirection.UP
        }
        return turns[direction]
    
    def turnLeft(self, direction):
        turns = {
            getDirection.UP: getDirection.LEFT,
            getDirection.LEFT: getDirection.DOWN,  
            getDirection.DOWN: getDirection.RIGHT,
            getDirection.RIGHT: getDirection.UP  
        }
        return turns[direction]
        
    def isCollision(self, position):
        x, y = position
        
        # Wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        
        # Self collision
        if position in self.snake:
            return True
    
        return False
        
    def step(self, action):
        self.steps += 1
        
        if action == 1:
            self.direction = self.turnRight(self.direction)
        elif action == 2:
            self.direction = self.turnLeft(self.direction)
            
        newHead = self.getNextPosition(self.direction)
        
        if self.isCollision(newHead):
            reward = -10
            done = True
            return self.getState(), reward, done, {'score': self.score}  

        self.snake.insert(0, newHead)
        

        if newHead == self.food:
            self.score += 10
            reward = 10
            self.food = self.generateFood()  
            self.maxSteps = 100 * len(self.snake)
        else:
            self.snake.pop()
            reward = 0 
            
        done = self.steps > self.maxSteps
        
        if done:
            reward = -10
            
        return self.getState(), reward, done, {'score': self.score}
        
    def render(self):
        if not self.renderGame:
            return
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
        black = (0, 0, 0)
        green = (0, 255, 0)
        red = (255, 0, 0)
        blue = (0, 0, 255)
        
        self.window.fill(black)
        
        for i, segment in enumerate(self.snake):
            x, y = segment
            color = green if i == 0 else (0, 255, 0) 
            pygame.draw.rect(self.window, color, (x * 20, y * 20, 19, 19))
            
        x, y = self.food
        pygame.draw.rect(self.window, red, (x * 20, y * 20, 19, 19))
        
        pygame.display.flip()
        self.clock.tick(60)

if __name__ == "__main__":
    env = snakeEnvironment(render=True)
    state = env.reset()
    
    running = True
    for _ in range(1000):
        if not running:
            break
            
        action = random.randint(0, 2)  
        state, reward, done, info = env.step(action)
        env.render()
        

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if done:
            print(f"Game over! Score: {info['score']}")
            state = env.reset()
    
    pygame.quit()
