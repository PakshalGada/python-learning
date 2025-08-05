import pygame
import random

pygame.init()
width=720
height=540
blockSize=20
gridWidth = width // blockSize
gridHeight = height // blockSize
screen=pygame.display.set_mode((width,height))
pygame.display.set_caption("Snake")
clock = pygame.time.Clock()
FPS = 15

black=(0,0,0)
red=(255,0,0)
green=(0,255,0)
white=(255,255,255)

font = pygame.font.SysFont("arial", 24)

snake=[((gridWidth)//2,(gridHeight)//2)]

speed=1
dx=0
dy=0
score=0

def spawnFood(snake):
    while True:
        foodX = random.randrange(0, gridWidth)
        foodY = random.randrange(0, gridHeight)
        if (foodX, foodY) not in snake:
            return foodX, foodY

foodX, foodY = spawnFood(snake)

run=True

while run:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            run=False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT and dx==0:
                dx, dy = -speed, 0
            elif event.key == pygame.K_RIGHT and dx==0:
                dx, dy = speed, 0
            elif event.key == pygame.K_UP and dy==0:
                dx, dy = 0, -speed
            elif event.key == pygame.K_DOWN and dy==0:
                dx, dy = 0, speed
                
    snakeX,snakeY=snake[0]  
    newX=snakeX+dx
    newY=snakeY+dy
    
    if newX<0 or newX>=gridWidth or newY<0 or newY>=gridHeight:
        snakeX = gridWidth // 2  
        snakeY = gridHeight // 2
        snake = [(snakeX, snakeY)]
        dx, dy = 0, 0
        score = 0
        foodX, foodY = spawnFood(snake)  
    else:
        snake.insert(0, (newX, newY))
        if newX == foodX and newY == foodY:
            score += 1
            foodX, foodY = spawnFood(snake)
        else:
            snake.pop()
   
    screen.fill(black)
    
    for x,y in snake:
        pygame.draw.rect(screen, green, (x*blockSize, y*blockSize, blockSize, blockSize))
    
    pygame.draw.rect(screen, red, (foodX*blockSize, foodY*blockSize, blockSize, blockSize))
    
    score_text = font.render(f"Score: {score}", True, white)
    screen.blit(score_text, (10, 10))
    
    pygame.display.flip()
    
    clock.tick(FPS)
    
pygame.quit()

