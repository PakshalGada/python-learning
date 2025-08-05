import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import deque

from snakeEnvironment import snakeEnvironment
from neuralNetwork import DQNagent

class snakeTrainer:

    def __init__(self, episodes=1000, renderFreq=100):
        self.episodes = episodes
        self.renderFreq = renderFreq
        
        self.env = snakeEnvironment(width=20, height=20, render=True)
        self.agent = DQNagent(stateSize=11, actionSize=3, lr=0.001)
        
        self.scores = []
        self.avgScores = [] 
        self.losses = []
        self.highScore = 0
        
        os.makedirs('models', exist_ok=True)
        
    def train(self): 
        print("\nStarting Training...")
        startTime = time.time()
        
        recentScores = deque(maxlen=100)
        
        for episode in range(self.episodes):
            state = self.env.reset()
            totalReward = 0
            steps = 0
            
            if episode < 50 or episode % 50 == 0:
                self.env.renderGame = True
                renderThisEpisode = True
            else:
                self.env.renderGame = False
                renderThisEpisode = False
            
            while True:
                action = self.agent.act(state, training=True)
                nextState, reward, done, info = self.env.step(action)  
                self.agent.remember(state, action, reward, nextState, done)  
                
                if len(self.agent.memory) > self.agent.batchSize:
                    loss = self.agent.replay()
                    if loss is not None:
                        self.losses.append(loss)  
            
                state = nextState  
                totalReward += reward
                steps += 1
                
                if renderThisEpisode:
                    self.env.render()
                    
                if done:
                    break
                    
            if isinstance(info, dict):
                score = info.get('score', 0)
            elif hasattr(self.env, 'score'):
                score = self.env.score
            else:
                score = totalReward  
            self.scores.append(score) 
            recentScores.append(score)
            avgScore = np.mean(recentScores)
            self.avgScores.append(avgScore) 
                
            if score > self.highScore:
                self.highScore = score
                self.agent.saveModel('models/best_snake_model.pth')
                print(f"🏆 New high score: {score}!")
                    
            if episode % 50 == 0 or episode == self.episodes - 1:
                elapsedTime = time.time() - startTime
                print(f"Episode {episode:4d} | "
                          f"Score: {score:3d} | "
                          f"Avg Score: {avgScore:6.2f} | "
                          f"High Score: {self.highScore:3d} | "
                          f"Epsilon: {self.agent.epsilon:.3f} | "
                          f"Time: {elapsedTime:.1f}s")    
                
            if episode % 500 == 0 and episode > 0:
                self.agent.saveModel(f'models/snake_model_episode_{episode}.pth')  

        self.agent.saveModel('models/final_snake_model.pth') 

        totalTime = time.time() - startTime 
        print(f"\n✅ Training completed in {totalTime:.1f} seconds!")
        print(f"🏆 Highest score achieved: {self.highScore}")  
        print(f"📈 Final average score: {self.avgScores[-1]:.2f}")  
        
   
        
    def testAI(self, modelPath='models/best_snake_model.pth', games=5):
        print(f"Testing AI with model: {modelPath}")
        self.agent.loadModel(modelPath)
        self.agent.epsilon = 0 

        testScores = []

        for game in range(games):
            print(f"\nGame {game + 1}/{games}")
            state = self.env.reset()
            self.env.renderGame = True
            steps = 0

            while True:
                action = self.agent.act(state, training=False)
                state, reward, done, info = self.env.step(action)
                self.env.render()
                steps += 1

                time.sleep(0.1)

                if done:
                    if isinstance(info, dict):
                        score = info.get('score', 0)
                    elif hasattr(self.env, 'score'):
                        score = self.env.score
                    else:
                        score = steps 
                    
                    testScores.append(score)
                    print(f"Score: {score}, Steps: {steps}")
                    break

        avgTestScore = np.mean(testScores)
        print(f"\n🎯 Test Results:")
        print(f"Average Score: {avgTestScore:.1f}")
        print(f"Best Score: {max(testScores)}")
        print(f"All Scores: {testScores}")

def main():
    print("🐍 Welcome to Snake AI Training!")
    print("This will train an AI to play Snake using Deep Q-Learning")

    EPISODES = 2000  
    RENDER_FREQ = 200  

    trainer = snakeTrainer(episodes=EPISODES, renderFreq=RENDER_FREQ)

    print("\nWhat would you like to do?")
    print("1. Train new AI from scratch")
    print("2. Continue training existing model")

    choice = input("Enter your choice: ").strip()    
    
    if choice == "1":
        print("🚀 Training new AI from scratch...")
        trainer.train()

    elif choice == "2":
        modelPath = input("Enter model path (or press Enter for 'models/final_snake_model.pth'): ").strip()
        if not modelPath:
            modelPath = 'models/final_snake_model.pth'

        try:
            trainer.agent.loadModel(modelPath)
            print(f"✅ Loaded model from {modelPath}")
            print("🚀 Continuing training...")
            trainer.train()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🚀 Training new model instead...")
            trainer.train()
    
    elif choice == "3":
        modelPath = input("Enter model path (or press Enter for 'models/best_snake_model.pth'): ").strip()
        if not modelPath:
            modelPath = 'models/best_snake_model.pth'

        try:
            numGames = int(input("How many test games? (default 5): ") or "5")
            trainer.testAI(modelPath, games=numGames)
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            
    elif choice == "4":
        print("🚀 Running quick demo...")
        trainer.episodes = 50
        trainer.renderFreq = 10
        trainer.train()

if __name__ == "__main__":
    main()
