import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class DQN(nn.Module):
    
    def __init__(self, inputSize=11, hiddenSize=256, outputSize=3): 
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.fc3 = nn.Linear(hiddenSize, outputSize)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class DQNagent:
    def __init__(self, stateSize=11, actionSize=3, lr=0.001):  
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.995  
        self.learningRate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQN(stateSize, 256, actionSize).to(self.device)
        self.target_network = DQN(stateSize, 256, actionSize).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.batchSize = 32
        self.gamma = 0.95
        self.updateTargetFreq = 1000
        self.stepCount = 0
        
        self.updateTargetNetwork()
        
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))        
        
    def act(self, state, training=True):  
        if training and random.random() <= self.epsilon:
            return random.randrange(self.actionSize)
            
        stateTensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
        qValues = self.q_network(stateTensor)
        return np.argmax(qValues.cpu().data.numpy())
        
    def replay(self):  
        if len(self.memory) < self.batchSize:
            return None
            
        batch = random.sample(self.memory, self.batchSize)
        
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        nextStates = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)
        
        currentQvalues = self.q_network(states).gather(1, actions.unsqueeze(1))  
        
        nextQvalues = self.target_network(nextStates).max(1)[0].detach()
        targetQvalues = rewards + (self.gamma * nextQvalues * ~dones)
        
        loss = F.mse_loss(currentQvalues.squeeze(), targetQvalues)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilonMin:  
            self.epsilon *= self.epsilonDecay
            
        self.stepCount += 1
        if self.stepCount % self.updateTargetFreq == 0:
            self.updateTargetNetwork()
            
        return loss.item()
        
    def updateTargetNetwork(self): 
        """Update target network by copying weights from main network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def saveModel(self, filepath): 
        torch.save({
            'qNetworkStateDict': self.q_network.state_dict(),  
            'targetNetworkStateDict': self.target_network.state_dict(),
            'optimizerStateDict': self.optimizer.state_dict(),  
            'epsilon': self.epsilon,
            'stepCount': self.stepCount
        }, filepath)
        print(f"Model saved to {filepath}")
        
    def loadModel(self, filepath):  
        checkpoint = torch.load(filepath, map_location=self.device)  
        self.q_network.load_state_dict(checkpoint['qNetworkStateDict']) 
        self.target_network.load_state_dict(checkpoint['targetNetworkStateDict'])
        self.optimizer.load_state_dict(checkpoint['optimizerStateDict'])
        self.epsilon = checkpoint['epsilon']
        self.stepCount = checkpoint['stepCount']
        print(f"Model loaded from {filepath}")
