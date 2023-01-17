from pyrsistent import optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        
    
    
class QTrain:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.g = gamma
        self.model = model
        self.op = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() #Loss function which is (Qnew - Q)²
        
    def train_step(self, state, action, reward, nextState, over):
        state = torch.tensor(state, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        #(n, x)
        
        if len(state.shape) == 1:
            #(1, x)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            nextState = torch.unsqueeze(nextState, 0)
            reward = torch.unsqueeze(reward, 0)
            over = (over, )
                
        pred = self.model(state) # 1: predicted Q values with current state
        
        # 2: Q new = r + y * max(next predicted Q value)
        target = pred.clone()
        for idx in range(len(over)):
            Qnew = reward[idx]
            if not over[idx]:
                Qnew = reward[idx] + self.g * torch.max(self.model(nextState[idx]))
                
            target[idx][torch.argmax(action[idx]).item()] = Qnew

        self.op.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.op.step()
                
        
        
            
        
    
        
        
        