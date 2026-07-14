
import torch

from torch import nn
# Setting up hyperparameters here so that its easy to change
learning_rate = 1e-3
batch_size = 64
epochs = 100

import random

class GridWorld:
    def __init__(self):
        self.world = [
                        ["S", ".", ".", "."],
                        [".", "#", ".", "G"],
                        [".", ".", ".", "."]
                    ]
        self.position = [0,0]
        self.goal = [1,3]
    
    # 0 - up, 1 - down, 2 - left, 3 - right
    def move(self, movement):
        finished = False
        new_row = self.position[0]
        new_col = self.position[1]
        if movement == 0:
            new_row -= 1
        elif movement == 1:
            new_row += 1
        elif movement == 2:
            new_col -= 1
        elif movement == 3:
            new_col += 1

        if 0 <= new_row < len(self.world) and 0 <= new_col < len(self.world[0]):
            if self.world[new_row][new_col] != "#":
                self.position = [new_row, new_col]
        
        new_position = self.position
        if self.world[self.position[0]][self.position[1]] == "G":
            reward = 10
            finished = True
        else:
            reward = -1

        return new_position, reward, finished

    def reset(self):
        self.position = [0,0]

def random_agent():
    highest = random.randint(0,3)
    return highest

# The model class itself
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() # Getting the init of the parent

        # The steps the nn will follow to learn
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 16), # Converts the 784 elements each image has (each pixel of the image is an element) into 512 new outputs
            nn.LeakyReLU(), # Adds complexities
            nn.Linear(16,16), # Converts the 512 features into 512 new features
            nn.Tanh(),
            nn.Linear(16,4) # Eventually makes 10 outputs since there are 10 digits
        )
    
    # What to do upon getting an input
    def forward(self, x):
        q_values = self.linear_relu_stack(x) # Going through the steps
        return q_values # Returning the logits

def train_model(model, optimiser, loss_fn, action, new_state):
    model.train()
    q_values = model(state)
    target_q = q_values[action] + a * (reward + y + max(q_values[new_state]) - q_values[action])

    loss = loss_fn(q_values, target_q) # CrossEntropy automatically turns logits into probabilities so no need for Softmax

    # Backpropagation to calculate gradients and learn
    loss.backward() # Go through each operation
    optimiser.step() # Learn
    optimiser.zero_grad() # Remove gradients
    

# Actually running everything
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
state = torch.tensor([0.0, 0.0])


world = GridWorld()
done = False


a = 0.1
y = 0.9
count = 0
state = 0
while epochs > 0:
    
    state = world.position
    action = random_agent()
    new_state, reward, done = world.move(action)
    train_model(model, optimiser, loss_fn, action, new_state)
