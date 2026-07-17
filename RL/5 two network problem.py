import torch

from torch import nn
# Setting up hyperparameters here so that its easy to change
learning_rate = 1e-3
batch_size = 64
epochs = 60

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

def rl_agent(q_values):
    highest = q_values.argmax().item()
    probability = random.randint(1,10)
    if probability == 10:
        highest = random.randint(0,3)
    return highest

# The model class itself
class OnlineNetwork(nn.Module):
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

def calculate_target(model, new_state, reward, done):
    with torch.no_grad():
        next_q_values = model(new_state)
    y = 0.9

    if done:
        target_q = reward
    else:
        target_q = reward + y * torch.max(next_q_values)
    
    target_q = torch.tensor(target_q, dtype=torch.float32)

    return target_q

def train_model(model, state, action, target_q, optimiser):

    model.train()
    q_values = model(state)

    loss = loss_fn(q_values[action], target_q) # CrossEntropy automatically turns logits into probabilities so no need for Softmax

    # Backpropagation to calculate gradients and learn
    loss.backward() # Go through each operation
    optimiser.step() # Learn
    optimiser.zero_grad() # Remove gradients
    
    

# Actually running everything
online_model = OnlineNetwork()
target_model = OnlineNetwork()

loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(online_model.parameters(), lr=learning_rate)
memory = []

world = GridWorld()
done = False

state = torch.tensor(world.position, dtype=torch.float32)
epoch_count = 0
target_change_count = 100
# q_values = train_model(model, optimiser, loss_fn, [0,0], 1, [1,0], 0)

while epochs > 0:
    
    state = torch.tensor(world.position, dtype=torch.float32)
    q_values = online_model(state)
    action = rl_agent(q_values)
    new_position, reward, done = world.move(action)
    new_state = torch.tensor(new_position, dtype=torch.float32)

    memory.append((state, action, reward, new_state, done))

    if target_change_count >= 100:
        target_model.load_state_dict(online_model.state_dict())
        target_change_count = 0

    target_q = calculate_target(target_model, new_state, reward, done)
    train_model(online_model, state, action, target_q, optimiser)

    if len(memory) > 10000:
        memory = memory[1:]

    epoch_count += 1
    target_change_count += 1

    sample = random.sample(memory, min(32, len(memory)))

    for i in sample:
        state, action, reward, new_state, done = i
        target_q = calculate_target(target_model, new_state, reward, done)
        train_model(online_model, state, action, target_q, optimiser)


    if done:
        epochs -= 1
        world.reset()
        if epochs % 20 == 0:
            print(f"Position: {new_position}, Up: {q_values[0]}, Down {q_values[1]}, Left {q_values[2]}, Right {q_values[3]},\n")
            print(f"Average number of steps needed: {epoch_count/20}")
            print("\n")
            epoch_count = 0
        done = False
    