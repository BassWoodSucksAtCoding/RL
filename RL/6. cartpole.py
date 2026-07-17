import torch
import gymnasium as gym

from torch import nn
# Setting up hyperparameters here so that its easy to change
learning_rate = 1e-3

epochs = 600 # How many episodes to go through

import random

# The agent itself with 90% exploitation, 10% exploration
def rl_agent(q_values):
    highest = q_values.argmax().item() # Finds the highest q value (i.e., the option with the highest value and picks that [Basically means it picks the best option])
    probability = random.randint(1,10)
    if probability == 10: # 10% of the time it picks a random action instead of the best one (Exploration)
        highest = random.randint(0,1)
    return highest

# The model class itself
class OnlineNetwork(nn.Module):
    def __init__(self):
        super().__init__() # Getting the init of the parent

        # The steps the nn will follow to learn
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 16), # Converts the 4 outputs from each step or action into 16 neurons
            nn.LeakyReLU(), # Adds complexities
            nn.Linear(16,16), # Converts the 16 features into 16 new features
            nn.Tanh(),
            nn.Linear(16,2) # Eventually makes 2 outputs which refer to the 2 actions the AI can take (left or right)
        )
    
    # What to do upon getting an input
    def forward(self, x):
        q_values = self.linear_relu_stack(x) # Going through the steps
        return q_values # Returning the q_values (How valuable each action is)

# Calculating the q_target
def calculate_target(model, new_state, reward, done):
    with torch.no_grad():
        next_q_values = model(new_state) # Getting the q_value of the next state
        # This is basically the neural network predicting how good each of the future actions you can take are
    y = 0.9 # Discount rate

    if done:
        target_q = reward # When the episode ends, the reward itself is the target
    else:
        target_q = reward + y * torch.max(next_q_values) 

        # Ignoring the math behind the target_q, what it actually does is find out how valuable the action you took was
        # So, for example, if you moved left from Room A to Room B and got 10 reward 
        # next_q_values will give you the q_values of each action you can take from room B
        # So, next_q_values might return Right = 7, Left, = 2
        # Assuming we take the best option for Room B, which is Right, we could get future rewards
        # So the bellman target uses the reward we got (10 points) and the reward we could get (7 for right) to calculate the overall value of moving left from Room A to Room B

        # y is the discount factor and it tells us how much we care about future rewards
        # Right now y is 90%, so, moving left from Room A to Room B will not only get the 10 points but 90% of that 7 points to go right

        # This "How valuable is an action" is the q_value

    return torch.as_tensor(target_q, dtype=torch.float32) # Returns the target q as a tensor

# Actually training the model
def train_model(model, state, action, target_q, optimiser):

    model.train()
    q_values = model(state)

    loss = loss_fn(q_values[action], target_q) 
    # q_values is what the model predicts how valuable each action will be
    # q_values[action] is the value of the action we just took
    # target_q gives the actual q_value of each action
    # So the loss calculates how far off the neural network was

    # Backpropagation to calculate gradients and learn
    loss.backward() # Go through each operation
    optimiser.step() # Learn
    optimiser.zero_grad() # Remove gradients
    
    

# Actually running everything
online_model = OnlineNetwork()
target_model = OnlineNetwork()

# We will have two models to help us
# The online model will be the one actually learning and making moves
# The target model is essentially a "frozen" version of the online model, it only calculates the target_q
# Every 100 episodes, the online model's weights are copied to the target model

# Basically, if we used one neural network, it would calculate the q_values of each of its moves, but now that it has learnt a bit
# when it goes to predict the target_q, the predicted target_q will change too
# Thus, the neural network would keep chasing a moving target

# By using a frozen target model, we ensure the predicted target_q remains the same so that the neural network can actually learn and get closer to the target_q

# For instance, let's say 2+3 = 5
# A student writes 2+3=4, so he is 1 off from the answer
# But then lets say the answer changes to 2+3=6
# The student will get confused
# ^^ Highly oversimplified explanation


# Just loading the saved weights
checkpoint = torch.load("cartpole_checkpoint.pth")

online_model.load_state_dict(checkpoint["model"])

# Initialising some things
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(online_model.parameters(), lr=learning_rate)
memory = []

optimiser.load_state_dict(checkpoint["optimizer"])

# Ensuring the target model begins with the same weights as the online model
target_model.load_state_dict(online_model.state_dict())

# This is the actual environment
env = gym.make("CartPole-v1", render_mode="human")

# Done tells us if the episode is over
done = False

# This tells us after how many episodes target should copy online's weights
target_change_count = 100

# Getting our current postion (state), info doesn't matter here
state, info = env.reset()

# Converting state into a tensor
state = torch.tensor(state, dtype=torch.float32)

# Train = True means training mode, Train = False means testing mode
train = True

# These messages are for outputting progress
# Step count counts the number of steps or moves the nn performs
step_count = 0

print("\n")
print(f"Statistics for every 20 episodes")
print(f"How many episodes are left: {epochs}")
print(f"Average episode length: {step_count / 20:.1f} steps")

if train: 
    online_model.train()
    while epochs > 0:
        
        # Predicting the q_values of each move
        q_values = online_model(state)

        # Taking an action
        action = rl_agent(q_values)

        # Getting values from the action
        new_state, reward, terminated, truncated, info = env.step(action)

        # New state is the next position (So if we moved from Room A to Room B, Room A would be first state, and Room B would be new state)
        # State is not just position, it contains all the info the nn needs 
        # Here the state contains Cart position, Cart Velocity, Pole Angle, Pole Angular velocity
        new_state = torch.tensor(new_state, dtype=torch.float32)
        done = terminated or truncated

        # Putting the current state and some info into a replay experience
        memory.append((state, action, reward, new_state, done))

        # Copying weights every 100 episodes
        if target_change_count >= 100:
            target_model.load_state_dict(online_model.state_dict())
            target_change_count = 0

        # Actually training with our values
        target_q = calculate_target(target_model, new_state, reward, done)
        train_model(online_model, state, action, target_q, optimiser)

        # If there are more than 10000 memories in the replay, remove the oldest one
        if len(memory) > 10000:
            memory = memory[1:]

        # Increasing our counts
        step_count += 1
        target_change_count += 1

        # Every step, we train on 32 random old experiences which we stored in memory
        # This has some advantages like breaking correlation
        sample = random.sample(memory, min(32, len(memory)))


        # Here we train on each of the 32 experiences 
        for experience in sample:
            replay_state, replay_action, replay_reward, replay_new_state, replay_done = experience

            target_q = calculate_target(
                target_model,
                replay_new_state,
                replay_reward,
                replay_done,
            )

            train_model(
                online_model,
                replay_state,
                replay_action,
                target_q,
                optimiser,
            )

        # When the episode is over
        if done:
            epochs -= 1
            state, info = env.reset()

            # Every 20 episodes we print statistics and save the model
            if epochs % 20 == 0:
                print("\n")
                print(f"Statistics for every 20 episodes")
                print(f"How many episodes are left: {epochs}")
                print(f"Average episode length: {step_count / 20:.1f} steps")
                step_count = 0

                torch.save({
                    "model": online_model.state_dict(),
                    "optimizer": optimiser.state_dict(),
                }, "cartpole_checkpoint.pth")

            # Obviously turn done off since we will turn on a new episode
            done = False
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = new_state

else:
    
    online_model.eval()

    # Basically in testing mode, the model is tested over 50 episodes (Untested so cannot confirm correct code)
    for i in range(50):
        q_values = online_model(state)
        action = q_values.argmax().item()

        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = torch.tensor(new_state, dtype=torch.float32)
        done = terminated or truncated

        if done:
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = new_state

# Close environment
env.close()


