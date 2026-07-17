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
        if self.world[position[0]][position[1]] == "G":
            reward = 10
            finished = True
        else:
            reward = -1

        return new_position, reward, finished

    def reset(self):
        self.position = [0,0]

def random_agent(position, q_table):
    highest = q_table[position].index(max(q_table[position]))
    probability = random.randint(1,10)
    if probability == 10:
        highest = random.randint(0,3)
    return highest
       
world = GridWorld()
done = False

q_table = []
for row in range(12):
    q_table.append([])
    for column in range(4):
        q_table[row].append(0)

print(q_table)

a = 0.1
y = 0.9
epochs = 100
count = 0
state = 0
while epochs > 0:
    
    position = world.position
    
    state = position[0] * len(world.world[0]) + position[1]
    
    action = random_agent(state, q_table)
    new_position, reward, done = world.move(action)
    new_state = new_position[0] * len(world.world[0]) + new_position[1]
    q_table[state][action] = q_table[state][action] + a * (reward + y * max(q_table[new_state]) - q_table[state][action])
    count += 1
    if done:
        epochs -= 1
        world.reset()
        if epochs % 20 == 0:
            for index, row in enumerate(q_table):
                print(f"Position: {index}, Left: {row[0]}, Right {row[1]}\n")
            print(f"Average number of steps needed: {count/20}")
            print("\n")
            count = 0
        done = False
