import random

class LineWorld:
    def __init__(self):
        self.position = 0
        self.goal = 4

    def move(self, movement):
        finished = False
        reward = 0
        if movement == 1 and self.position != self.goal:
            self.position += 1
        elif movement == 0 and self.position != 0:
            self.position -= 1
        
        if self.position == 3:
            finished = True
            reward = -10
        
        new_position = self.position
        if new_position == self.goal:
            reward = 10
            finished = True

        return new_position, reward, finished
    
    def reset(self):
        self.position = 0

def random_agent(position, q_table):
    highest = q_table[position].index(max(q_table[position]))
    probability = random.randint(1,10)
    if probability == 10:
        highest = random.randint(0,1)
    return highest
       
world = LineWorld()
done = False

q_table = []
for row in range(world.goal + 1):
    q_table.append([])
    for column in range(2):
        q_table[row].append(0)
    

a = 0.1
y = 0.9
epochs = 100
count = 0
while epochs > 0:
    action = random_agent(world.position, q_table)
    position = world.position
    new_position, reward, done = world.move(action)
    q_table[position][action] = q_table[position][action] + a * (reward + y * max(q_table[new_position]) - q_table[position][action])
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