import os
import sys
import random
import copy

N = int(sys.argv[1])

task_folder = os.path.join(os.getcwd(), "../taskonomy_full_dataset/rgb/")

names = []
for building in os.listdir(task_folder):
    if os.path.isdir(task_folder + building):
        names.append([building, len(os.listdir(task_folder + building))])

names.sort(key=lambda x: x[1])


# Create N dataset
tot = 0
num = 0
num_buildings = len(names)
train_dataset = set()
while tot < N and num < num_buildings:
    possible_buildings = []
    for i in range(num_buildings):    
        if names[i][0] not in train_dataset and tot + names[i][1] < N:
            possible_buildings.append(i)
    if len(possible_buildings) == 0:
        break
    new = random.choice(possible_buildings)
    train_dataset.add(names[new][0])
    tot += names[new][1]
    num += 1

tot_n = tot
print("Created N training dataset with {} images (from {} buildings).".format(tot, num))

# Create now 5N dataset
train5_dataset = copy.copy(train_dataset)
# print(train_dataset)

while tot < 5*N and num < num_buildings:
    possible_buildings = []
    for i in range(num_buildings):    
        if names[i][0] not in train5_dataset and tot + names[i][1] < 5*N:
            possible_buildings.append(i)
    if len(possible_buildings) == 0:
        break
    new = random.choice(possible_buildings)
    train5_dataset.add(names[new][0])
    tot += names[new][1]
    num += 1

print("Created 5N training dataset with {} images (from {} buildings).".format(tot, num))
print("N dataset has {} labels in total, and 5N dataset has {} labels".format(5*tot_n, tot))
# print(train5_dataset)

# Write now these to file
with open("trainN_models.txt", "w") as f:
    f.write("\n".join(train_dataset))


with open("train5N_models.txt", "w") as f:
    f.write("\n".join(train5_dataset))
