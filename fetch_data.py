import os
import sys
import random


all_files = sys.argv[1]
tasknames = sys.argv[2]
train_size,val_size,test_size = 50,6,6


#ROOT = "/Users/arvind/Downloads"
ROOT = "/home/arvindsk/taskonomy_alpha"
# hard coded to have only character per task...
acr_to_task = { "d": "depth_zbuffer", 
				"n": "normal", 
				"k": "keypoints2d", 
				"t": "edge_texture", 
				"s": "segment_semantic"}

def run_shell_cmd(cmd):
    print(cmd)
    rval = os.system(cmd)
    if rval != 0:
        signal = rval & 0xFF
        exit_code = rval >> 8
        if signal != 0:
            sys.stderr.write("\nCommand %s exits with signal %d\n\n" % (cmd, signal))
            sys.exit(signal)
        sys.stderr.write("\nCommand %s failed with return code %d\n\n" % (cmd, exit_code))
        sys.exit(exit_code)

tasks = ["rgb"]
for acr in tasknames:
	tasks += [acr_to_task[acr]]

names = []
with open(sys.argv[1],"r") as f:
	all_files = []
	for i in f.readlines():
		name = i.rstrip("\n").split("/")[-1].split("_")[0]
		if name not in names: # to make sure we take the same subset at all times.
			names.append(name)

random.seed(0xAB1)
print(len(names))
train_names = random.sample(names,train_size)
names = list(set(names) - set(train_names))
val_names = random.sample(names,val_size)
names = list(set(names) - set(val_names))
test_names = random.sample(names,test_size)
names = list(set(names) - set(test_names))

print(len(train_names),len(train_names),len(val_names))


for blah,blah2 in zip(["train","val", "test"],[train_names,val_names,test_names]):
	with open(blah+"_names.txt","w") as f:
		f.write("\n".join(blah2))

for name in train_names + val_names + test_names :
	for task in tasks:
		os.makedirs(os.path.join(ROOT,task,name),exist_ok=True)
		url = "http://downloads.cs.stanford.edu/downloads/taskonomy_data/{0}/{1}_{0}.tar".format(task,name)
		run_shell_cmd("wget " + url)
		#print("tar -C {} -xvf {}".format(os.path.join(ROOT,task,name), os.path.join(ROOT,"{}_{}.tar".format(name,task))))
		run_shell_cmd("tar -C {} -xvf {}".format(os.path.join(ROOT,task,name), os.path.join("{}_{}.tar".format(name,task))))
		run_shell_cmd("rm {}".format(os.path.join("{}_{}.tar".format(name,task))))
