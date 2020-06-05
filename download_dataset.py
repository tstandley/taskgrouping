import os
import sys
import random
import multiprocessing

all_files = "all_links_taskonomydata.txt" # sys.argv[1]
tasknames = "dnkts" #sys.argv[2]

#ROOT = "/Users/arvind/Downloads"
ROOT = "/home/naagnes/taskonomy_full_dataset"
# hard coded to have only character per task...
acr_to_task = { "d": "depth_zbuffer", 
                "n": "normal", 
                "k": "keypoints2d", 
                "t": "edge_texture", 
                "s": "segment_semantic"}

def run_shell_cmd(cmd, verbose=False):
    if verbose:
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
# tasks = []
for acr in tasknames:
    tasks += [acr_to_task[acr]]

names = []
with open(all_files, "r") as f:
    all_files = []
    for i in f.readlines():
        name = i.rstrip("\n").split("/")[-1].split("_")[0]
        if name not in names: # to make sure we take the same subset at all times.
            names.append(name)

names.sort(reverse=True)

models = names[0:5] # Reached index 30 in ascending order
print("Downloading models:", models, end="\n\n")

def download_models(name):
    for task in tasks:
        os.makedirs(os.path.join(ROOT,task,name),exist_ok=True)
        url = "http://downloads.cs.stanford.edu/downloads/taskonomy_data/{0}/{1}_{0}.tar".format(task,name)
        run_shell_cmd("wget " + url + " -N -q --show-progress")
        run_shell_cmd("tar -C {} -xf {}".format(os.path.join(ROOT,task,name), os.path.join("{}_{}.tar".format(name,task))))
        # run_shell_cmd("mv {}/*.png {}".format(os.path.join(ROOT,task,name,task), os.path.join(ROOT,task,name)))
        run_shell_cmd("find {}".format(os.path.join(ROOT,task,name,task)) + " -name '*.png' -exec mv -t {}".format(os.path.join(ROOT,task,name) + " {} +"))
        run_shell_cmd("rmdir {}".format(os.path.join(ROOT,task,name,task)))
        if task == "segment_semantic":
            cwd = os.getcwd()
            os.chdir(os.path.join(ROOT,task,name))
            run_shell_cmd("find . -type f -name '*segmentsemantic.png' | while read FILE ; do newfile=\"$(echo ${FILE} |sed -e 's/segmentsemantic/segment\_semantic/')\" ; mv \"${FILE}\" \"${newfile}\"; done")
            os.chdir(cwd)    
        run_shell_cmd("rm {}".format(os.path.join("{}_{}.tar".format(name,task))))
    print("Done downloading " + name + ".")


cpus = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cpus)
pool.map(download_models, models)

"""
for name in buildings:
    for task in tasks:
        os.makedirs(os.path.join(root,task,name),exist_ok=True)
        url = "http://downloads.cs.stanford.edu/downloads/taskonomy_data/{0}/{1}_{0}.tar".format(task,name)
        run_shell_cmd("wget " + url + " -q --show-progress")
        run_shell_cmd("tar -C {} -xf {}".format(os.path.join(root,task,name), os.path.join("{}_{}.tar".format(name,task))))
        run_shell_cmd("rm {}".format(os.path.join("{}_{}.tar".format(name,task))))


def download_tasks(task):
    for name in buildings:
        print("TEST1: ", os.path.join(ROOT, task, name))
        print("TEST2: ", os.path.join("{}_{}.tar".format(name, task)))
        os.makedirs(os.path.join(ROOT,task,name),exist_ok=True)
        url = "http://downloads.cs.stanford.edu/downloads/taskonomy_data/{0}/{1}_{0}.tar".format(task,name)
        run_shell_cmd("wget " + url + " -q --show-progress")
        run_shell_cmd("tar -C {} -xf {}".format(os.path.join(ROOT,task,name), os.path.join("{}_{}.tar".format(name,task))))
        run_shell_cmd("rm {}".format(os.path.join("{}_{}.tar".format(name,task))))
        
        print("MOVING folders, from {} to {}".format(os.path.join(ROOT,task,name,task), os.path.join(ROOT,task,name)))
        run_shell_cmd("mv {}/*.png {}".format(os.path.join(ROOT,task,name,task), os.path.join(ROOT,task,name)))
        run_shell_cmd("rmdir {}".format(os.path.join(ROOT,task,name,task)))

cpus = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cpus)
pool.map(download_tasks, tasks)
"""
