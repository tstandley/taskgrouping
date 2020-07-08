# Code for Which Tasks to Train Together in Multi-Task Learning
Trevor Standley, Amir R. Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, Silvio Savarese
ICML 2020
http://taskgrouping.stanford.edu/

1. Install pytorch,torchvision
2. Install apex
```
conda install -c conda-forge nvidia-apex
```
3. (optional) install data loading speedups:
```
conda install -c thomasbrandon -c defaults -c conda-forge pillow-accel-avx2
conda install -c conda-forge libjpeg-turbo
```
4. Get training data
https://github.com/StanfordVL/taskonomy/tree/master/data
The data must be formatted in 

  root/task or rgb/building/point_x_view_x.png

order.

usage example
```
python3 train_taskonomy.py -d=/taskonomy_data/ -a=xception_taskonomy_new -j 4 -b 96 -lr=.1 --fp16 -sbn --tasks=sdnerac -r
```
