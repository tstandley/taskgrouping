#!/bin/bash

# Change settings here
data_dir="taskonomy-sample-model-1/"
model_name="xception_taskonomy_new"
tasks_to_train_on="sdnkt"
model_dir="test_run/"
number_of_workers="1"
experiment_name="dry_run"

# Compile command string
command="python taskgrouping/train_taskonomy.py "
command+="--data_dir $data_dir "
command+="--arch $model_name "
command+="--tasks $tasks_to_train_on "
command+="--model_dir $model_dir "
command+="--workers $number_of_workers "
command+="--experiment_name $experiment_name "

# Run command string
echo "Running command: $command"
eval $command
