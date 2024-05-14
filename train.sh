#!/usr/bin/bash 


model='ppo'
logDir="results/$model/v29"
numSteps=50_00_00
#25000
PYTHON="/home/airlab/anaconda3/envs/RL-trainagle-formation/bin/python"

$PYTHON main.py --model $model --log-dir $logDir --training-steps $numSteps