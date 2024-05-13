#!/usr/bin/bash 


model='ppo'
logDir="results/$model/v3"
numSteps=5_00_000

./main.py --model $model --log-dir $logDir --training-steps $numSteps