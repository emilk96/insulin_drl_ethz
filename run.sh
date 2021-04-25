#!/bin/bash

docker run --gpus device=1 -it --rm\
	-p 8888:8888 \
	--name sim \
 	-v /home/krauch/diabetes/simulator_env/sim_transfer:/homedirectory: \
	sim:latest

#To mount directory from hostmachine, an absolte path must be used. Change this script (-v /home/deeplearning...) to run container on different host machine.


