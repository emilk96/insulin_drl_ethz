#!/bin/bash

docker run -it --rm\
	--privileged=true \
    -e DISPLAY=$DISPLAY \
	-p 8888:8888 \
	-p 6006:6006 \
 	--name sim \
 	-v/Users/emilkrauch/Desktop/GDrive/ST/Simulator/simulator_env/sim_transfer:/homedirectory: \
	sim:latest

#To mount directory from hostmachine, an absolte path must be used. Change this script (-v /home/deeplearning...) to run container on different host machine.


