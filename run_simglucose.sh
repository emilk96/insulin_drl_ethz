#!/bin/bash

#Install specific version of simglucose
docker exec -it sim sh -c "apt update -y && apt upgrade -y"
docker exec -it sim sh -c "cd simglucose && python3 setup.py install && cd .."
docker exec -it sim sh -c "pip3 install --upgrade gym"
docker exec -it sim sh -c "cd T1DEK_gym && pip install -e . && cd .."
docker exec -it sim sh -c "pip3 install pybullet"
docker exec -it sim sh -c "pip3 install tf-agents[reverb]"

#Open jupyter notebook
#open http://127.0.0.1:8888
#docker exec -it sim sh -c "jupyter notebook --ip=0.0.0.0 --no-browser --allow-root"
#git pull && git checkout master