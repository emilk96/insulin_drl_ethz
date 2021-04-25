# ETHZ/pdz DRL system for T1D blood glucose control

## Quick start guide
* The system is optimized for a GPU (NVIDIA) enabled machine. 
* To build the image use ``build.sh``
* Move necessary files to sim_transfer directory, it will be mounted to the container. 
* Change the absolute path of sim_transfer in run.sh to make it possible to mount. This can only be done by putting in the absolute path of directory in host machine after the -v command.
* To run the container use ``run.sh``
* To run simglucose (T1D UVA/Padova python implementation) use ``run_simglucose.sh`` from a seperate terminal, this installs simglucose within the container. It is not included during container build to facilitate easy changes to the simulator.  
* To run soft actor critic training script (from inside the container) ``python3 sac/sac.py``, runtime approx. 3hrs on NVDIA Titan X. 
* Run inference in a validation environment use ``python3 sac/sac_inference.py``.

