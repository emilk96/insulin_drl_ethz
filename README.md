# Tensorflow 2.0 GPU Docker

Simulator environment

## Build instructions
* To build the image use ``build.sh``
* Move necessary files to tf20_transfer directory, it will be mounted to the container. 
* Change the absolute path of tf2_transfer in run.sh to make it possible to mount. This can only be done by putting in the absolute path of directory in host machine after the -v command.
* To run the container use ``run.sh``
* To open jupyter notebook use ``j.sh`` (from outside container) or ``bash ji.sh`` (from inside container, the script is in the /homedirectory folder which is default folder)


