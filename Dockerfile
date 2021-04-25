FROM tensorflow/tensorflow:nightly-gpu

#Essentials and pip3
RUN apt-get update -y && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
	apt-utils \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python-dev \
        rsync \
        unzip \
	python3-pip \
	git \
	vim \
	cmake \
	python-pil \
	python-lxml \
	nano \
	libsm6 \
	libxext6 \
	libxrender-dev \
	xvfb \
	python-opengl \
	ffmpeg \
	wget && apt-get update -y && apt-get upgrade -y
		#Insert additional apt-gets here

#Update pip and setuptools; install tensorflow2.0 and dependencies 
RUN pip3 install --upgrade pip setuptools
RUN pip3 install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
	pyyaml \
	tqdm \
	imageio
		#Insert additional pip3s here 

#Install specific version of gym for simglucose
RUN pip3 install -Iv "gym == 0.9.4"

RUN pip3 install --upgrade pip setuptools

#Create Workdirectory
RUN mkdir -p /homedirectory
WORKDIR /homedirectory


#Expose port 8888 (of container!) to allow communication to host machine. ATTENTION: exposing port 8888 of container does not automatically map to 8888 of host machine, use 'docker port <CONTAINER_ID>' to check
EXPOSE 8888
EXPOSE 6006

##########################################################################################################
##Shell code to run container 
##docker run -p 8888:8888 -it <CONTAINER_ID> 
##
##Jupyter Notebook 
##jupyter notebook --ip=0.0.0.0 --allow-root
