FROM nvcr.io/nvidia/pytorch:23.01-py3

RUN apt -y update
RUN apt -y upgrade

# RUN apt-get install -y libgl1-mesa-glx

RUN pip install opencv-python
RUN pip install wandb
RUN pip install tqdm
RUN pip install pyyaml
RUN pip install termcolor colorama

WORKDIR /M-fast
