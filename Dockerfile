FROM nvcr.io/nvidia/pytorch:23.01-py3

RUN apt -y update
RUN apt -y upgrade

RUN pip install opencv-python
RUN pip install wandb
RUN pip install tqdm
RUN pip install pyyaml

RUN echo -e "machine api.wandb.ai\n  login user\n  password 336a1139e7b5522aabbd0eff612d30b3b4c33366" >> /root/.netrc

WORKDIR /home/od_ssd_torch
