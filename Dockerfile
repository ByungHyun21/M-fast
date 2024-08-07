FROM nvcr.io/nvidia/pytorch:24.07-py3
# important pakages
# python 3.10.12
# pytorch 2.4.0a0+3bcc3cddb5.nv24.07
# torchvision 0.19.0a0
# opencv 4.7.0
# tensorboard 2.9.0
# matplotlib 3.9.1
# numba 0.59.1
# numpy 1.24.4
# onnx 1.16.0
# pandas 2.2.1
# scikit-learn 1.5.1
# scipy 1.13.1

RUN apt -y update
RUN apt -y upgrade

# additional packages
RUN python -m pip install --upgrade pip==24.1.2
RUN pip install termcolor==2.4.0
RUN pip install colorama==0.4.6
RUN pip install ptflops==0.7.3


WORKDIR /M-fast