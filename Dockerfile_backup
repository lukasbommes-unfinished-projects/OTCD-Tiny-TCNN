FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && \
  apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && apt-get install -y \
    pkg-config \
    python3.6-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

RUN ln -s $(which python3.6) /usr/local/bin/python

RUN pip3 install numpy pkgconfig


WORKDIR /workspace
RUN chmod -R a+w /workspace

# Install coviar
WORKDIR /workspace/pytorch-coviar
COPY pytorch-coviar .
WORKDIR /workspace
COPY install.sh .
RUN chmod +x install.sh && \
  ./install.sh

# Set environment variables
ENV PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/ffmpeg_build/lib/pkgconfig"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH":/ffmpeg_build/lib

WORKDIR /workspace/pytorch-coviar/data_loader
RUN chmod +x install.sh && \
  ./install.sh

WORKDIR /workspace

# Install Python packages
COPY requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

CMD ["sh", "-c", "tail -f /dev/null"]
