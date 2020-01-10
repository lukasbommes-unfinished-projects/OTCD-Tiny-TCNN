FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

###############################################################################
#
#						 Basic Packages and Python 3.6
#
###############################################################################

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

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

###############################################################################
#
#						 MV-Extractor (+ OpenCV & FFMPEG)
#
###############################################################################

ENV HOME "/home"

# Download and build sfmt-videocap from source
RUN cd $HOME && \
  git clone -b "v1.2.0" https://sfmt-auto:Ow36ODbBoSSezciC@github.com/LukasBommes/sfmt-videocap.git video_cap && \
  cd video_cap && \
  chmod +x install.sh && \
  ./install.sh

# Set environment variables
ENV PATH "$PATH:$HOME/bin"
ENV PKG_CONFIG_PATH "$PKG_CONFIG_PATH:$HOME/ffmpeg_build/lib/pkgconfig"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH":$HOME/ffmpeg_build/lib

###############################################################################
#
#						 Coviar
#
###############################################################################

WORKDIR /workspace/pytorch-coviar
COPY pytorch-coviar .
WORKDIR /workspace/pytorch-coviar/data_loader
RUN chmod +x install.sh && \
  ./install.sh

###############################################################################
#
#						 Python Packages
#
###############################################################################

WORKDIR /workspace
RUN chmod -R a+w /workspace

# Install Python packages
COPY requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

CMD ["sh", "-c", "tail -f /dev/null"]
