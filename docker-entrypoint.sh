#!/bin/sh
set -e

conda install -y av -c conda-forge  # needed to fix ffmpeg error during torchvision build
export FORCE_CUDA=1
cd /opt/pytorch
rm -rf vision
git config --global user.email "Lukas.Bommes@gmx.de"
git config --global user.name "LukasBommes"
git clone https://github.com/LukasBommes/vision.git
cd vision
pip install -v .
cd /workspace

# temporary workaround for torchvision issue #1712 (incompatibility with pillow 7.0.0)
pip install 'pillow<7'

# install coviar
cd /workspace && chmod +x install.sh && ./install.sh

exec "$@"
