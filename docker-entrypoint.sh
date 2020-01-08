#!/bin/sh
set -e

cd /workspace/lib/model/roi_align && chmod +x install.sh && ./install.sh
cd /workspace/lib/model/nms && chmod +x make.sh && ./make.sh

exec "$@"
