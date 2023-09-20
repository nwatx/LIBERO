#!/bin/bash

if ! command -v python3.8 &> /dev/null; then
    echo "Python 3.8 is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3.8
fi

pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
python3 libero/lifelong/main.py benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=base