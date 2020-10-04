#!/bin/bash

conda env create -f environment.yaml
eval "$(conda shell.bash hook)"
conda activate myna

git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install . && cd ..
rm -rf ctcdecode
