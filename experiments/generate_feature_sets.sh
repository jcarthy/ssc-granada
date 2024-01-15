#! /bin/zsh

# Split into train test and validation sets
python bin/split-dataset.py \
    --dataset-path datasets/llaima

# This script generates the feature sets for the experiments.
python bin/fft-split-dataset.py \
    --n-splits 1

python bin/fft-split-dataset.py \
    --n-splits 3
