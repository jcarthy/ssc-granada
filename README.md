# ssc-granada

This repo contains code used for a short course on machine learning and exploratory data analysis related to geophysical datasets.

## Python Requirements

The following packages must be installed:

- scikit-learn (v1.4)
- umap-learn
- jupyter
- pandas
- seaborn
- matplotlib

It is advised that students create a conda environment for this course and install the requisite packages into this conda environment. The following command will create a conda environment and install the requisite packages to this environment.

```bash
conda create -n ssc-granada -c conda-forge scikit-learn=1.4 umap-learn jupyter pandas seaborn matplotlib
```


Alternatively pip may be used (not recommended)

## Repository Structure

### bin

The bin directory contains code for different python scripts. The scripts are as follows:

- **split-dataset.py** This script will split a dataset into training, validation and test sections.
- **fft-split-dataset.py** This script will convert each section of the dataset into the FFT. Additionally it is able to split each sample within the dataset in the time domain and obtain the FFT of each split. This is similar to creating a spectrogam of a fixed number of windows.

### experiments

This contains bash scripts with hyperparameters for running scripts within the bin directory

### **datasets**

This directory is for different datasets. There is an empty folder, llaima where the data.npy and labels.npy files from Google Drive should be placed.

### **scripts**

Scripts for different feature extraction routines

### jupyter

Notebooks with different machine learning exercises can be found here.
