#!/bin/bash

## This is an example of an sbatch script to run a keras script
## using Singularity to run the keras image.
##
## Set the DATA_PATH to the directory you want the job to run in.
##
## On the singularity command line, replace ./test.py with your program
##
## Make a copy of this script and change reserved resources and command to run as needed for your job.
##
## Submit this script using sbatch.

#SBATCH --job-name=tensorflow
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --time=7-0:00:00
#SBATCH --partition=gpu
#SBATCH --output=run-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

module restore keras_gpu

module list

unset OMP_NUM_THREADS

# Set SIMG path
#SIMG_PATH=/nas/longleaf/apps/tensorflow_nogpu_py3/1.9.0/simg
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/1.12.0/simg

# Set SIMG name
#SIMG_NAME=tensorflow1.9.0-py3-nogpu-ubuntu18.04.simg
SIMG_NAME=tensorflow1.12.0-py3-cuda9.0-ubuntu16.04.simg

# CPU with Singularity
singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "while true; do bash run.sh; done"

