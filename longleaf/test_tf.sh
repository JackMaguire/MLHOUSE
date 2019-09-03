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

#SBATCH --job-name=keras
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0:01:00
#SBATCH --partition=interact
#SBATCH --output=run-%j.log

module restore keras_cpu
#module add cuda
#module restore keras_cpu_py2

module list

unset OMP_NUM_THREADS

# Set SIMG path
#SIMG_PATH=/nas/longleaf/apps/keras_py3/2.2.4/simg
#SIMG_PATH=/nas/longleaf/apps/keras/2.2.4/simg
SIMG_PATH=/nas/longleaf/apps/tensorflow_nogpu_py3/1.9.0/simg

# Set SIMG name
#SIMG_NAME=keras2.2.4_py3-tf-cuda9.0-ubuntu16.04.simg
#SIMG_NAME=keras2.2.4-tf1.9.0-cuda9.0-ubuntu16.04.simg
SIMG_NAME=tensorflow1.9.0-py3-nogpu-ubuntu18.04.simg

python --version

python3 --version

# CPU with Singularity
singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "python3 create.base2.lcn.sum.py"

