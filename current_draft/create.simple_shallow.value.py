import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


#import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks
import tensorflow.keras
import numpy

import sys
import h5py

import argparse
import random
import time
import subprocess

########
# INIT #
########

numpy.random.seed( 0 )

parser = argparse.ArgumentParser()
parser.add_argument( "--model", help="filename for output file", default="start.simple_shallow.value", required=False )
args = parser.parse_args()


num_input_dimensions = 9600
num_neurons_in_layer1 = 1500
num_neurons_in_layer2 = 1000
num_output_dimensions = 1
model = Sequential()

model.add( Dense( num_neurons_in_layer1, input_dim=num_input_dimensions, activation='relu' ) )
model.add( Dense( num_neurons_in_layer2, activation='relu') )
model.add( Dense( num_output_dimensions, activation='sigmoid') )

metrics_to_output=[ 'accuracy' ]
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
