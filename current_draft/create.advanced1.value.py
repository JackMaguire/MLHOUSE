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
parser.add_argument( "--model", help="filename for output file", default="start.advanced1.value", required=False )
args = parser.parse_args()


num_input_dimensions1 = 24
num_input_dimensions2 = 9600 - num_input_dimensions1

# IN1 -> A -> B
#                -> C -> D -> E -> out
#           IN2

#num_neurons_in_layerA = 1500
#num_neurons_in_layerB = 1500

num_neurons_in_layerC = 500
num_neurons_in_layerD = 500
num_neurons_in_layerE = 500

num_output_dimensions = 1


input1 = Input(shape=(num_input_dimensions1,), name="in1", dtype="float32" )
input2 = Input(shape=(num_input_dimensions2,), name="in2", dtype="float32" )

convA = Conv2D(         name="convA", filters=5, kernel_size=(1,1), padding='valid', input_shape=(36, 19, 14), data_format='channels_last', activation='relu', use_bias=True )( input2 )
lcB = LocallyConnected2D( name="lcB", filters=5, kernel_size=(3,3), padding='valid', input_shape=(36, 19,  5), data_format='channels_last', activation='relu', use_bias=True )( convA )

merge = keras.layers.concatenate( [lcB, input1], name="merge_lcB_input2" )

denseC = Dense( name="denseC", units=num_neurons_in_layerC, activation='relu' )( merge )
denseD = Dense( name="denseD", units=num_neurons_in_layerD, activation='relu' )( denseC )
denseE = Dense( name="denseE", units=num_neurons_in_layerE, activation='relu' )( denseD )

output = Dense( name="output", units=1, activation='linear' )

model = Model(inputs=[input1, input2], outputs=[output])

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
