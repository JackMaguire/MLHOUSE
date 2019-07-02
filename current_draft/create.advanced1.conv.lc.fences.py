import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


#import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LocallyConnected2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
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
parser.add_argument( "--model", help="filename for output file", default="start.advanced1.conv.lc.fences", required=False )
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

num_output_dimensions = 8


input1 = Input(shape=(num_input_dimensions1,), name="in1", dtype="float32" )

#input2 = Input(shape=(num_input_dimensions2,), name="in2", dtype="float32" )
input2 = Input(shape=(36, 19, 14,), name="in2", dtype="float32" )

convA = Conv2D(         name="convA", filters=5, kernel_size=(1,1), padding='valid', input_shape=(36, 19, 14), data_format='channels_last', activation='relu', use_bias=True )( input2 )
lcB = LocallyConnected2D( name="lcB", filters=5, kernel_size=(3,3), padding='valid', input_shape=(36, 19,  5), data_format='channels_last', activation='relu', use_bias=True )( convA )
flatB = Flatten( name="flatB", data_format='channels_last' )( lcB )

merge = tensorflow.keras.layers.concatenate( [flatB, input1], name="merge_flatB_input2" )

denseC = Dense( name="denseC", units=num_neurons_in_layerC, activation='relu' )( merge )
denseD = Dense( name="denseD", units=num_neurons_in_layerD, activation='relu' )( denseC )
denseE = Dense( name="denseE", units=num_neurons_in_layerE, activation='relu' )( denseD )

output = Dense( name="output", units=num_output_dimensions, activation='sigmoid' )( denseE )

model = Model(inputs=[input1, input2], outputs=output )

metrics_to_output=[ 'accuracy' ]
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
model.summary()
