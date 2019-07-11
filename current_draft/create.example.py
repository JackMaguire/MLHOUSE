import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


#import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
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
parser.add_argument( "--model", help="filename for output file", default="start.example", required=False )
args = parser.parse_args()

num_input_dimensions1 = 26
num_input_dimensions2 = 18494 - num_input_dimensions1

input1 = Input(shape=(num_input_dimensions1,), name="in1", dtype="float32" )

#input2 = Input(shape=(36, 19, 27,), name="in2", dtype="float32" )
input2 = Input(shape=(18468,), name="in2", dtype="float32" )

pre = Reshape( target_shape=(36, 19, 27,) )( input2 )#ALWAYS DO WIDTH, HEIGHT, CHANNELS
l1 = LocallyConnected2D( name="layer1", filters=30, kernel_size=(3,2), padding='valid', input_shape=(36, 19, 27), data_format='channels_last', activation='relu', use_bias=True )( pre )
l2 = LocallyConnected2D( name="layer2", filters=25, kernel_size=(3,2), padding='valid', input_shape=(34, 18, 30), data_format='channels_last', activation='relu', use_bias=True )( l1 )
l3 = LocallyConnected2D( name="layer3", filters=20, kernel_size=(3,2), padding='valid', input_shape=(32, 17, 25), data_format='channels_last', activation='relu', use_bias=True )( l2 )
flat = Flatten( name="flat", data_format='channels_last' )( l3 )
merge = tensorflow.keras.layers.concatenate( [flat, input1], name="merge_flatJ_input2" )
l4 = Dense( name="dense4", units=500, activation='relu' )( merge )
l5 = Dense( name="dense5", units=500, activation='relu' )( l4 )
output = Dense( name="output", units=1, activation='linear' )( l5 )

model = Model(inputs=[input1, input2], outputs=output )

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
model.summary()
