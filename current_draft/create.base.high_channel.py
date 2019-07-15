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
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import UpSampling3D

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
parser.add_argument( "--model", help="filename for output file", default="start.base.high_channel", required=False )
args = parser.parse_args()

num_input_dimensions1 = 26
num_input_dimensions2 = 18494 - num_input_dimensions1

input1 = Input(shape=(1,1,num_input_dimensions1,), name="in1", dtype="float32" )
in1up = UpSampling3D( size=(36,19,1), data_format='channels_last' )( input1 )


#input2 = Input(shape=(36, 19, 27,), name="in2", dtype="float32" )
input2 = Input(shape=(num_input_dimensions2,), name="in2", dtype="float32" )

# Phase 1: in2 -> ABCDE
pre = Reshape( target_shape=(36, 19, 27,) )( input2 )#ALWAYS DO WIDTH, HEIGHT, CHANNELS
A = Conv2D( name="layerA", filters=20, kernel_size=(1,1), padding='valid', input_shape=pre.shape, data_format='channels_last', activation='relu', use_bias=True )( pre )
B = Conv2D( name="layerB", filters=20, kernel_size=(1,1), padding='valid', input_shape=A.shape, data_format='channels_last', activation='relu', use_bias=True )( A )
C = Conv2D( name="layerC", filters=20, kernel_size=(1,1), padding='valid', input_shape=B.shape, data_format='channels_last', activation='relu', use_bias=True )( B )
D = Conv2D( name="layerD", filters=20, kernel_size=(1,1), padding='valid', input_shape=C.shape, data_format='channels_last', activation='relu', use_bias=True )( C )
E = Conv2D( name="layerE", filters=15, kernel_size=(1,1), padding='valid', input_shape=D.shape, data_format='channels_last', activation='relu', use_bias=True )( D )
merge = tensorflow.keras.layers.concatenate( [E,in1up], name="merge", axis=-1 )

# Phase 2: FGHIJ
F = LocallyConnected2D( name="layerF", filters=15, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( merge )
G = LocallyConnected2D( name="layerG", filters=13, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( F )
H = LocallyConnected2D( name="layerH", filters=11, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( G )
I = LocallyConnected2D( name="layerI", filters=9, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( H )
J = LocallyConnected2D( name="layerJ", filters=7, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( I )
K = LocallyConnected2D( name="layerK", filters=7, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( J )
L = LocallyConnected2D( name="layerL", filters=7, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( K )
M = LocallyConnected2D( name="layerM", filters=7, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( L )
N = LocallyConnected2D( name="layerN", filters=7, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( M )
O = LocallyConnected2D( name="layerO", filters=5, kernel_size=(3,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( N )

# Phase 3: flatJ, merge, KLMN, output
flat = Flatten( name="flat", data_format='channels_last' )( O )
#print( flatJ.shape )
#exit( 1 )
 
dense1 = Dense( name="dense1", units=400, activation='relu' )( flat )
dense2 = Dense( name="dense2", units=300, activation='relu' )( dense1 )
dense3 = Dense( name="dense3", units=200, activation='relu' )( dense2 )
dense4 = Dense( name="dense4", units=100, activation='relu' )( dense3 )
output = Dense( name="output", units=1, activation='linear' )( dense4 )

model = Model(inputs=[input1, input2], outputs=output )

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
model.summary()
