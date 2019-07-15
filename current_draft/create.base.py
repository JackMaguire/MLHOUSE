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
parser.add_argument( "--model", help="filename for output file", default="start.base", required=False )
args = parser.parse_args()

num_input_dimensions1 = 26
num_input_dimensions2 = 18494 - num_input_dimensions1

input1 = Input(shape=(num_input_dimensions1,), name="in1", dtype="float32" )
in1_A = Dense( name="in1_A", units=50, activation='relu' )( input1 )
in1_B = Dense( name="in1_B", units=50, activation='relu' )( in1_A )
in1_C = Dense( name="in1_C", units=50, activation='relu' )( in1_B )
in1_D = Dense( name="in1_D", units=50, activation='relu' )( in1_C )


#input2 = Input(shape=(36, 19, 27,), name="in2", dtype="float32" )
input2 = Input(shape=(18468,), name="in2", dtype="float32" )

# Phase 1: in2 -> ABCDE
pre = Reshape( target_shape=(36, 19, 27,) )( input2 )#ALWAYS DO WIDTH, HEIGHT, CHANNELS
A = Conv2D( name="layerA", filters=30, kernel_size=(1,1), padding='valid', input_shape=pre.shape, data_format='channels_last', activation='relu', use_bias=True )( pre )
B = Conv2D( name="layerB", filters=25, kernel_size=(1,1), padding='valid', input_shape=A.shape, data_format='channels_last', activation='relu', use_bias=True )( A )
C = Conv2D( name="layerC", filters=20, kernel_size=(1,1), padding='valid', input_shape=B.shape, data_format='channels_last', activation='relu', use_bias=True )( B )
D = Conv2D( name="layerD", filters=20, kernel_size=(1,1), padding='valid', input_shape=C.shape, data_format='channels_last', activation='relu', use_bias=True )( C )
E = Conv2D( name="layerE", filters=15, kernel_size=(1,1), padding='valid', input_shape=D.shape, data_format='channels_last', activation='relu', use_bias=True )( D )


# Phase 2: FGHIJ
F = LocallyConnected2D( name="layerF", filters=15, kernel_size=(3,2), padding='valid', input_shape=E.shape, data_format='channels_last', activation='relu', use_bias=True )( E )
G = LocallyConnected2D( name="layerG", filters=13, kernel_size=(3,2), padding='valid', input_shape=F.shape, data_format='channels_last', activation='relu', use_bias=True )( F )
H = LocallyConnected2D( name="layerH", filters=11, kernel_size=(3,2), padding='valid', input_shape=G.shape, data_format='channels_last', activation='relu', use_bias=True )( G )
I = LocallyConnected2D( name="layerI", filters=9, kernel_size=(3,2), padding='valid', input_shape=H.shape, data_format='channels_last', activation='relu', use_bias=True )( H )
J = LocallyConnected2D( name="layerJ", filters=7, kernel_size=(3,2), padding='valid', input_shape=I.shape, data_format='channels_last', activation='relu', use_bias=True )( I )

# Phase 3: flatJ, merge, KLMN, output
flatJ = Flatten( name="flatJ", data_format='channels_last' )( J )
print( flatJ.shape )
#exit( 1 )


merge = tensorflow.keras.layers.concatenate( [flatJ, in1_D], name="merge_flatJ_input2" )

 
denseK = Dense( name="denseK", units=500, activation='relu' )( merge )
denseL = Dense( name="denseL", units=500, activation='relu' )( denseK )
denseM = Dense( name="denseM", units=500, activation='relu' )( denseL )
denseN = Dense( name="denseN", units=500, activation='relu' )( denseM )
output = Dense( name="output", units=1, activation='linear' )( denseN )

model = Model(inputs=[input1, input2], outputs=output )

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
model.summary()
