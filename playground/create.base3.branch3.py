import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


#import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
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
parser.add_argument( "--model", help="filename for output file", default="start.base3.branch3", required=False )
args = parser.parse_args()

num_input_dimensions1 = 26
num_input_dimensions2 = 18494 - num_input_dimensions1

input1 = Input(shape=(num_input_dimensions1,), name="in1", dtype="float32" )
in1dense1 = Dense( name="in1dense1", units=20 )( input1 )
in1dense1 = LeakyReLU( name="indense1_act" )( in1dense1 )
in1dense2 = Dense( name="in1dense2", units=10 )( in1dense1 )
in1dense2 = LeakyReLU( name="indense2_act" )( in1dense2 )
in1dense3 = Dense( name="in1dense3", units=10 )( in1dense2 )
in1dense3 = LeakyReLU( name="indense3_act" )( in1dense3 )
merge_depth = 4
in1dense4_size = 18 * 9 * merge_depth
in1dense4 = Dense( name="in1dense4", units=in1dense4_size, activation=None )( in1dense3 )
in1merge = Reshape( target_shape=(18, 9, merge_depth,) )( in1dense4 )#ALWAYS DO WIDTH, HEIGHT, CHANNELS


input2 = Input(shape=(num_input_dimensions2,), name="in2", dtype="float32" )
pre = Reshape( target_shape=(36, 19, 27,) )( input2 )#ALWAYS DO WIDTH, HEIGHT, CHANNELS
A = Conv2D( name="layerA", filters=8, kernel_size=(1,1), padding='valid',   data_format='channels_last', use_bias=True )( pre )
A = LeakyReLU()( A )
B1 = Conv2D( name="layerB1", filters=8, kernel_size=(1,1), padding='valid', data_format='channels_last', use_bias=True )( A )
B1 = LeakyReLU()( B1 )
B4 = LocallyConnected2D( name="layerB4", filters=merge_depth, kernel_size=(2,3), strides=(2,2), padding='valid', data_format='channels_last', activation=None, use_bias=True )( B1 )

X1a = Conv2D( name="layerX1a", filters=4, kernel_size=(1,1), padding='valid',   data_format='channels_last', use_bias=True )( pre )
X1a = LeakyReLU()( X1a )
Y1a = LocallyConnected2D( name="layerY1a", filters=4, kernel_size=(1,1), padding='valid', data_format='channels_last', use_bias=True )( X1a )
Y1a = LeakyReLU()( Y1a )
Z1a = LocallyConnected2D( name="layerZ1a", filters=merge_depth, kernel_size=(2,3), strides=(2,2), padding='valid', data_format='channels_last', activation=None, use_bias=True )( Y1a )

X1b = Conv2D( name="layerX1b", filters=4, kernel_size=(1,1), padding='valid',   data_format='channels_last', use_bias=True )( pre )
X1b = LeakyReLU()( X1b )
Y1b = LocallyConnected2D( name="layerY1b", filters=4, kernel_size=(1,1), padding='valid', data_format='channels_last', use_bias=True )( X1b )
Y1b = LeakyReLU()( Y1b )
Z1b = LocallyConnected2D( name="layerZ1b", filters=merge_depth, kernel_size=(2,3), strides=(2,2), padding='valid', data_format='channels_last', activation=None, use_bias=True )( Y1b )

X1c = Conv2D( name="layerX1c", filters=4, kernel_size=(1,1), padding='valid',   data_format='channels_last', use_bias=True )( pre )
X1c = LeakyReLU()( X1c )
Y1c = LocallyConnected2D( name="layerY1c", filters=4, kernel_size=(1,1), padding='valid', data_format='channels_last', use_bias=True )( X1c )
Y1c = LeakyReLU()( Y1c )
Z1c = LocallyConnected2D( name="layerZ1c", filters=merge_depth, kernel_size=(2,3), strides=(2,2), padding='valid', data_format='channels_last', activation=None, use_bias=True )( Y1c )

X1d = Conv2D( name="layerX1d", filters=4, kernel_size=(1,1), padding='valid',   data_format='channels_last', use_bias=True )( pre )
X1d = LeakyReLU()( X1d )
Y1d = LocallyConnected2D( name="layerY1d", filters=4, kernel_size=(1,1), padding='valid', data_format='channels_last', use_bias=True )( X1d )
Y1d = LeakyReLU()( Y1d )
Z1d = LocallyConnected2D( name="layerZ1d", filters=merge_depth, kernel_size=(2,3), strides=(2,2), padding='valid', data_format='channels_last', activation=None, use_bias=True )( Y1d )

X1e = Conv2D( name="layerX1e", filters=4, kernel_size=(1,1), padding='valid',   data_format='channels_last', use_bias=True )( pre )
X1e = LeakyReLU()( X1e )
Y1e = LocallyConnected2D( name="layerY1e", filters=4, kernel_size=(1,1), padding='valid', data_format='channels_last', use_bias=True )( X1e )
Y1e = LeakyReLU()( Y1e )
Z1e = LocallyConnected2D( name="layerZ1e", filters=merge_depth, kernel_size=(2,3), strides=(2,2), padding='valid', data_format='channels_last', activation=None, use_bias=True )( Y1e )

C = Add()( [ in1merge, Z1a, Z1b, Z1c, Z1d, Z1e ] )
C = LeakyReLU( )( C )
E = LocallyConnected2D( name="layerE", filters=4, kernel_size=(2,1), strides=(2,1), padding='valid', data_format='channels_last', use_bias=True )( C )
E = LeakyReLU()( E )

F = LocallyConnected2D( name="layerF", filters=3, kernel_size=(5,5), strides=(2,2), padding='valid', data_format='channels_last', use_bias=True )( E )
F = LeakyReLU()( F )
print( F.shape )
flat = Flatten( name="flat", data_format='channels_last' )( F )
 
dense1 = Dense( name="dense1", units=10 )( flat )
dense1 = LeakyReLU( name="dense1_act" )( dense1 )
dense2 = Dense( name="dense2", units=5 )( dense1 )
dense2 = LeakyReLU( name="dense2_act" )( dense2 )
output = Dense( name="output", units=1, activation='linear' )( dense2 )

model = Model(inputs=[input1, input2], outputs=output )

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
model.summary()
