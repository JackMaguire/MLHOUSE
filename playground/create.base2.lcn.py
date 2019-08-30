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

a_val=0.25

numpy.random.seed( 0 )

parser = argparse.ArgumentParser()
parser.add_argument( "--model", help="filename for output file", default="start.base2.lcn", required=False )
args = parser.parse_args()

num_input_dimensions1 = 26
num_input_dimensions2 = 18494 - num_input_dimensions1

input1 = Input(shape=(num_input_dimensions1,), name="in1", dtype="float32" )
in1dense1 = Dense( name="in1dense1", units=50 )( input1 )
in1dense1_act = LeakyReLU( name="indense1_act", alpha=a_val )( in1dense1 )
in1dense2 = Dense( name="in1dense2", units=50 )( in1dense1_act )
in1dense2_act = LeakyReLU( name="indense2_act", alpha=a_val )( in1dense2 )
in1dense3 = Dense( name="in1dense3", units=20 )( in1dense2_act )
in1dense3_act = LeakyReLU( name="indense3_act", alpha=a_val )( in1dense3 )
pre1 = Reshape( target_shape=(1,1,20,) )( in1dense3_act )#ALWAYS DO WIDTH, HEIGHT, CHANNELS
in1up = UpSampling3D( size=(18,9,1), data_format='channels_last' )( pre1 )


#input2 = Input(shape=(36, 19, 27,), name="in2", dtype="float32" )
input2 = Input(shape=(num_input_dimensions2,), name="in2", dtype="float32" )

# Phase 1: in2 -> ABCDE
pre = Reshape( target_shape=(36, 19, 27,) )( input2 )#ALWAYS DO WIDTH, HEIGHT, CHANNELS
A = Conv2D( name="layerA", filters=30, kernel_size=(1,1), padding='valid',   data_format='channels_last', activation='relu', use_bias=True )( pre )
B1 = Conv2D( name="layerB1", filters=20, kernel_size=(1,1), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( A )
B2 = Conv2D( name="layerB2", filters=20, kernel_size=(1,1), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( B1 )
B3 = Conv2D( name="layerB3", filters=12, kernel_size=(1,1), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( B2 )
B4 = LocallyConnected2D( name="layerB4", filters=15, kernel_size=(2,3), strides=(2,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( B3 )
merge = tensorflow.keras.layers.concatenate( [B4,in1up], name="merge", axis=-1 )

# Phase 2: FGHIJ
C = LocallyConnected2D( name="layerC", filters=12, kernel_size=(1,1), strides=(1,1), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( merge )
D = LocallyConnected2D( name="layerD", filters=10, kernel_size=(1,1), strides=(1,1), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( C )
E = LocallyConnected2D( name="layerE", filters=10, kernel_size=(2,1), strides=(2,1), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( D )

#Epool = MaxPooling2D(pool_size=(2, 1), strides=(2,1), padding='valid', data_format='channels_last')( E )
#print( Epool.shape )
F = LocallyConnected2D( name="layerF", filters=10, kernel_size=(5,5), strides=(2,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( E )
print( F.shape )
flat = Flatten( name="flat", data_format='channels_last' )( F )
 
dense1 = Dense( name="dense1", units=100 )( flat )
dense1_act = LeakyReLU( name="dense1_act", alpha=a_val )( dense1 )
dense2 = Dense( name="dense2", units=50 )( dense1_act )
dense2_act = LeakyReLU( name="dense2_act", alpha=a_val )( dense2 )
dense3 = Dense( name="dense3", units=50 )( dense2_act )
dense3_act = LeakyReLU( name="dense3_act", alpha=a_val )( dense3 )
dense4 = Dense( name="dense4", units=50 )( dense3_act )
dense4_act = LeakyReLU( name="dense4_act", alpha=a_val )( dense4 )
output = Dense( name="output", units=1, activation='linear' )( dense4_act )

model = Model(inputs=[input1, input2], outputs=output )

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
model.summary()
