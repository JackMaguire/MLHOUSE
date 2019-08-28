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
parser.add_argument( "--model", help="filename for output file", default="start.base2.lcn.sum", required=False )
args = parser.parse_args()

num_input_dimensions1 = 26
num_input_dimensions2 = 18494 - num_input_dimensions1

input1 = Input(shape=(num_input_dimensions1,), name="in1", dtype="float32" )
in1dense1 = Dense( name="in1dense1", units=50 )( input1 )
in1dense1_act = LeakyReLU( name="indense1_act" )( in1dense1 )
in1dense2 = Dense( name="in1dense2", units=50 )( in1dense1_act )
in1dense2_act = LeakyReLU( name="indense2_act" )( in1dense2 )
in1dense3 = Dense( name="in1dense3", units=20 )( in1dense2_act )
in1dense3_act = LeakyReLU( name="indense3_act" )( in1dense3 )
in1dense4_size = 18 * 9 * 15
in1dense4 = Dense( name="in1dense4", units=in1dense4_size, activation=None )( in1dense3_act )
in1merge = Reshape( target_shape=(18, 9, 15,) )( in1dense4 )#ALWAYS DO WIDTH, HEIGHT, CHANNELS


#input2 = Input(shape=(36, 19, 27,), name="in2", dtype="float32" )
input2 = Input(shape=(num_input_dimensions2,), name="in2", dtype="float32" )

pre = Reshape( target_shape=(36, 19, 27,) )( input2 )#ALWAYS DO WIDTH, HEIGHT, CHANNELS
A = Conv2D( name="layerA", filters=50, kernel_size=(1,1), padding='valid',   data_format='channels_last', use_bias=True )( pre )
A = LeakyReLU()( A )
B1 = Conv2D( name="layerB1", filters=30, kernel_size=(1,1), padding='valid', data_format='channels_last', use_bias=True )( A )
B1 = LeakyReLU()( B1 )
B2 = Conv2D( name="layerB2", filters=20, kernel_size=(1,1), padding='valid', data_format='channels_last', use_bias=True )( B1 )
B2 = LeakyReLU()( B2 )
B3 = Conv2D( name="layerB3", filters=12, kernel_size=(1,1), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( B2 )
B3 = LeakyReLU()( B3 )
B4 = LocallyConnected2D( name="layerB4", filters=15, kernel_size=(2,3), strides=(2,2), padding='valid', data_format='channels_last', activation=None, use_bias=True )( B3 )


C = Add()( [B4, in1merge] )
C = LeakyReLU( )( C )
#D = LocallyConnected2D( name="layerD", filters=10, kernel_size=(1,1), strides=(1,1), padding='valid', data_format='channels_last', use_bias=True )( C )
#D = LeakyReLU()( D )
E = LocallyConnected2D( name="layerE", filters=15, kernel_size=(2,1), strides=(2,1), padding='valid', data_format='channels_last', use_bias=True )( C )
E = LeakyReLU()( E )

#Epool = MaxPooling2D(pool_size=(2, 1), strides=(2,1), padding='valid', data_format='channels_last')( E )
#print( Epool.shape )
F = LocallyConnected2D( name="layerF", filters=15, kernel_size=(5,5), strides=(2,2), padding='valid', data_format='channels_last', activation='relu', use_bias=True )( E )
print( F.shape )
flat = Flatten( name="flat", data_format='channels_last' )( F )
 
dense1 = Dense( name="dense1", units=100 )( flat )
dense1_act = LeakyReLU( name="dense1_act" )( dense1 )
dense2 = Dense( name="dense2", units=50 )( dense1_act )
dense2_act = LeakyReLU( name="dense2_act" )( dense2 )
dense3 = Dense( name="dense3", units=50 )( dense2_act )
dense3_act = LeakyReLU( name="dense3_act" )( dense3 )
dense4 = Dense( name="dense4", units=50 )( dense3_act )
dense4_act = LeakyReLU( name="dense4_act" )( dense4 )
output = Dense( name="output", units=1, activation='linear' )( dense4_act )

model = Model(inputs=[input1, input2], outputs=output )

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
model.summary()
