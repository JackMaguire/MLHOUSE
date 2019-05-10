import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#from keras import *
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics

from keras.models import load_model
import keras.backend as K
import keras.callbacks
import keras
import numpy

import sys
#sys.path.append("/nas/longleaf/home/jackmag")#for h5py
import h5py

import argparse
import random
import time
import subprocess

import tensorflow as tf

########
# INIT #
########

numpy.random.seed( 0 )

#Get sha1
pwd = os.path.realpath(__file__)
MLHOUSE_index = pwd.find( "MLHOUSE" )
path = pwd[:MLHOUSE_index]
full_name = "~/MLHOUSE/.git".replace( "~", path )
sha1 = subprocess.check_output(["git", "--git-dir", full_name, "rev-parse", "HEAD"]).strip()
print ( "JackMaguire/MLHOUSE: " + str( sha1 ) )

parser = argparse.ArgumentParser()
parser.add_argument( "--model", help="filename for output file", default="very_small_blank_model", required=False )
args = parser.parse_args()


num_input_dimensions = 10
num_neurons_in_middle_layer = 5
num_output_dimensions = 2
model = Sequential()

model.add( Dense( num_neurons_in_middle_layer, input_dim=num_input_dimensions, activation='relu') )
model.add( Dense( num_output_dimensions, activation='linear') )

# 3) Compile Model

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )

dummy_training_input = numpy.zeros( shape=( 1, 10 ) )
for x in range( 0, 9 ):
    dummy_training_input[ 0 ][ x ] = 2 - ( x / 10 )

dummy_training_output = numpy.zeros( shape=( 1, 2 ) )
dummy_training_input[ 0 ][ 0 ] = -1
dummy_training_input[ 0 ][ 1 ] = 1


dummy_test_input = numpy.zeros( shape=( 1, 10 ) )
for x in range( 0, 9 ):
    dummy_test_input[ 0 ][ x ] = 3
print( dummy_test_input )
    
predictions = model.predict( x=dummy_test_input[:] )
print( predictions )
