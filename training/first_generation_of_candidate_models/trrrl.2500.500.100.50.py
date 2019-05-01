import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
parser.add_argument( "--model", help="filename for output file", default="blank_model", required=False )
args = parser.parse_args()


num_input_dimensions = 20021
num_neurons_in_layer1 = 2500
num_neurons_in_layer2 = 500
num_neurons_in_layer3 = 100
num_neurons_in_layer4 = 50
num_output_dimensions = 2
model = Sequential()

model.add( Dense( num_neurons_in_layer1, input_dim=num_input_dimensions, activation='tanh') )
model.add( Dense( num_neurons_in_layer2, activation='relu') )
model.add( Dense( num_neurons_in_layer3, activation='relu') )
model.add( Dense( num_neurons_in_layer4, activation='relu') )
model.add( Dense( num_output_dimensions, activation='linear') )

# 3) Compile Model

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
