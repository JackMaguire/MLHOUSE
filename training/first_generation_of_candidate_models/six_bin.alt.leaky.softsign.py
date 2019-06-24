import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#from keras import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras import metrics

from keras.models import load_model
import keras.backend as K
import keras.callbacks
import keras
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

#Get sha1
pwd = os.path.realpath(__file__)
MLHOUSE_index = pwd.find( "MLHOUSE" )
path = pwd[:MLHOUSE_index]
full_name = "~/MLHOUSE/.git".replace( "~", path )
sha1 = subprocess.check_output(["git", "--git-dir", full_name, "rev-parse", "HEAD"]).strip()
print ( "JackMaguire/MLHOUSE: " + str( sha1 ) )

parser = argparse.ArgumentParser()
parser.add_argument( "--model", help="filename for output file", default="six_bin.alt.leaky.softsign", required=False )
args = parser.parse_args()


num_input_dimensions = 9600
num_neurons_in_layer1 = 1500
num_neurons_in_layer2 = 500
num_output_dimensions = 12
model = Sequential()

model.add( Dense( num_neurons_in_layer1, input_dim=num_input_dimensions ) )
model.add( LeakyReLU(alpha=.01) )
model.add( Dense( num_neurons_in_layer2 ) )
model.add( LeakyReLU(alpha=.01) )
model.add( Dense( num_output_dimensions, activation='softsign') )

# 3) Compile Model

metrics_to_output=[ 'accuracy' ]
model.compile( loss='mean_absolute_error', optimizer='adam', metrics=metrics_to_output )
model.save( args.model + ".h5" )
