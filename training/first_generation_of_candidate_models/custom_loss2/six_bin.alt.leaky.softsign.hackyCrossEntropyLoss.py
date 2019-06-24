import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#from keras import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras import metrics
from keras import optimizers

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
parser.add_argument( "--model", help="filename for output file", default="six_bin.alt.leaky.softsign.hackyCrossEntropyLoss", required=False )
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

#https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
def custom_loss():
    def loss( y_true, y_pred ):
        #using notation from https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html where
        #p: predicted probability ( 0 to 1 )
        #y: actual probability ( 0 to 1 )
        #y_pred: predicted probability (-1 to 1)
        p=(y_pred + 1)/2
        if y_true == 1:
            return -1*K.log( p )
        else: # y_true == 0 or -1
            return -1*K.log( 1 - p )

    return loss


metrics_to_output=[ 'accuracy' ]
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile( loss=custom_loss(), optimizer=sgd, metrics=metrics_to_output )
model.save( args.model + ".h5" )
