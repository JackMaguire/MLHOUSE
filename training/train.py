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

import pandas as pd

import argparse
import random

import threading
import time
import subprocess

########
# INIT #
########

num_input_dimensions = 20021
num_neurons_in_layer1 = 4000
num_neurons_in_layer2 = 2000
num_neurons_in_layer3 = 500
num_neurons_in_layer4 = 100
num_output_dimensions = 2

numpy.random.seed( 0 )

#Get sha1
pwd = os.path.realpath(__file__)
MLHOUSE_index = pwd.find( "MLHOUSE" )
path = pwd[:MLHOUSE_index]
full_name = "~/MLHOUSE/.git".replace( "~", path )
sha1 = subprocess.check_output(["git", "--git-dir", full_name, "rev-parse", "HEAD"]).strip()
print ( "JackMaguire/MLHOUSE: " + str( sha1 ) )

parser = argparse.ArgumentParser()

parser.add_argument( "--model", help="Most recent model file", required=True )

parser.add_argument( "--training_data", help="CSV where each line has two elements. First element is the absolute path to the input csv file, second element is the absolute path to the corresponding output csv file.", required=True )
# Example: "--training_data foo.csv" where foo.csv looks like:
# /home/jack/input.1.csv,/home/jack/output.1.csv
# /home/jack/input.2.csv,/home/jack/output.2.csv
# /home/jack/input.3.csv,/home/jack/output.3.csv
# ...

parser.add_argument( "--starting_epoch", help="For bookkeeping purposes, what is the epoch number of the model loaded with --model?", type=int, required=True )
parser.add_argument( "--epoch_checkpoint_frequency_in_hours", help="How often should we be saving models?", type=int, required=True )
parser.add_argument( "--num_epochs", help="Number of epochs to run.", type=int, required=True )

parser.add_argument( "--prefetch", help="Save time by prefetching data in a different thread", type=bool, required=False, default=True )


args = parser.parse_args()

#num_neurons_in_first_hidden_layer = args.num_neurons_in_first_hidden_layer
#print( "num_neurons_in_first_hidden_layer: " + str( num_neurons_in_first_hidden_layer ) )

#########
# FUNCS #
#########

def my_assert_equals( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        exit( 1 )


#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

def assert_vecs_line_up( input, output ):
    #Each line starts with "RESID: XXX,"
    my_assert_equals( "input file length", len( input ), len( output ) )
    for i in range( 0, len( input ) ):
        in_elem = input[ i ][ 0 ]
        in_resid = int( in_elem.split( " " )[ 1 ] )
        out_elem = output[ i ][ 0 ]
        out_resid = int( out_elem.split( " " )[ 1 ] )
        my_assert_equals( "out_resid", out_resid, in_resid )

def generate_data_from_files( filenames_csv ):
    #dataset = numpy.genfromtxt( filename, delimiter=",", skip_header=0 )
    split = filenames_csv.split( "\n" )[ 0 ].split( "," );
    my_assert_equals( "split.length", len( split ), 2 );

    # Both of these elements lead with a dummy
    input  = pd.read_csv( split[ 0 ] ).values
    output = pd.read_csv( split[ 1 ] ).values

    assert_vecs_line_up( input, output )

    input_no_resid = input[:,1:]
    output_no_resid = output[:,1:]

    my_assert_equals( "len( input_no_resid[ 0 ] )", len( input_no_resid[ 0 ] ), num_input_dimensions );
    my_assert_equals( "len( output_no_resid[ 0 ] )", len( output_no_resid[ 0 ] ), num_output_dimensions );

    #https://www.kaggle.com/vishwasgpai/guide-for-creating-cnn-model-using-csv-file

    return input_no_resid, output_no_resid

#generate_data_from_files( "../sample_data/sample.repack.input.csv,../sample_data/sample.repack.output.csv" )
#exit( 0 )

###########
# CLASSES #
###########
#
# Possible state values:
# 0: Not yet loaded
# 1: Running
# 2: Loaded
class PrefetchThread( threading.Thread ):
    def __init__( self, threadID ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.next_filenames = ""
        self.state = 0

    def set_next_filenames( setting ):
        self.next_filenames = setting

    def get_results():
        return self.input, self.output

    def get_state():
        return self.state

    def set_state( setting ):
        self.state = setting

    def run(self):
        my_assert_equals( "state", self.state, 0 )
        self.state = 1
        self.input, self.output = generate_data_from_files( self.next_filenames )
        self.state = 2


###########
# METRICS #
###########

def mean_pred( y_true, y_pred ):
    return K.mean( y_pred )

#########
# START #
#########


if os.path.isfile( args.model ):
    model = load_model( args.model )
else:
    print( "Model " + args.model + " is not a file" )
    exit( 1 )

# 4) Fit Model
starting_epoch = args.starting_epoch
last_epoch = starting_epoch + args.num_epochs

time_of_last_save = time.time()

save_frequency_in_seconds = args.epoch_checkpoint_frequency_in_hours * 60 * 60

for epoch in range( starting_epoch + 1, last_epoch + 1 ):

    prefetcher = PrefetchThread( 0 )
    prefetcher.set_state( 0 )

    file = open( args.training_data, "r" )

    for line in file:
        while prefetcher.isAlive() == 1:
            my_assert_equals( "prefetcher state", prefetcher.get_state(), 1 )

        if prefetcher.get_state() == 2 :
            input, output = prefetcher.get_results() #generate_data_from_files( line )
            prefetcher.set_next_filenames( line )
            prefetcher.set_state( 0 )
            prefetcher.start()
            model.train_on_batch( x=input, y=output )
        else : #this must be the first line
            prefetcher.set_next_filenames( line )
            prefetcher.set_state( 0 )
            prefetcher.start()

    file.close()
    
    #run final batch
    if prefetcher.get_state() == 2 :
        input, output = prefetcher.get_results() #generate_data_from_files( line )
        prefetcher.set_state( 0 )
        model.train_on_batch( x=input, y=output )

    if ( time.time() - time_of_last_save >= save_frequency_in_seconds ):
        time_of_last_save = time.time()
        model.save( "epoch_" + str( epoch ) + ".h5" )

model.save( "final.h5" )
