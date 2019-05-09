import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from random import shuffle

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
import gzip

import argparse
import random

import threading
import time
import subprocess

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

########
# INIT #
########

num_input_dimensions = 17809 #TODO UPDATE
#num_neurons_in_layer1 = 4000
#num_neurons_in_layer2 = 2000
#num_neurons_in_layer3 = 500
#num_neurons_in_layer4 = 100
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

parser.add_argument( "--nthread", help="Number of threads to use", type=int, required=True )

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
    if split[ 0 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 0 ], "r" )
        input = pd.read_csv( f ).values
        f.close()
    elif split[ 0 ].endswith( ".csv" ):
        input = pd.read_csv( split[ 0 ] ).values
    elif split[ 0 ].endswith( ".npy.gz" ):
        f = gzip.GzipFile( split[ 0 ], "r" )
        input = numpy.load( f )
        f.close()
    elif split[ 0 ].endswith( ".npy" ):
        input = numpy.load( split[ 0 ] )
    else:
        print ( "We cannot open this file format: " + split[ 0 ] )
        exit( 1 )

    if split[ 1 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 1 ], "r" )
        output = pd.read_csv( f ).values
        f.close()
    elif split[ 1 ].endswith( ".csv" ):
        output = pd.read_csv( split[ 1 ] ).values
    elif split[ 1 ].endswith( ".npy.gz" ):
        f = gzip.GzipFile( split[ 1 ], "r" )
        output = numpy.load( f )
        f.close()
    elif split[ 1 ].endswith( ".npy" ):
        output = numpy.load( split[ 1 ] )
    else:
        print ( "We cannot open this file format: " + split[ 1 ] )
        exit( 1 )

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
class DataLoadingThread( threading.Thread ):
    def __init__( self ):
        threading.Thread.__init__( self )
        self.dead = True

    def set_filenames( self, setting ):
        self.filenames = setting
        self.dead = False

    def set_dead( self, setting ):
        self.dead = setting

    def is_dead( self ):
        return self.dead

    def get_results( self ):
        return self.input, self.output

    def run( self ):
        self.input, self.output = generate_data_from_files( self.filenames )

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

n_threads = args.nthread

time_of_last_save = time.time()

save_frequency_in_seconds = args.epoch_checkpoint_frequency_in_hours * 60 * 60

with open( args.training_data, "r" ) as f:
    file_lines = f.readlines()
num_file_lines = len( file_lines )

time_spent_loading = 0
time_spent_training = 0

for epoch in range( starting_epoch + 1, last_epoch + 1 ):

    shuffle( file_lines )

    data_loaders = [ DataLoadingThread() for count in range( 0, n_threads ) ]
    print( "created " + str( len( data_loaders ) ) + " data loaders" )

    current_line = 0
    while True:
        if current_line >= num_file_lines:
            break

        t0 = time.time()

        #spawn loads
        for x in range( 0, n_threads ):
            index = current_line + x
            if index < num_file_lines:
                data_loaders[ x ].set_filenames( file_lines[ index ] )
                data_loaders[ x ].start()
            else:
                data_loaders[ x ].set_dead( True )
        current_line += n_threads

        t1 = time.time()
        time_spent_loading += t1 - t0

        #Access loaded data for training
        for x in range( 0, n_threads ):
            if data_loaders[ x ].is_dead():
                continue
            t0a = time.time()
            while data_loaders[ x ].isAlive():
                #wait for this one to finish
                pass            
            input, output = data_loaders[ x ].get_results()
            t1a = time.time()
            time_spent_loading += t1a - t0a

            t2 = time.time()
            model.train_on_batch( x=input, y=output )
            t3 = time.time()
            time_spent_training += t3 - t2

    if ( time.time() - time_of_last_save >= save_frequency_in_seconds ):
        time_of_last_save = time.time()
        model.save( "epoch_" + str( epoch ) + ".h5" )

print( str( float( time_spent_loading ) / float(time_spent_loading + time_spent_training) ) + " fraction of time was spent loading" )
print( time_spent_loading )
print( time_spent_training )

model.save( "final.h5" )
