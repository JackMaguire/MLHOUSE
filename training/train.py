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

#import threading
import time
import subprocess

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

########
# INIT #
########

num_input_dimensions = 9600 #TODO UPDATE
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

#parser.add_argument( "--nthread", help="Number of threads to use", type=int, required=True )

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

time_box1 = 0
time_box2 = 0
time_box3 = 0
time_box4 = 0
time_box5 = 0

def generate_data_from_files( filenames_csv ):
    #dataset = numpy.genfromtxt( filename, delimiter=",", skip_header=0 )
    split = filenames_csv.split( "\n" )[ 0 ].split( "," );
    my_assert_equals( "split.length", len( split ), 2 );

    t0 = time.time()

    # Both of these elements lead with a dummy
    if split[ 0 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 0 ], "r" )
        input = pd.read_csv( f ).values
        f.close()
    elif split[ 0 ].endswith( ".csv" ):
        input = pd.read_csv( split[ 0 ] ).values
    elif split[ 0 ].endswith( ".npy.gz" ):
        f = gzip.GzipFile( split[ 0 ], "r" )
        input = numpy.load( f, allow_pickle=True )
        f.close()
    elif split[ 0 ].endswith( ".npy" ):
        input = numpy.load( split[ 0 ], allow_pickle=True )
    else:
        print ( "We cannot open this file format: " + split[ 0 ] )
        exit( 1 )

    t1 = time.time()

    if split[ 1 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 1 ], "r" )
        output = pd.read_csv( f ).values
        f.close()
    elif split[ 1 ].endswith( ".csv" ):
        output = pd.read_csv( split[ 1 ] ).values
    elif split[ 1 ].endswith( ".npy.gz" ):
        f = gzip.GzipFile( split[ 1 ], "r" )
        output = numpy.load( f, allow_pickle=True )
        f.close()
    elif split[ 1 ].endswith( ".npy" ):
        output = numpy.load( split[ 1 ], allow_pickle=True )
    else:
        print ( "We cannot open this file format: " + split[ 1 ] )
        exit( 1 )

    t2 = time.time()

    assert_vecs_line_up( input, output )

    t3 = time.time()

    input_no_resid = input[:,1:]

    t4 = time.time()

    output_no_resid = output[:,1:]

    t5 = time.time()

    global time_box1
    global time_box2
    global time_box3
    global time_box4
    global time_box5

    time_box1 += t1 - t0
    time_box2 += t2 - t1
    time_box3 += t3 - t2
    time_box4 += t4 - t3
    time_box5 += t5 - t4

    my_assert_equals( "len( input_no_resid[ 0 ] )", len( input_no_resid[ 0 ] ), num_input_dimensions );
    my_assert_equals( "len( output_no_resid[ 0 ] )", len( output_no_resid[ 0 ] ), num_output_dimensions );

    #https://www.kaggle.com/vishwasgpai/guide-for-creating-cnn-model-using-csv-file

    return input_no_resid, output_no_resid


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

#n_threads = args.nthread

time_of_last_save = time.time()

save_frequency_in_seconds = args.epoch_checkpoint_frequency_in_hours * 60 * 60

with open( args.training_data, "r" ) as f:
    file_lines = f.readlines()
num_file_lines = len( file_lines )

time_spent_loading = float( 0 )
time_spent_training = float( 0 )

next_save_number = int( 1 )

for epoch in range( starting_epoch + 1, last_epoch + 1 ):

    shuffle( file_lines )

    for line in file_lines:
        print( "reading " + str( line ) )
        t0 = time.time()
        input, output = generate_data_from_files( line )
        t1 = time.time()
        model.train_on_batch( x=input, y=output )
        t2 = time.time()
        time_spent_loading += t1 - t0
        time_spent_training += t2 - t1

        if ( time.time() - time_of_last_save >= save_frequency_in_seconds ):
            time_of_last_save = time.time()
            model.save( "save_" + str( next_save_number ) + ".h5" )
            if next_save_number > 1:
                os.remove( "save_" + str( next_save_number - 1 ) + ".h5" )
            next_save_number = next_save_number + 1

print( str( float( time_spent_loading ) / float(time_spent_loading + time_spent_training) ) + " fraction of time was spent loading" )
print( time_spent_loading )
print( time_spent_training )

all_time_boxes = time_box1 + time_box2 + time_box3 + time_box4 + time_box5

print( "time_box1: " + str( time_box1 ) + " " + str( float( time_box1 ) / float( all_time_boxes ) ) )
print( "time_box2: " + str( time_box2 ) + " " + str( float( time_box2 ) / float( all_time_boxes ) ) )
print( "time_box3: " + str( time_box3 ) + " " + str( float( time_box3 ) / float( all_time_boxes ) ) )
print( "time_box4: " + str( time_box4 ) + " " + str( float( time_box4 ) / float( all_time_boxes ) ) )
print( "time_box5: " + str( time_box5 ) + " " + str( float( time_box5 ) / float( all_time_boxes ) ) )

model.save( "final.h5" )
