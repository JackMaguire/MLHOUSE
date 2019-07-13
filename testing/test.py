#import jack_mouse_test

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from random import shuffle

#from tensorflow.keras import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics

import tensorflow.keras.losses

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks
import tensorflow.keras
import numpy

import sys
#sys.path.append("/nas/longleaf/home/jackmag")#for h5py
import h5py

import pandas as pd
import gzip
import math

import argparse
import random

#import threading
import time
import subprocess

import scipy
import scipy.stats

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import tensorflow as tf
#from tensorflow.keras.backend.tensorflow_backend import set_session

'''
#Only use part of the GPU, from https://github.com/keras-team/keras/issues/4161
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.visible_device_list = "0"
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

'''

########
# INIT #
########

num_input_dimensions = 18494
num_source_residue_inputs = 26
num_ray_inputs = 18494 - 26
WIDTH = 36
HEIGHT = 19
CHANNELS = 27
if( WIDTH * HEIGHT * CHANNELS != num_ray_inputs ):
    print( "WIDTH * HEIGHT * CHANNELS != num_ray_inputs" )
    exit( 1 )
num_output_dimensions = 1

numpy.random.seed( 0 )

parser = argparse.ArgumentParser()

parser.add_argument( "--model", help="Most recent model file", required=True )

parser.add_argument( "--data", help="CSV where each line has two elements. First element is the absolute path to the input csv file, second element is the absolute path to the corresponding output csv file.", required=True )
# Example: "--data foo.csv" where foo.csv looks like:
# /home/jack/input.1.csv,/home/jack/output.1.csv
# /home/jack/input.2.csv,/home/jack/output.2.csv
# /home/jack/input.3.csv,/home/jack/output.3.csv
# ...

args = parser.parse_args()

#########
# FUNCS #
#########

def hello_world():
    x1 = [ 1., 2., 4. ]
    y1 = [ 2., 4., 6. ]

    x2 = [ 1., 2., 4. ]
    y2 = [ 0., 3., 3. ]

    print( scipy.stats.pearsonr( x1, y1 )[ 0 ] )
    print( scipy.stats.pearsonr( x2, y2 )[ 0 ] )

    print( scipy.stats.spearmanr( x1, y1 ).correlation )
    print( scipy.stats.spearmanr( x2, y2 ).correlation )

    print( scipy.stats.kendalltau( x1, y1 ).correlation )
    print( scipy.stats.kendalltau( x2, y2 ).correlation )


def my_assert_equals( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        exit( 1 )

class AssertError(Exception):
    pass

def my_assert_equals_thrower( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        raise AssertError

#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

def assert_vecs_line_up( input, output ):
    #Each line starts with "RESID: XXX,"
    my_assert_equals_thrower( "input file length", len( input ), len( output ) )
    for i in range( 0, len( input ) ):
        in_elem = input[ i ][ 0 ]
        in_resid = int( in_elem.split( " " )[ 1 ] )
        out_elem = output[ i ][ 0 ]
        out_resid = int( out_elem.split( " " )[ 1 ] )
        my_assert_equals_thrower( "out_resid", out_resid, in_resid )

def generate_data_from_files( filenames_csv, six_bin ):
    #dataset = numpy.genfromtxt( filename, delimiter=",", skip_header=0 )
    split = filenames_csv.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    #t0 = time.time()

    # Both of these elements lead with a dummy
    if split[ 0 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 0 ], "r" )
        input = pd.read_csv( f, header=None ).values
        f.close()
    elif split[ 0 ].endswith( ".csv" ):
        input = pd.read_csv( split[ 0 ], header=None ).values
    elif split[ 0 ].endswith( ".npy.gz" ):
        f = gzip.GzipFile( split[ 0 ], "r" )
        input = numpy.load( f, allow_pickle=True )
        f.close()
    elif split[ 0 ].endswith( ".npy" ):
        input = numpy.load( split[ 0 ], allow_pickle=True )
    else:
        print ( "We cannot open this file format: " + split[ 0 ] )
        exit( 1 )

    if split[ 1 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 1 ], "r" )
        output = pd.read_csv( f, header=None ).values
        f.close()
    elif split[ 1 ].endswith( ".csv" ):
        output = pd.read_csv( split[ 1 ], header=None ).values
    elif split[ 1 ].endswith( ".npy.gz" ):
        f = gzip.GzipFile( split[ 1 ], "r" )
        output = numpy.load( f, allow_pickle=True )
        f.close()
    elif split[ 1 ].endswith( ".npy" ):
        output = numpy.load( split[ 1 ], allow_pickle=True )
    else:
        print ( "We cannot open this file format: " + split[ 1 ] )
        exit( 1 )

    assert_vecs_line_up( input, output )

    source_input_no_resid = input[:,1:27]
    ray_input_no_resid = input[:,27:]

    #print( "output.shape:" )
    #print( output.shape )
    output_no_resid = output[:,1:2]
    #print( "output_no_resid.shape:" )
    #print( output_no_resid.shape )

    my_assert_equals_thrower( "len( source_input_no_resid[ 0 ] )", len( source_input_no_resid[ 0 ] ), num_source_residue_inputs );
    my_assert_equals_thrower( "len( ray_input_no_resid[ 0 ] )",    len( ray_input_no_resid[ 0 ] ), num_ray_inputs );

    #https://www.kaggle.com/vishwasgpai/guide-for-creating-cnn-model-using-csv-file        

    if six_bin:
        six_bin_output_no_resid = output_no_resid.copy()
        new_shape = ( output_no_resid.shape[ 0 ], num_output_dimensions )
        six_bin_output_no_resid.resize( new_shape )
        for x in range( 0, len( output_no_resid ) ):
            my_assert_equals_thrower( "len(six_bin_output_no_resid[ x ])", len( six_bin_output_no_resid[ x ] ), num_output_dimensions )
            six_bin_output_no_resid[ x ][ 0 ] = 1.0 if output_no_resid[ x ][ 0 ] <= -7.0 else 0.0
            six_bin_output_no_resid[ x ][ 1 ] = 1.0 if output_no_resid[ x ][ 0 ] <= -5.0 else 0.0
            six_bin_output_no_resid[ x ][ 2 ] = 1.0 if output_no_resid[ x ][ 0 ] <= -3.0 else 0.0
            six_bin_output_no_resid[ x ][ 3 ] = 1.0 if output_no_resid[ x ][ 0 ] <= -1.0 else 0.0
            six_bin_output_no_resid[ x ][ 4 ] = 1.0 if output_no_resid[ x ][ 0 ] <= 1.0  else 0.0
            six_bin_output_no_resid[ x ][ 5 ] = 1.0 if output_no_resid[ x ][ 0 ] <= 3.0  else 0.0
            six_bin_output_no_resid[ x ][ 6 ] = 1.0 if output_no_resid[ x ][ 0 ] <= 5.0  else 0.0
            six_bin_output_no_resid[ x ][ 7 ] = 1.0 if output_no_resid[ x ][ 0 ] <= 7.0  else 0.0
        return source_input_no_resid, ray_input_no_resid, six_bin_output_no_resid
    else:
        my_assert_equals_thrower( "len( output_no_resid[ 0 ] )", len( output_no_resid[ 0 ] ), num_output_dimensions );
        for x in range( 0, len( output_no_resid ) ):
            my_assert_equals_thrower( "len(output_no_resid[x])", len(output_no_resid[x]), 1 )
            val = output_no_resid[x][0]
            #Stunt large values
            if( val > 1 ):
                val = val**0.75
            #subtract mean of -2:
            val += 2.0
            #divide by span of 3:
            val /= 3.0
        return source_input_no_resid, ray_input_no_resid, output_no_resid


#########
# START #
#########

if os.path.isfile( args.model ):
    model = load_model( args.model )
else:
    print( "Model " + args.model + " is not a file" )
    exit( 1 )

allxs = []
allys = []

xs_lt_0 = []
ys_lt_0 = []

xs_lt_n2 = []
ys_lt_n2 = []

xs_lt_n4 = []
ys_lt_n4 = []

xs_lt_n6 = []
ys_lt_n6 = []



# 4) Fit Model

with open( args.data, "r" ) as f:
    file_lines = f.readlines()

for line in file_lines:
    #print( "reading " + str( line ) )
    #split = line.split( "\n" )[ 0 ].split( "," );
    #my_assert_equals_thrower( "split.length", len( split ), 2 );

    try:
        #cpp_structs = jack_mouse_test.read_mouse_data( split[ 0 ], split[ 1 ] )
        #source_input = cpp_structs[ 0 ]
        #ray_input = cpp_structs[ 1 ]
        #output = cpp_structs[ 2 ]
        source_input, ray_input, output = generate_data_from_files( line, False )
    except AssertError:
        continue
    predictions = model.predict( x=[source_input,ray_input] )
    my_assert_equals_thrower( "len( predictions )", len( predictions ), len( output ) );
    print( predictions[ 0 ].shape )
    print( output[ 0 ].shape )
    exit( 0 )
