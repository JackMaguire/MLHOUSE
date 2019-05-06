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
import gzip

import sys
#sys.path.append("/nas/longleaf/home/jackmag")#for h5py
import h5py

import pandas as pd

import argparse
import random
import time
import subprocess

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#Only use part of the GPU, from https://github.com/keras-team/keras/issues/4161
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.visible_device_list = "0"
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

########
# INIT #
########

num_input_dimensions = 17809
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

#parser.add_argument( "--evaluate_individually", help="Print predictions vs actual data", type=bool, default=False )

parser.add_argument( "--testing_data", help="CSV where each line has two elements. First element is the absolute path to the input csv file, second element is the absolute path to the corresponding output csv file.", required=True )
# Example: "--testing_data foo.csv" where foo.csv looks like:
# /home/jack/input.1.csv,/home/jack/output.1.csv
# /home/jack/input.2.csv,/home/jack/output.2.csv
# /home/jack/input.3.csv,/home/jack/output.3.csv
# ...

args = parser.parse_args()

#########
# FUNCS #
#########

def my_assert_equals( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        exit( 1 )


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
        input = numpy.load( f, allow_pickle=True )
        f.close()
    elif split[ 0 ].endswith( ".npy" ):
        input = numpy.load( split[ 0 ], allow_pickle=True )
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
        output = numpy.load( f, allow_pickle=True )
        f.close()
    elif split[ 1 ].endswith( ".npy" ):
        output = numpy.load( split[ 1 ], allow_pickle=True )
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


#########
# START #
#########


if os.path.isfile( args.model ):
    model = load_model( args.model )
else:
    print( "Model " + args.model + " is not a file" )
    exit( 1 )

# 4) Test Model
count = 0
int_count = 0
deviation_score = 0
deviation_score_int = 0
deviation_ddg = 0
deviation_ddg_int = 0

file = open( args.testing_data, "r" )

for line in file:
    try:
        input, output = generate_data_from_files( line )    
    except pd.errors.EmptyDataError:
        print( "pd.errors.EmptyDataError" )
        continue

    predictions = model.predict( x=input[:] )
    for i in range( 0, len( input ) ):
        count = count + 1
        deviation_score += abs( output[ i ][ 0 ] - predictions[ i ][ 0 ] )
        deviation_ddg += abs( output[ i ][ 1 ] - predictions[ i ][ 1 ] )
        if ( output[ i ][ 1 ] < -0.1 ) or ( output[ i ][ 1 ] > 0.1 ):
            int_count = int_count + 1
            deviation_score_int += abs( output[ i ][ 0 ] - predictions[ i ][ 0 ] )
            deviation_ddg_int += abs( output[ i ][ 1 ] - predictions[ i ][ 1 ] )


file.close()

average_score = deviation_score / count
average_ddg = deviation_ddg / count
average_score_int = deviation_score_int / int_count
average_ddg_int = deviation_ddg_int / _intcount

print( "RESULTS: " + str(average_score) + " " + str(average_ddg) + " " + str(average_score_int) + " " + str(average_ddg_int) )

