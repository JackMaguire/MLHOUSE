import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import tensorflow as tf

#from tensorflow.keras import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks
import tensorflow.keras
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

########
# INIT #
########

num_input_dimensions = 9600
num_output_dimensions = 2

numpy.random.seed( 0 )

parser = argparse.ArgumentParser()

parser.add_argument( "--model", help="Most recent model file", required=True )

#parser.add_argument( "--evaluate_individually", help="Print predictions vs actual data", type=bool, default=False )

parser.add_argument( "--testing_data", help="CSV where each line has two elements. First element is the absolute path to the input csv file, second element is the absolute path to the corresponding output csv file.", required=True )
# Example: "--testing_data foo.csv" where foo.csv looks like:
# /home/jack/input.1.csv,/home/jack/output.1.csv
# /home/jack/input.2.csv,/home/jack/output.2.csv
# /home/jack/input.3.csv,/home/jack/output.3.csv
# ...

#parser.add_argument( "--six_bin", help="Apply six bin transformation to output", type=bool, required=True )


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

def generate_data_from_files( filenames_csv, six_bin ):
    #dataset = numpy.genfromtxt( filename, delimiter=",", skip_header=0 )
    split = filenames_csv.split( "\n" )[ 0 ].split( "," );
    my_assert_equals( "split.length", len( split ), 2 );

    t0 = time.time()

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

    input_no_resid = input[:,25:]
    test = input[:,1:25]

    print( input_no_resid.shape )
    print( test.shape )

    output_no_resid = output[:,1:]

    #my_assert_equals( "len( input_no_resid[ 0 ] )", len( input_no_resid[ 0 ] ), num_input_dimensions );

    #https://www.kaggle.com/vishwasgpai/guide-for-creating-cnn-model-using-csv-file

    if six_bin:
        six_bin_output_no_resid = output_no_resid.copy()
        new_shape = ( output_no_resid.shape[ 0 ], num_output_dimensions )
        six_bin_output_no_resid.resize( new_shape )
        for x in range( 0, len( output_no_resid ) ):
            my_assert_equals( "len(six_bin_output_no_resid[ x ])", len( six_bin_output_no_resid[ x ] ), num_output_dimensions )
            six_bin_output_no_resid[ x ][ 0 ] = 1 if output_no_resid[ x ][ 0 ] <= -5.0 else 0
            six_bin_output_no_resid[ x ][ 1 ] = 1 if output_no_resid[ x ][ 0 ] <= -3.0 else 0
            six_bin_output_no_resid[ x ][ 2 ] = 1 if output_no_resid[ x ][ 0 ] <= -1.0 else 0
            six_bin_output_no_resid[ x ][ 3 ] = 1 if output_no_resid[ x ][ 0 ] >= 1.0 else 0
            six_bin_output_no_resid[ x ][ 4 ] = 1 if output_no_resid[ x ][ 0 ] >= 3.0 else 0
            six_bin_output_no_resid[ x ][ 5 ] = 1 if output_no_resid[ x ][ 0 ] >= 5.0 else 0
            six_bin_output_no_resid[ x ][ 6 ] = 1 if output_no_resid[ x ][ 1 ] <= -5.0 else 0
            six_bin_output_no_resid[ x ][ 7 ] = 1 if output_no_resid[ x ][ 1 ] <= -3.0 else 0
            six_bin_output_no_resid[ x ][ 8 ] = 1 if output_no_resid[ x ][ 1 ] <= -1.0 else 0
            six_bin_output_no_resid[ x ][ 9 ] = 1 if output_no_resid[ x ][ 1 ] >= 1.0 else 0
            six_bin_output_no_resid[ x ][ 10] = 1 if output_no_resid[ x ][ 1 ] >= 3.0 else 0
            six_bin_output_no_resid[ x ][ 11] = 1 if output_no_resid[ x ][ 1 ] >= 5.0 else 0
        return input_no_resid, six_bin_output_no_resid
    else:
        my_assert_equals( "len( output_no_resid[ 0 ] )", len( output_no_resid[ 0 ] ), num_output_dimensions );
        return input_no_resid, output_no_resid

def custom_loss():
    def loss( y_true, y_pred ):
        #using notation from https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html where
        #p: predicted probability ( 0 to 1 )
        #y: actual probability ( 0 to 1 )
        #y_pred: predicted probability (-1 to 1)
        p=(y_pred + 1)/2
        if y_true == 1:
            return -1*K.log( p )
        else: # y_true == 0
            return -1*K.log( 1 - p )

    return loss

def loss( y_true, y_pred ):
    #using notation from https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html where
    #p: predicted probability ( 0 to 1 )
    #y: actual probability ( 0 to 1 )
    #y_pred: predicted probability (-1 to 1)
    p=(y_pred + 1)/2
    if y_true == 1:
        return -1*K.log( p )
    else: # y_true == 0
        return -1*K.log( 1 - p )


tensorflow.keras.losses.custom_loss = custom_loss

#########
# START #
#########


if os.path.isfile( args.model ):
    model = load_model( args.model, custom_objects={'loss': loss} )
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

loop_count = 0

file = open( args.testing_data, "r" )

for line in file:
    try:
        input, output = generate_data_from_files( line, False )
    except pd.errors.EmptyDataError:
        print( "pd.errors.EmptyDataError" )
        continue
    except OSError as e:
        print ( e.errno )
        print ( e )
        continue

    #input_pixel_data = input[14:]

    print( input.shape )
    #print( input_pixel_data.shape )
    #exit( 0 )

    converted = tf.convert_to_tensor(input[:], dtype=tf.float32)
    predictions = model.predict( x=converted )

    #predictions = model.predict( x=input[:] )
    print( predictions )
    print( predictions.shape )
    pred = predictions[ 0 ]
    for x in range( 0, len( pred ) ):
        predx = pred[ x ]
        for y in range( 0, len(predx) ):
            predxy = predx[ y ]
            index = 6
            if predxy[ index ] > 0.5:
                sys.stdout.write(' ')
            elif predxy[ index ] > -0.5:
                sys.stdout.write('.')
            else:
                sys.stdout.write('X')
        sys.stdout.write('\n')
    sys.stdout.flush()

file.close()

#average_score = deviation_score / count
#average_ddg = deviation_ddg / count
#average_score_int = deviation_score_int / int_count
#average_ddg_int = deviation_ddg_int / int_count

#print( "RESULTS: " + str(average_score) + " " + str(average_ddg) + " " + str(average_score_int) + " " + str(average_ddg_int) + " " + str(count) + " " + str(int_count) )