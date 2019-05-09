import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from random import shuffle

import numpy

import sys

import pandas as pd
import gzip

import argparse
import random


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

parser = argparse.ArgumentParser()

parser.add_argument( "--training_data", help="CSV where each line has two elements. First element is the absolute path to the input csv file, second element is the absolute path to the corresponding output csv file.", required=True )
# Example: "--training_data foo.csv" where foo.csv looks like:
# /home/jack/input.1.csv,/home/jack/output.1.csv
# /home/jack/input.2.csv,/home/jack/output.2.csv
# /home/jack/input.3.csv,/home/jack/output.3.csv
# ...

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
        print( str( in_resid ) + " " + str( out_resid ) )
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
# METRICS #
###########

def mean_pred( y_true, y_pred ):
    return K.mean( y_pred )

#########
# START #
#########


with open( args.training_data, "r" ) as f:
    file_lines = f.readlines()

for x in range( 0, 1 ):
    for line in file_lines:
        input, output = generate_data_from_files( line )
