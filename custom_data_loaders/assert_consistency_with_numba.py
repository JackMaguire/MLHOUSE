import jack_mouse_test

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from random import shuffle

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

#import scipy
#import scipy.stats

#import numba
#from numba import jit

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

#parser.add_argument( "--model", help="Most recent model file", required=True )

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

#@jit
def my_assert_equals( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        exit( 1 )

#@jit
def my_assert_close( name, actual, theoretical ):
    if abs( actual - theoretical ) > 0.001:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        exit( 1 )

#@jit
def my_assert( name, test ):
    if not test:
        print( "Failed: ", name )
        exit( 1 )

#@jit
class AssertError(Exception):
    pass

#@jit
def my_assert_equals_thrower( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        raise AssertError

#@jit
def assert_vecs_line_up( input, output ):
    #Each line starts with "RESID: XXX,"
    my_assert_equals_thrower( "input file length", len( input ), len( output ) )
    for i in range( 0, len( input ) ):
        in_elem = input[ i ][ 0 ]
        in_resid = int( in_elem.split( " " )[ 1 ] )
        out_elem = output[ i ][ 0 ]
        out_resid = int( out_elem.split( " " )[ 1 ] )
        my_assert_equals_thrower( "out_resid", out_resid, in_resid )

#@jit( nopython=True )
def denormalize_val( val ):
    if val <= -0.999:
        val = -0.999
    return math.log( val + 1.0 ) * -15.0

#@jit( nopython=True )
def normalize_val( val ):
    return math.exp( val / -15.0 ) - 1.0;

def generate_data_from_files( filenames_csv ):
    split = filenames_csv.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    split = filenames_csv.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    # Both of these elements lead with a dummy
    if split[ 0 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 0 ], "r" )
        input = pd.read_csv( f, header=None ).values
        f.close()
    elif split[ 0 ].endswith( ".csv" ):
        input = pd.read_csv( split[ 0 ], header=None ).values
    else:
        print ( "We cannot open this file format: " + split[ 0 ] )
        exit( 1 )

    if split[ 1 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 1 ], "r" )
        output = pd.read_csv( f, header=None ).values
        f.close()
    elif split[ 1 ].endswith( ".csv" ):
        output = pd.read_csv( split[ 1 ], header=None ).values
    else:
        print ( "We cannot open this file format: " + split[ 1 ] )
        exit( 1 )

    assert_vecs_line_up( input, output )

    source_input_no_resid = input[:,1:27]
    ray_input_no_resid = input[:,27:]

    output_no_resid = output[:,1:2]

    #my_assert_equals_thrower( "len( source_input_no_resid[ 0 ] )", len( source_input_no_resid[ 0 ] ), num_source_residue_inputs );
    #my_assert_equals_thrower( "len( ray_input_no_resid[ 0 ] )",    len( ray_input_no_resid[ 0 ] ), num_ray_inputs );
    #my_assert_equals_thrower( "len( output_no_resid[ 0 ] )", len( output_no_resid[ 0 ] ), num_output_dimensions );

    for x in range( 0, len( output_no_resid ) ):
        #my_assert_equals_thrower( "len(output_no_resid[x])", len(output_no_resid[x]), 1 )
        output_no_resid[x][0] = normalize_val( output_no_resid[x][0] );
    return source_input_no_resid, ray_input_no_resid, output_no_resid

#########
# START #
#########

# 4) Fit Model

with open( args.data, "r" ) as f:
    file_lines = f.readlines()

for line in file_lines:
    #print( "reading " + str( line ) )
    split = line.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    try:
        t0 = time.time()

        cpp_structs = jack_mouse_test.read_mouse_data( split[ 0 ], split[ 1 ] )
        source_input = cpp_structs[ 0 ]
        ray_input = cpp_structs[ 1 ]
        output = cpp_structs[ 2 ]

        t1 = time.time()
        
        py_source_input, py_ray_input, py_output = generate_data_from_files( line )

        t2 = time.time()

        cpp = t1 - t0
        py = t2 - t1

        print( py / cpp )

        my_assert_equals( "#1", len(source_input), len(py_source_input) );
        my_assert_equals( "#2", len(ray_input), len(py_ray_input) );
        my_assert_equals( "#3", len(output), len(py_output) );

        for x in range( 0, len( source_input[ 0 ] ) ):
            my_assert_close( "#4", source_input[ 0 ][ x ], py_source_input[ 0 ][ x ] )

        for x in range( 0, len( ray_input[ 0 ] ) ):
            my_assert_close( "#5", ray_input[ 0 ][ x ], py_ray_input[ 0 ][ x ] )

        for x in range( 0, len( output[ 0 ] ) ):
            my_assert_close( "#6", output[ 0 ][ x ], py_output[ 0 ][ x ] )
    except AssertError:
        continue

print( "Everything is good" )
