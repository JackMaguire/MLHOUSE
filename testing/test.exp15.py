import jack_mouse_test

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


def denormalize_val( val ):
    #print( "denromalizing ", val, " to ", (math.exp( math.exp( val + 1 ) ) - 10) )
    #return math.exp( math.exp( val + 1 ) ) - 10;
    if val <= -1.0:
        val = -0.999
    try:
        i = math.log( val + 1.0 ) * -15.0
    except:
        print( val )
        #print( e )
        exit( 1 )
    return i

#########
# START #
#########

if os.path.isfile( args.model ):
    model = load_model( args.model )
else:
    print( "Model " + args.model + " is not a file" )
    exit( 1 )

mse_pre_denorm = 0.
mse_post_denorm = 0.
mse_post_denorm_lt_0 = 0.
mse_post_denorm_lt_n2 = 0.
mse_post_denorm_lt_n4 = 0.
mse_post_denorm_lt_n6 = 0.

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
    split = line.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    try:
        cpp_structs = jack_mouse_test.read_mouse_data( split[ 0 ], split[ 1 ] )
        source_input = cpp_structs[ 0 ]
        ray_input = cpp_structs[ 1 ]
        output = cpp_structs[ 2 ]
    except AssertError:
        continue
    predictions = model.predict( x=[source_input,ray_input] )
    my_assert_equals_thrower( "len( predictions )", len( predictions ), len( output ) );
    for i in range( 0, len( predictions ) ):
        '''
        denorm_val=output[ i ][ 0 ]
        norm_val = denorm_val
        if norm_val > 1:
            norm_val = norm_val**0.75
        norm_val += 2.0
        norm_val /= 3.0
        '''
        norm_val=output[ i ][ 0 ]
        denorm_val = denormalize_val( norm_val )

        norm_pred=predictions[ i ][ 0 ]
        #print( "prediction: ", predictions[ i ][ 0 ] )
        denorm_pred = denormalize_val( norm_pred )

        mse_pre_denorm += (norm_val-norm_pred)**2
        denorm_mse = (denorm_val-denorm_pred)**2
        mse_post_denorm += denorm_mse

        #print( denorm_val, denorm_pred )

        allxs.append( denorm_val )
        allys.append( denorm_pred )
        
        if denorm_val < 0.:
            xs_lt_0.append( denorm_val )
            ys_lt_0.append( denorm_pred )
            mse_post_denorm_lt_0 += denorm_mse
            if denorm_val < -2.:
                xs_lt_n2.append( denorm_val )
                ys_lt_n2.append( denorm_pred )
                mse_post_denorm_lt_n2 += denorm_mse
                if denorm_val < -4.:
                    xs_lt_n4.append( denorm_val )
                    ys_lt_n4.append( denorm_pred )
                    mse_post_denorm_lt_n4 += denorm_mse
                    if denorm_val < -6.:
                        xs_lt_n6.append( denorm_val )
                        ys_lt_n6.append( denorm_pred )
                        mse_post_denorm_lt_n6 += denorm_mse
                
mse_pre_denorm /= len(allxs)
mse_post_denorm /= len(allxs)
mse_post_denorm_lt_0 /= len(xs_lt_0)
mse_post_denorm_lt_n2 /= len(xs_lt_n2)
mse_post_denorm_lt_n4 /= len(xs_lt_n4)
mse_post_denorm_lt_n6 /= len(xs_lt_n6)

print( mse_pre_denorm, len(allxs) )
print( mse_post_denorm, len(allxs) )
print( mse_post_denorm_lt_0, len(xs_lt_0) )
print( mse_post_denorm_lt_n2, len(xs_lt_n2) )
print( mse_post_denorm_lt_n4, len(xs_lt_n4)  )
print( mse_post_denorm_lt_n6, len(xs_lt_n6) )
print( " " )
print( scipy.stats.pearsonr( allxs, allys )[ 0 ] )
print( scipy.stats.pearsonr( xs_lt_0, ys_lt_0 )[ 0 ] )
print( scipy.stats.pearsonr( xs_lt_n2, ys_lt_n2 )[ 0 ] )
print( scipy.stats.pearsonr( xs_lt_n4, ys_lt_n4 )[ 0 ] )
print( scipy.stats.pearsonr( xs_lt_n6, ys_lt_n6 )[ 0 ] )
print( " " )
print( scipy.stats.spearmanr( allxs, allys ).correlation )
print( scipy.stats.spearmanr( xs_lt_0, ys_lt_0 ).correlation )
print( scipy.stats.spearmanr( xs_lt_n2, ys_lt_n2 ).correlation )
print( scipy.stats.spearmanr( xs_lt_n4, ys_lt_n4 ).correlation )
print( scipy.stats.spearmanr( xs_lt_n6, ys_lt_n6 ).correlation )
print( " " )
print( scipy.stats.kendalltau( allxs, allys ).correlation )
print( scipy.stats.kendalltau( xs_lt_0, ys_lt_0 ).correlation )
print( scipy.stats.kendalltau( xs_lt_n2, ys_lt_n2 ).correlation )
print( scipy.stats.kendalltau( xs_lt_n4, ys_lt_n4 ).correlation )
print( scipy.stats.kendalltau( xs_lt_n6, ys_lt_n6 ).correlation )

#for x in range( 0, len( allxs ) ):
#    print( allxs[x], allys[x] )
