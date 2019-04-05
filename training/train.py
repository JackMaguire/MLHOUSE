import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

parser.add_argument( "--model", help="Most recent model file", required=True )

parser.add_argument( "--training_data", help="CSV where each line has two elements. First element is the absolute path to the input csv file, second element is the absolute path to the corresponding output csv file.", required=True )
# Example: "--training_data foo.csv" where foo.csv looks like:
# /home/jack/input.1.csv,/home/jack/output.1.csv
# /home/jack/input.2.csv,/home/jack/output.2.csv
# /home/jack/input.3.csv,/home/jack/output.3.csv
# ...

parser.add_argument( "--starting_epochs", help="For bookkeeping purposes, what is the epoch number of the model loaded with --model?", type=int, required=True )
parser.add_argument( "--epoch_checkpoint_frequency", help="How often should we be saving models?", type=int, required=True )
parser.add_argument( "--num_epochs", help="Number of epochs to run. 0 means infinite loop.", type=int, required=True )

#args = parser.parse_args()

#num_neurons_in_first_hidden_layer = args.num_neurons_in_first_hidden_layer
#print( "num_neurons_in_first_hidden_layer: " + str( num_neurons_in_first_hidden_layer ) )

#########
# FUNCS #
#########

def my_assert_equals( name, actual, theoretical ):
    if actual != theoretical:
        print( name + " is equal to " + actual + " instead of " + theoretical )
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
    split = filenames_csv.split( "," );
    my_assert_equals( "split.length", len( split ), 2 );

    # Both of these elements lead with a dummy
    input  = pd.read_csv( split[ 0 ] ).values
    output = pd.read_csv( split[ 1 ] ).values

    assert_vecs_line_up( input, output )
   
    print( input )

    print( input[:,1:] )

    return input, output

generate_data_from_files( "../sample_data/sample.repack.input.csv,../sample_data/sample.repack.output.csv" )
exit( 0 )

###########
# CLASSES #
###########
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

###########
# METRICS #
###########

def mean_pred( y_true, y_pred ):
    return K.mean( y_pred )

#########
# START #
#########

# 1) Define Filenames
#training_input_file = "training.dat"
#testing_input_file = "testing.dat"

#training_input, training_output_hbond = generate_data_from_file( training_input_file )
#testing_input, testing_output_hbond = generate_data_from_file( testing_input_file )

training_input = numpy.load( "training.dat.input.npy" )
training_output_hbond = numpy.load( "training.dat.hbond.npy" )

testing_input = numpy.load( "testing.dat.input.npy" )
testing_output_hbond = numpy.load( "testing.dat.hbond.npy" )

nonnative_training_input = numpy.load( "nonnative.input.npy" )
nonnative_training_hbond = numpy.load( "nonnative.hbond.npy" )

native_testing_input, native_testing_output_hbond = generate_data_from_file( "native.csv" )

weight1 = calc_weight( nonnative_training_hbond, training_output_hbond )
print( "weight: " + str( weight1 ) )

# 2) Define Model

if os.path.isfile( "best.h5" ):
    model = load_model( "best.h5" )
else:
    num_input_dimensions = 9
    model = Sequential()

    model.add( Dense( num_neurons_in_first_hidden_layer, input_dim=num_input_dimensions, activation='relu') )

    for x in range( 0, num_intermediate_hidden_layers ):
        model.add( Dense( num_neurons_in_intermediate_hidden_layer, activation='relu') )

    num_neurons_in_final_layer = int( 1 )
    model.add( Dense( num_neurons_in_final_layer, activation='sigmoid') )

    # 3) Compile Model

    metrics_to_output=[ 'accuracy' ]
    model.compile( loss='binary_crossentropy', optimizer='adam', metrics=metrics_to_output )

# 4) Fit Model
best_score_so_far = 0
test_ppv, test_npv = evaluate_model( model, best_score_so_far, testing_input, testing_output_hbond, 0 )
native_ppv, native_npv = evaluate_model( model, best_score_so_far, native_testing_input, native_testing_output_hbond, 0 )
best_score_so_far = score( test_ppv, test_npv, native_ppv, native_npv )
print( "0 " + str( test_ppv ) + " " +  str( test_npv ) + " " +  str( native_ppv ) + " " +  str( native_npv ) )

for x in range( 0, num_epochs ):
    #start = time.time()
    #print( "Beginning epoch: " + str(x) )
    
    shuffle_in_unison( training_input, training_output_hbond )
    i=0
    while i < len(training_input):
        j = len(training_input) - i
        if j >  100000:
            j = 100000
        i +=    100000

        model.train_on_batch( x=training_input[ i : i+j ], y=training_output_hbond[ i : i+j ], class_weight={ 0 : 1, 1 : weight1 } )

    shuffle_in_unison( nonnative_training_input, nonnative_training_hbond )
    i=0
    while i < len(nonnative_training_input):
        j = len(nonnative_training_input) - i
        if j >  10000:
            j = 10000
        i +=    10000

        model.train_on_batch( x=nonnative_training_input[ i : i+j ], y=nonnative_training_hbond[ i : i+j ], class_weight={ 0 : 1, 1 : weight1 } )

    if ( x % 10 == 0 ):
        model.save( "epoch_" + str(x) + ".h5" )
        test_ppv, test_npv = evaluate_model( model, best_score_so_far, testing_input, testing_output_hbond, 0 )
        native_ppv, native_npv = evaluate_model( model, best_score_so_far, native_testing_input, native_testing_output_hbond, 0 )
        best_score_so_far = score( test_ppv, test_npv, native_ppv, native_npv )
        print( str( x ) + " " + str( test_ppv ) + " " +  str( test_npv ) + " " +  str( native_ppv ) + " " +  str( native_npv ) + " " + str( best_score_so_far ) )

    #end = time.time()
    #print( "\tseconds: " + str( end - start ) )
    sys.stdout.flush()

model.save( "final.h5" )
