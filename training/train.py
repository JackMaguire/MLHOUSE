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

exit( 0 )

#############
# CONSTANTS #
#############

#0                         1                          2  3  4  5  6  7  8      9      10
#best_possible_hbond_score,worst_possible_clash_score,tx,ty,tz,rz,ry,rz,angle1,angle2,dist

BEST_POSSIBLE_HBOND_SCORE  = int( 0 )
WORST_POSSIBLE_CLASH_SCORE = int( 1 )

TX = int( 2 )
TY = int( 3 )
TZ = int( 4 )

RX = int( 5 )
RY = int( 6 )
RZ = int( 7 )

ANGLE1 = int( 8 )
ANGLE2 = int( 9 )
DIST   = int( 10 )

#########################
# COMMAND LINE SETTINGS #
#########################

parser = argparse.ArgumentParser()

parser.add_argument( "--num_neurons_in_first_hidden_layer", help="Number of neruons for first hidden layer.", default="100", type=int, required=False )
parser.add_argument( "--num_neurons_in_intermediate_hidden_layer", help="Number of neruons for intermediate hidden layer.", default="100", type=int, required=False )
parser.add_argument( "--num_intermediate_hidden_layers", help="Number of intermediate hidden layers.", default="4", type=int, required=False )

parser.add_argument( "--num_epochs", help="Number of epochs to give to model.fit()", default="150", type=int, required=False )

parser.add_argument( "--test_predictions", help="filename for test predictions", default="", required=False )

#parser.add_argument( "--weight", help="Class weight for 1", default="3.0", required=True )

args = parser.parse_args()

num_neurons_in_first_hidden_layer = args.num_neurons_in_first_hidden_layer
print( "num_neurons_in_first_hidden_layer: " + str( num_neurons_in_first_hidden_layer ) )

num_neurons_in_intermediate_hidden_layer = args.num_neurons_in_intermediate_hidden_layer
print( "num_neurons_in_intermediate_hidden_layer: " + str( num_neurons_in_intermediate_hidden_layer ) )

num_intermediate_hidden_layers = args.num_intermediate_hidden_layers
print( "num_intermediate_hidden_layers: " + str( num_intermediate_hidden_layers ) )

num_epochs = args.num_epochs #150 is small
print( "num_epochs: " + str( num_epochs ) )

#weight1 = args.weight
#print( "class weight for 1: " + str( weight1 ) );

#########
# FUNCS #
#########

def my_assert_equals( name, actual, theoretical ):
    if actual != theoretical:
        print( name + " is equal to " + actual + " instead of " + theoretical )
        exit( 1 )

def keep_hbond_score( score ):
    hbond_score = score[ BEST_POSSIBLE_HBOND_SCORE ]
    print( score )
    print( hbond_score )
    exit( 0 )
    if score == 0:
        return true
    if score <= -0.5:
        return true
    return false

def calc_weight( nonnative_training_hbond, training_output_hbond ):
    num_pos = int( 0 )
    num_neg = int( 0 )
    for x in nonnative_training_hbond:
        for i in range( 0, len(x) ):
            if x[i] != 0:
                num_pos += 1
            else:
                num_neg += 1
    for x in training_output_hbond:
        for i in range( 0, len(x) ):
            if x[i] != 0:
                num_pos += 1
            else:
                num_neg += 1

    return float( num_neg ) / float( num_pos )

def normalize_single_input( input ):
    input[0] /= 20. #Tx
    input[1] /= 20. #Ty
    input[2] /= 20. #Tz
        
    input[3] /= 3.14 #Rx
    input[4] /= 3.14 #Ry
    input[5] -= 1.6  #Rz

    input[6] -= 1.6 #Theta1
    input[7] -= 1.6 #Theta2
    input[8] = (input[8]/15.) - 1 #D

def generate_data_from_file( filename ):
    dataset = numpy.genfromtxt( filename, delimiter=",", skip_header=0 )

    input = dataset[:,[ TX, TY, TZ, RX, RY, RZ, ANGLE1, ANGLE2, DIST ] ]
    output_hbond = dataset[:,[ BEST_POSSIBLE_HBOND_SCORE  ] ]

    for x in output_hbond:
        for i in range( 0, len(x) ):
            if x[i] > 0:
                print( "Some hbond value is positive! " + str(x[i]) )
                exit( 1 )
            if x[i] != 0:
                x[i] = 1
     
    for x in input:
        normalize_single_input( x )
   
    return input, output_hbond

def evaluate_model( model, best_score_so_far, test_input, test_output_hbond, batch ):
    num_positives_actual = 0.
    num_positives_predicted = 0.
    num_positives_actual_and_predicted = 0.

    num_negatives_actual = 0.
    num_negatives_predicted = 0.
    num_negatives_actual_and_predicted = 0.
        
    predictions = model.predict( x=test_input );

    for i in range( 0, len(test_input) ):

        actual = test_output_hbond[ i ][ 0 ]
        prediction = predictions[ i ][ 0 ]

        if actual == 0:
            num_negatives_actual += 1
            if prediction < 0.5:
                num_negatives_predicted += 1
                num_negatives_actual_and_predicted += 1
            else:
                num_positives_predicted += 1
        else:
            num_positives_actual += 1
            if prediction < 0.5:
                num_negatives_predicted += 1
            else:
                num_positives_actual_and_predicted += 1
                num_positives_predicted += 1

    #min = 1;
    ppv = num_positives_actual_and_predicted/num_positives_actual
    npv = num_negatives_actual_and_predicted/num_negatives_actual
    return ppv, npv
'''
    if score1 < min:
        min = score1
    if score2 < min:
        min = score2

    saved = 0
'''
    #if min >= best_score_so_far:
        #best_score_so_far = min
        #model.save( "best.h5" )
        #saved = 1

    #print( str(batch) + " " + str(score1) + " " + str(score2) + " " + str(saved) )

    #return best_score_so_far

#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

def score( ppv1, npv1, ppv2, npv2 ):
    min = 1
    if( ppv1 < min ):
        min = ppv1
    if( npv1 < min ):
        min = npv1
    if( ppv2 < min ):
        min = ppv2
    if( npv2 < min ):
        min = npv2
    return min

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
