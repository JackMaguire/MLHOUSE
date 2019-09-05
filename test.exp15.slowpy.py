import numpy
import pandas as pd
import gzip
import math
import argparse
import random
import time

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

def generate_data_from_files_control( filenames_csv, six_bin ):
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

    #assert_vecs_line_up( input, output )

    source_input_no_resid = input[:,1:27]
    ray_input_no_resid = input[:,27:]
    output_no_resid = output[:,1:2]

    my_assert_equals_thrower( "len( source_input_no_resid[ 0 ] )", len( source_input_no_resid[ 0 ] ), num_source_residue_inputs );
    my_assert_equals_thrower( "len( ray_input_no_resid[ 0 ] )",    len( ray_input_no_resid[ 0 ] ), num_ray_inputs );
    my_assert_equals_thrower( "len( output_no_resid[ 0 ] )", len( output_no_resid[ 0 ] ), num_output_dimensions );

    for x in range( 0, len( output_no_resid ) ):
        val = output_no_resid[x][0]
        output_no_resid[x][0] = math.exp( val / -15.0 ) - 1.0
    return source_input_no_resid, ray_input_no_resid, output_no_resid

def generate_data_from_files_fast( filenames_csv, six_bin ):
    #dataset = numpy.genfromtxt( filename, delimiter=",", skip_header=0 )
    split = filenames_csv.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    #t0 = time.time()

    # Both of these elements lead with a dummy
    #input = pd.read_csv( split[ 0 ], header=None, dtype=numpy.float32 ).values
    input = numpy.loadtxt( split[ 0 ], delimiter=',', skiprows=0, usecols=range(1,num_source_residue_inputs+num_ray_inputs+1), dtype=numpy.float32)

    #output = pd.read_csv( split[ 1 ], header=None, dtype=numpy.float32 ).values
    output = numpy.loadtxt( split[ 1 ], delimiter=',', skiprows=0, usecols=(1,2), dtype=numpy.float32)

    #assert_vecs_line_up( input, output )

    source_input_no_resid = input[:,0:26]
    ray_input_no_resid = input[:,26:]
    output = output[:,0:1]

    my_assert_equals_thrower( "len( source_input_no_resid[ 0 ] )", len( source_input_no_resid[ 0 ] ), num_source_residue_inputs );
    my_assert_equals_thrower( "len( ray_input_no_resid[ 0 ] )",    len( ray_input_no_resid[ 0 ] ), num_ray_inputs );
    my_assert_equals_thrower( "len( output[ 0 ] )", len( output[ 0 ] ), num_output_dimensions );

    for x in range( 0, len( output ) ):
        val = output[x][0]
        output[x][0] = math.exp( val / -15.0 ) - 1.0
    return source_input_no_resid, ray_input_no_resid, output


#########
# START #
#########

with open( args.data, "r" ) as f:
    file_lines = f.readlines()

t0 = time.time()

for line in file_lines:
    #print( "reading " + str( line ) )
    split = line.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    try:
        source_input1, ray_input1, output1 = generate_data_from_files_fast( line, False )
    except AssertError:
        continue
 
t1 = time.time()
print( "time:", (t1-t0) )


t0 = time.time()

for line in file_lines:
    #print( "reading " + str( line ) )
    split = line.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    try:
        source_input2, ray_input2, output2 = generate_data_from_files_control( line, False )
    except AssertError:
        continue
 
t1 = time.time()
print( "time:", (t1-t0) )

for x in range(0,len(source_input1)):
    for y in range( 0, len( source_input1[x] ) ):
        if abs( source_input1[x][y] - source_input2[x][y] > 0.001 ):
            print( "source_input is bad" )

for x in range(0,len(ray_input1)):
    for y in range( 0, len( ray_input1[x] ) ):
        if abs( ray_input1[x][y] - ray_input2[x][y] > 0.001 ):
            print( "ray_input is bad" )

for x in range(0,len(output1)):
    for y in range( 0, len( output1[x] ) ):
        if abs( output1[x][y] - output2[x][y] > 0.001 ):
            print( "output is bad" )


#original: 116.92752194404602
