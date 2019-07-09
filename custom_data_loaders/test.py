import jack_mouse_test

import numpy
import pandas as pd
import gzip
import math
import time

############
# SETTINGS #
############

input_file_path= "test.input.csv"
output_file_path="test.output.csv"

#############
# CONSTANTS #
#############

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

##########

def my_assert_equals( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        exit( 1 )

def my_assert_equals_close( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        exit( 1 )


class AssertError(Exception):
    pass

def my_assert_equals_thrower( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        raise AssertError

def my_assert_equals_close_thrower( name, actual, theoretical ):
    if abs(actual - theoretical) > 0.01:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        raise AssertError


def assert_vecs_line_up( input, output ):
    #Each line starts with "RESID: XXX,"
    my_assert_equals_thrower( "input file length", len( input ), len( output ) )
    for i in range( 0, len( input ) ):
        in_elem = input[ i ][ 0 ]
        in_resid = int( in_elem.split( " " )[ 1 ] )
        out_elem = output[ i ][ 0 ]
        out_resid = int( out_elem.split( " " )[ 1 ] )
        my_assert_equals_thrower( "out_resid", out_resid, in_resid )

def generate_data_from_files( input_filename, output_filename, six_bin ):

    # Both of these elements lead with a dummy
    if input_filename.endswith( ".csv.gz" ):
        f = gzip.GzipFile( input_filename, "r" )
        input = pd.read_csv( f, header=None ).values
        f.close()
    elif input_filename.endswith( ".csv" ):
        input = pd.read_csv( input_filename, header=None ).values
    elif input_filename.endswith( ".npy.gz" ):
        f = gzip.GzipFile( input_filename, "r" )
        input = numpy.load( f, allow_pickle=True )
        f.close()
    elif input_filename.endswith( ".npy" ):
        input = numpy.load( input_filename, allow_pickle=True )
    else:
        print ( "We cannot open this file format: " + input_filename )
        exit( 1 )

    if output_filename.endswith( ".csv.gz" ):
        f = gzip.GzipFile( output_filename, "r" )
        output = pd.read_csv( f, header=None ).values
        f.close()
    elif output_filename.endswith( ".csv" ):
        output = pd.read_csv( output_filename, header=None ).values
    elif output_filename.endswith( ".npy.gz" ):
        f = gzip.GzipFile( output_filename, "r" )
        output = numpy.load( f, allow_pickle=True )
        f.close()
    elif output_filename.endswith( ".npy" ):
        output = numpy.load( output_filename, allow_pickle=True )
    else:
        print ( "We cannot open this file format: " + output_filename )
        exit( 1 )

    assert_vecs_line_up( input, output )

    source_input_no_resid = input[:,1:27]
    ray_input_no_resid = input[:,27:]

    output_no_resid = output[:,1:2]

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
                val = math.sqrt( val )
            #subtract mean of -2:
            val += 2.0
            #divide by span of 3:
            val /= 3.0
        return source_input_no_resid, ray_input_no_resid, output_no_resid



t0 = time.time()
cpp_structs = jack_mouse_test.read_mouse_data( output_file_path )
cpp_output = cpp_structs[ 0 ]
t1 = time.time()
py_source_input, py_ray_input, py_output = generate_data_from_files( input_file_path, output_file_path, False )
t2 = time.time()
#print( cpp_output.shape )
#print( py_output.shape )
#exit( 0 )

print( "CPP time: " + str( t1 - t0 ) )
print( "PY  time: " + str( t2 - t1 ) )
print( "Ratio: " + str( (t2 - t1) / (t1 - t0) ) )

my_assert_equals_thrower( "TEST 1", len( cpp_output ), len( py_output ) )
for x in range( 0, len( cpp_output ) ):
    my_assert_equals_close_thrower( "TEST 1." + str( x ), cpp_output[ x ], py_output[ x ] )
