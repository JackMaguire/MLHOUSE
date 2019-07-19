import jack_mouse_test

from random import shuffle

import numpy
import numpy as np

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
parser = argparse.ArgumentParser()

parser.add_argument( "--csv", help="csv file where first column is target value and second column is predicted value", required=True )

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

allxs = []
allys = []

xs_lt_p10 = []
ys_lt_p10 = []

xs_lt_p25 = []
ys_lt_p25 = []

xs_lt_p50 = []
ys_lt_p50 = []

xs_lt_p75 = []
ys_lt_p75 = []


# 4) Fit Model

with open( args.csv, "r" ) as f:
    file_lines = f.readlines()

all_vals = []

for line in file_lines:
    #print( "reading " + str( line ) )
    split = line.split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    val = float( split[ 0 ] )
    pred = float( split[ 1 ] )
    all_vals.append( val )

#Higher is worse
#print( np.percentile(all_vals, 50, axis=0) ); #-422
#print( np.percentile(all_vals, 90, axis=0) ); #15
#print( np.percentile(all_vals, 99, axis=0) ); #700

p10 = np.percentile(all_vals, 10, axis=0)
p25 = np.percentile(all_vals, 25, axis=0)
p50 = np.percentile(all_vals, 50, axis=0)
p75 = np.percentile(all_vals, 75, axis=0)

for line in file_lines:
    #print( "reading " + str( line ) )
    split = line.split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    val = float( split[ 0 ] )
    pred = float( split[ 1 ] )

    allxs.append( val )
    allys.append( pred )

    if val < p75:
        xs_lt_p75.append( val )
        ys_lt_p75.append( pred )
        if val < p50:
            xs_lt_p50.append( val )
            ys_lt_p50.append( pred )
            if val < p25:
                xs_lt_p25.append( val )
                ys_lt_p25.append( pred )
                if val < p10:
                    xs_lt_p10.append( val )
                    ys_lt_p10.append( pred )

print( scipy.stats.pearsonr( allxs, allys )[ 0 ] )
print( scipy.stats.pearsonr( xs_lt_p75, ys_lt_p75 )[ 0 ] )
print( scipy.stats.pearsonr( xs_lt_p50, ys_lt_p50 )[ 0 ] )
print( scipy.stats.pearsonr( xs_lt_p25, ys_lt_p25 )[ 0 ] )
print( scipy.stats.pearsonr( xs_lt_p10, ys_lt_p10 )[ 0 ] )
print( " " )
print( scipy.stats.spearmanr( allxs, allys ).correlation )
print( scipy.stats.spearmanr( xs_lt_p75, ys_lt_p75 ).correlation )
print( scipy.stats.spearmanr( xs_lt_p50, ys_lt_p50 ).correlation )
print( scipy.stats.spearmanr( xs_lt_p25, ys_lt_p25 ).correlation )
print( scipy.stats.spearmanr( xs_lt_p10, ys_lt_p10 ).correlation )
print( " " )
print( scipy.stats.kendalltau( allxs, allys ).correlation )
print( scipy.stats.kendalltau( xs_lt_p75, ys_lt_p75 ).correlation )
print( scipy.stats.kendalltau( xs_lt_p50, ys_lt_p50 ).correlation )
print( scipy.stats.kendalltau( xs_lt_p25, ys_lt_p25 ).correlation )
print( scipy.stats.kendalltau( xs_lt_p10, ys_lt_p10 ).correlation )
print( " " )
print( len( allxs ) )
print( len( xs_lt_p75 ) )
print( len( xs_lt_p50 ) )
print( len( xs_lt_p25 ) )
print( len( xs_lt_p10 ) )
