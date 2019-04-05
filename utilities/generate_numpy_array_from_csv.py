import numpy
import pandas as pd
import sys

for i in range( 1, len(sys.argv) ):
    contents = pd.read_csv( sys.argv[ i ] ).values
    numpy.save( sys.argv[ i ] + ".npy", contents )
