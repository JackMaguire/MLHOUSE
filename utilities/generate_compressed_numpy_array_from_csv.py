import numpy
import pandas as pd
import sys
import gzip

for i in range( 1, len(sys.argv) ):
    contents = pd.read_csv( sys.argv[ i ] ).values
    f = gzip.GzipFile( sys.argv[ i ] + ".npy.gz", "w" )
    numpy.save( f, contents )
    f.close()
