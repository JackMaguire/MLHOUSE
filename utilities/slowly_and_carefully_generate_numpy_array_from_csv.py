import numpy
import pandas as pd
import sys
import math
import gzip


for i in range( 1, len(sys.argv) ):
    if ( sys.argv[ i ].endswith( ".gz" ) ):
        f = gzip.GzipFile( sys.argv[ i ], "r" )
        contents = pd.read_csv( f ).values
        f.close()
    else:
        contents = pd.read_csv( sys.argv[ i ] ).values

    print( contents )
    numpy.save( sys.argv[ i ] + ".npy", contents )
    loaded = numpy.load( sys.argv[ i ] + ".npy" )
    if len( contents ) != len( loaded ):
        print ( "len( contents ) != len( loaded )" )
        exit( 1 )

    for x in range( 0, len( contents ) ):
        if len( contents[x] ) != len( loaded[x] ):
            print ( "len( contents[x] ) != len( loaded[x] )" )
            exit( 1 )

        for y in range( 0, len( contents[ x ] ) ):
            if contents[x][y] != loaded[x][y]:
                if ( not math.isnan( contents[x][y] ) ) or ( not math.isnan( loaded[x][y] ) ):
                    print ( str( contents[x][y] ) + " != " + str( loaded[x][y] ) )
                    exit( 1 )


exit( 0 )
        
