#include <iostream>
#include <math.h>

constexpr double FLOOR = 0.0001;
constexpr double CEILING = 0.01;
constexpr double CEILING2 = 2.0 * CEILING;


float sigmoid ( float val ) {
  return 1.0 / ( 1.0 + exp( -1.0 * val ) ) ;
}

float process( float val ){
  float const transformed_val = -val;
  float const sig = sigmoid( transformed_val );
  return FLOOR + ( CEILING2 * sig );
  //return sig;
}

int main( int argc, char* argv[] ){

  if ( argc != 2 ) {
    std::cerr << "Usage: " << argv[0] << " EPOCH.SUBEPOCH" << std::endl;
    return 1;
  }

  float val = std::atof( argv[ 1 ] );
  if( val >= 10.0 ) val += 10.0;
  // std::cout << val << std::endl;

  while( val >= 20.0 ){
    val -= 20.0;
  }

  std::cout << process( val ) << std::endl;
  /*
  for( double x = 0.0; x <= 50.0; x += 0.1 ){
    val = x;
    if( val >= 10.0 ) val += 10.0;
    while( val >= 20.0 ){
      val -= 20.0;
    }
    std::cout << x << "," << process( val ) << std::endl;
  }
  */
}
