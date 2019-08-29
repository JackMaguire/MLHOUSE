#include <iostream>
#include <math.h>

//constexpr double FLOOR = 0.0005;
//constexpr double CEILING = 0.01;
//constexpr double CEILING2 = 2.0 * CEILING;


float sigmoid ( float val ) {
  return 1.0 / ( 1.0 + exp( -1.0 * val ) ) ;
}

float process(
  float const floor,
  float const ceiling,
  float const val
){
  float const transformed_val = -val;
  float const sig = sigmoid( transformed_val );
  return floor + ( ceiling * 2.0 * sig );
  //return sig;
}

float outer_process(
  float const floor,
  float const ceiling,
  float val
){
  // if( val >= 10.0 ) val += 10.0;
  // std::cout << val << std::endl;

  while( val >= 10.0 ){
    val -= 15.0;
  }

  val += 5.0;

  return process( floor, ceiling, val );
}

int main( int argc, char* argv[] ){

  if ( argc != 4 ) {
    std::cerr << "Usage: " << argv[0] << " FLOOR CEILING EPOCH.SUBEPOCH" << std::endl;
    return 1;
  }

  float const floor = std::atof( argv[ 1 ] );
  float const ceiling = std::atof( argv[ 2 ] );
  float const val = std::atof( argv[ 3 ] );

  //std::cout << outer_process( floor, ceiling, val ) << std::endl;
  for( double x = 0.0; x <= 50.0; x += 0.1 ){
    std::cout << x << "," << outer_process( floor, ceiling, x ) << std::endl;
  }
}
