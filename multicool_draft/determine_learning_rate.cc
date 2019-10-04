#include <iostream>
#include <math.h>

//constexpr double FLOOR = 0.0005;
//constexpr double CEILING = 0.002;
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
  return floor + ( (ceiling-floor) * 2.0 * sig );
  //return sig;
}

float outer_process(
  float const floor,
  float const ceiling,
  float span,
  float val
){
  if( val > 75 ) return 0.01 / val;
  // if( val >= 10.0 ) val += 10.0;
  // std::cout << val << std::endl;

  while( val >= (span/2.0) ){
    val -= span;
    span += 2;
  }

  val += (span/2.0);

  return process( floor, ceiling, val );
}

//Try 8, 12, 16, 20
int main( int argc, char* argv[] ){

  if ( argc != 5 ) {
    std::cerr << "Usage: " << argv[0] << " FLOOR CEILING STARTSPAN EPOCH.SUBEPOCH" << std::endl;
    return 1;
  }

  float const floor = std::atof( argv[ 1 ] );
  float const ceiling = std::atof( argv[ 2 ] );
  float const span = std::atof( argv[ 3 ] );
  float const val = std::atof( argv[ 4 ] );

  std::cout << outer_process( floor, ceiling, span, val ) << std::endl;
  //for( double x = 0.0; x <= 100.0; x += 0.1 ){
  // std::cout << x << "," << outer_process( floor, ceiling, span, x ) << std::endl;
  //}
}
