#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <cmath>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>

#include <iostream>

/*
  https://blog.esciencecenter.nl/irregular-data-in-pandas-using-c-88ce311cb9ef  
*/

using namespace boost;
using namespace boost::python;
using namespace boost::python::numpy;

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace mouse_io {

namespace {

/*
float
normalize_output_value(
  float value
){
  //Center
  value += 2.0;

  //Clip
  if( value > 10.0 ){
    value = 10.0;
  }
  else if( value < -10.0 ){
    value = -10.0;
  }

  value /= 10.0;

  return value;
}
*/

float
normalize_output_value(
  float value
){
  value += 10.0;

  //Clip
  if( value < 1.0 ){
    value = 1.0;
  }

  value = log( log ( value ) ) - 1;

  return value;
}


ndarray
generate_output_data(
  std::vector< std::array< std::string, 2 > > const & tokenized_file_lines_of_output_file
){
  int const total_number_of_elements = tokenized_file_lines_of_output_file.size();

  //https://www.boost.org/doc/libs/1_64_0/libs/python/doc/html/numpy/tutorial/simple.html
  p::tuple shape = p::make_tuple( total_number_of_elements, 1 );
  np::dtype dtype = np::dtype::get_builtin<float>();
  np::ndarray output_values = np::empty( shape, dtype );
  float * ndarray_data = reinterpret_cast< float * > ( output_values.get_data() );

  for( int i = 0; i < total_number_of_elements; ++i ){
    ndarray_data[ i ] = normalize_output_value( std::stof( tokenized_file_lines_of_output_file[ i ][ 1 ] ) );
  }

  return output_values;
}

std::vector< std::array< std::string, 2 > >
read_in_output_data(
  std::string const & filename
) {
  //https://stackoverflow.com/questions/7868936/read-file-line-by-line-using-ifstream-in-c
  //https://stackoverflow.com/questions/11719538/how-to-use-stringstream-to-separate-comma-separated-strings
  std::vector< std::array< std::string, 2 > > tokenized_file_lines_of_output_file;
  std::ifstream infile( filename );
  std::string line;

  while( std::getline( infile, line ) ) {
    std::array< std::string, 2 > new_elements;
    // [0]: resid tag
    // [1]: energy
    // [2]: ddg (omitted for now)
    
    std::stringstream ss( line );
    if ( ! std::getline( ss, new_elements[ 0 ], ',' ) ){
      std::cout << "Error at [0]" << std::endl;
      throw int( 0 );
      //continue;//for now, just omit this line
    }

    if ( ! std::getline( ss, new_elements[ 1 ], ',' ) ){
      std::cout << "Error at [1]" << std::endl;
      throw int( 1 );
      //continue;//for now, just omit this line
    }

    //TODO check for third element?

    tokenized_file_lines_of_output_file.emplace_back( new_elements );
  }

  // infile.close(); // Let RAII handle this

  return tokenized_file_lines_of_output_file;
}

struct InputElements{
  std::vector< std::string > resids;
  std::vector< std::array< float, 26 > > residue_data;
  std::vector< std::array< float, 18468 > > ray_data;

  unsigned int next_index(){
    resids.emplace_back();
    residue_data.emplace_back();
    ray_data.emplace_back();

    return resids.size() - 1;
  }
};

ndarray
generate_residue_data(
  std::vector< std::array< float, 26 > > const & residue_data
){
  int const total_number_of_elements = residue_data.size();

  //https://www.boost.org/doc/libs/1_64_0/libs/python/doc/html/numpy/tutorial/simple.html
  p::tuple shape = p::make_tuple( total_number_of_elements, 26 );
  np::dtype dtype = np::dtype::get_builtin<float>();
  np::ndarray residue_values = np::empty( shape, dtype );
  
  float * ndarray_data = reinterpret_cast< float * > ( residue_values.get_data() );

  //Let's see how well this gets optimized
  for( uint line = 0; line < residue_data.size(); ++line ){
    for( int i = 0; i < 26; ++i ){
      ndarray_data[ line * 26 + i ] = residue_data[ line ][ i ];
    }
  }

  return residue_values;
}


ndarray
generate_ray_data(
  std::vector< std::array< float, 18468 > > const & ray_data
){
  //https://www.boost.org/doc/libs/1_64_0/libs/python/doc/html/numpy/tutorial/simple.html
  p::tuple shape = p::make_tuple( ray_data.size(), 18468 );
  np::dtype dtype = np::dtype::get_builtin<float>();
  np::ndarray ray_values = np::empty( shape, dtype );
  float * ndarray_data = reinterpret_cast< float * > ( ray_values.get_data() );

  //Let's see how well this gets optimized
  for( uint line = 0; line < ray_data.size(); ++line ){
    for( int ray = 0; ray < 18468; ++ray ){
      ndarray_data[ line * 18468 + ray ] = ray_data[ line ][ ray ];
    }
  }

  return ray_values;
}

InputElements
read_in_input_data(
  std::string const & filename
) {
  InputElements elements;

  std::ifstream infile( filename );
  std::string line;

  while( std::getline( infile, line ) ) {

    std::array< std::string, 18495 > tokens;
    //Caching strings here
    //We don't want to add the values to "elements"
    //directly just in case something breaks the loop

    std::string dummy;
    std::stringstream ss( line );
    //bool bad = false;
    for( unsigned int i = 0; i < 18495; ++i ){
      if ( ! std::getline( ss, dummy, ',' ) ){
	std::cout << "Error at [0]" << std::endl;
	throw int( 2 );
	//bad=true;
	//break;//for now, just omit this line
      }

      tokens[ i ] = dummy;
    }

    //if( bad ) continue;

    unsigned int const index = elements.next_index();
    elements.resids[ index ] = tokens[ 0 ];

    for( unsigned int i = 0; i < 26; ++i ){
      elements.residue_data[ index ][ i ] = std::stof( tokens[ i + 1 ] );
    }

    for( unsigned int i = 0; i < 18468; ++i ){
      elements.ray_data[ index ][ i ] = std::stof( tokens[ i + 27 ] );
    }

  }

  // infile.close(); // Let RAII handle this

  return elements;
}

void
assert_resids_line_up(
  std::vector< std::string > input_resids,
  std::vector< std::array< std::string, 2 > > const & tokenized_file_lines_of_output_file
) {
  if( input_resids.size() != tokenized_file_lines_of_output_file.size() ){
    throw int( 4 );
  }

  for( unsigned i = 0; i < input_resids.size(); ++i ){
    if( input_resids[ i ] != tokenized_file_lines_of_output_file[ i ][ 0 ] ){
      throw int( 1000 + i );
    }
  }
}

} // anonymous namespace

boost::python::tuple
read_mouse_data(
  std::string const & input_data_filename,
  std::string const & output_data_filename
) {
  try {
    InputElements const input_elements = read_in_input_data( input_data_filename );
    auto const residue_data = generate_residue_data( input_elements.residue_data );
    auto const ray_data = generate_ray_data( input_elements.ray_data );

    auto const tokenized_file_lines_of_output_file = read_in_output_data( output_data_filename );
    auto const output_data = generate_output_data( tokenized_file_lines_of_output_file );

    assert_resids_line_up( input_elements.resids, tokenized_file_lines_of_output_file );

    return boost::python::make_tuple( residue_data, ray_data, output_data );
  } catch ( int error_no ){
    std::cerr << "Caught exception #" << error_no << std::endl;
    return boost::python::make_tuple( error_no );
  } catch ( std::exception const & e ) {
    std::cout << "Caught C++ Exception: " << e.what() << std::endl;
    return boost::python::make_tuple( int( -1 ) );
  }
}

} //namespace mouse_io

BOOST_PYTHON_MODULE( jack_mouse_test )
{
  using namespace boost::python;
  Py_Initialize();
  np::initialize();
  def( "read_mouse_data", mouse_io::read_mouse_data );
}
