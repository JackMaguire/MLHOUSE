#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <math.h>
#include <fstream>
#include <sstream>
#include <string>

#include <iostream>

/*
  https://blog.esciencecenter.nl/irregular-data-in-pandas-using-c-88ce311cb9ef
  
*/

//namespace py = pybind11;

//using ndarray = xt::pyarray< float >;

using namespace boost;
using namespace boost::python;
using namespace boost::python::numpy;

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace mouse_io {

float
normalize_output_value(
  float value
){
  if( value > 1 ){
    value = sqrt( value );
  }
  value += 2.0;
  value /= 3.0;
  return value;
}

ndarray
generate_output_data(
  std::vector< std::array< std::string, 2 > > const & tokenized_file_lines_of_output_file
){
  int const total_number_of_elements = tokenized_file_lines_of_output_file.size();

  //https://www.boost.org/doc/libs/1_64_0/libs/python/doc/html/numpy/tutorial/simple.html
  p::tuple shape = p::make_tuple( 1, 1, 1, total_number_of_elements );
  np::dtype dtype = np::dtype::get_builtin<float>();
  np::ndarray output_values = np::empty( shape, dtype );
  
  // p::object tu = p::make_tuple( '1','1','1', std::to_string(total_number_of_elements) );
  //ndarray output_values( tu );
  float * ndarray_data = reinterpret_cast< float * > ( output_values.get_data() );

  for( int i = 0; i < total_number_of_elements; ++i ){
    ndarray_data[ i ] = std::stof( tokenized_file_lines_of_output_file[ i ][ 1 ] );
  }

  return output_values;
}

//ndarray
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
      // error
      // TODO - learn how to best handle errors in python
      continue;//for now, just omit this line
    }

    if ( ! std::getline( ss, new_elements[ 1 ], ',' ) ){
      std::cout << "Error at [1]" << std::endl;
      // error
      // TODO - learn how to best handle errors in python
      continue;//for now, just omit this line
    }

    tokenized_file_lines_of_output_file.emplace_back( new_elements );
  }

  // infile.close(); // Let RAII handle this

  return tokenized_file_lines_of_output_file;
}

//ndarray
boost::python::tuple
read_mouse_data(
  std::string const & output_data_filename
) {
  auto const tokenized_file_lines_of_output_file = read_in_output_data( output_data_filename );
  auto const output_data = generate_output_data( tokenized_file_lines_of_output_file );
  return boost::python::make_tuple( output_data );
}

} //namespace mouse_io

BOOST_PYTHON_MODULE( jack_mouse_test )
{
  using namespace boost::python;
  Py_Initialize();
  np::initialize();
  def( "read_mouse_data", mouse_io::read_mouse_data );
}
