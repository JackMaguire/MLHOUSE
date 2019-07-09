#include <pybind11/pybind11.h>

#include "xtensor-python/pyarray.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

#include <math.h>
#include <fstream>
#include <sstream>
#include <string>

namespace py = pybind11;

template <class T>
using ndarray_tmpl = xt::pyarray<T, xt::layout_type::row_major>;

using ndarray = ndarray_tmpl< float >;

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
  std::vector< std::array< std::string, 3 > > const & tokenized_file_lines_of_output_file
){
  int const total_number_of_elements = tokenized_file_lines_of_output_file.size();

  ndarray output_values = xt::zeros< float >({total_number_of_elements});//TODO use better ctor

  for( int i = 0; i < total_number_of_elements; ++i ){
    output_values[ i ] = std::stof( tokenized_file_lines_of_output_file[ 1 ] );
  }

  return output_values;
}

ndarray
read_in_output_data(
  std::string const & filename
) {
  //https://stackoverflow.com/questions/7868936/read-file-line-by-line-using-ifstream-in-c
  //https://stackoverflow.com/questions/11719538/how-to-use-stringstream-to-separate-comma-separated-strings
  std::vector< std::array< std::string, 3 > > tokenized_file_lines_of_output_file;
  std::ifstream infile( filename );
  std::string line;
  while( std::getline( infile, line, ',' ) ) {
    std::istringstream iss( line );
    float a, b, c;
    if ( !(iss >> a >> b >> c) ) {
      // error
      break;
    }

    // process pair (a,b)
  }
}

int add(int i, int j) {
  return i + j;
}

PYBIND11_MODULE(example, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("add", &add, "A function which adds two numbers");
}

//c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix
