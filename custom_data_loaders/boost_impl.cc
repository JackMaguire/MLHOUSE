#include <pybind11/pybind11.h>

#include "xtensor-python/pyarray.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

#include <math.h>
#include <fstream>
#include <sstream>
#include <string>

/*
  https://blog.esciencecenter.nl/irregular-data-in-pandas-using-c-88ce311cb9ef
  
*/

namespace py = pybind11;

using ndarray = xt::pyarray< float >;

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

  ndarray output_values = xt::zeros< float >({total_number_of_elements});//TODO use better ctor

  for( int i = 0; i < total_number_of_elements; ++i ){
    output_values[ i ] = std::stof( tokenized_file_lines_of_output_file[ i ][ 1 ] );
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
  while( std::getline( infile, line, ',' ) ) {
    std::istringstream iss( line );
    std::string resid_tag;
    std::string energy;
    std::string ddg;

    if ( !( iss >> resid_tag >> energy >> ddg ) ) {
      // error
      // TODO - learn how to best handle errors in python
      continue;//for now, just omit this line
    }

    /*
    //"RESID: ".size() == 7
    constexpr int resid_tag_prefix_size = 7;

    if( resid_tag.substr( 0, resid_tag_prefix_size ) != "RESID: " ){
      // error
      // TODO - learn how to best handle errors in python
      continue;//for now, just omit this line      
    }

    //remove "RESID: " from beginning of tag and parse it as an int
    int resid = std::stoi( resid_tag.substr( resid_tag_prefix_size ) );
*/
    //Can't seem to get emplace back to work. 
    //Let's just go the slow way for now
    std::array< std::string, 2 > new_element;
    new_element[ 0 ] = resid_tag;
    new_element[ 1 ] = energy;
    //new_element[ 2 ] = ddg;

    tokenized_file_lines_of_output_file.emplace_back( new_element );
  }

  // infile.close(); // Let RAII handle this

  return tokenized_file_lines_of_output_file;
}

ndarray
read_mouse_data(
  std::string const & output_data_filename
) {
  auto const tokenized_file_lines_of_output_file = read_in_output_data( output_data_filename );
  auto const output_data = generate_output_data( tokenized_file_lines_of_output_file );
}

} //namespace mouse_io

int add(int i, int j) {
  return i + j;
}

PYBIND11_MODULE( example, m ) {
  //m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("read_mouse_data", &mouse_io::read_mouse_data, "TODO");
}

//c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`
