#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <typeindex>
#include <typeinfo>
#include <fstream>

#include "csv.hpp"

using namespace std;

void get_series_idx (
    float * arr,
    int series_idx,
    int pol_idx,
    vector<float[2]> &vec
  )
{
  int polsize = vec.size();
  int total_polsize = 4 * polsize; // 2 polarizations, complex
  float real_part;
  float imag_part;
  int idx;
  for (int i=0; i<polsize; i++) {
    idx = (4*i)+(2*pol_idx) + (series_idx*total_polsize);
    real_part = arr[idx];
    imag_part = arr[idx+1];
    vec[i][0] = real_part;
    vec[i][1] = imag_part;
  }
}


int main () {
  const char * file_name = "os_channelized_pulsar.dump.ref";
  const int hdrsize = 4096;
  const int npol = 2;
  const int M = 7;
  const int Nin = M*pow(2, 14);
  const int nseries = 80;
  const int polsize = Nin/M ;
  const int datasize = 2*npol*polsize*nseries;

  printf("header size: %i\n", hdrsize);
  printf("data size: %i\n", datasize);

  uint8_t * hdr = new uint8_t[hdrsize];
  float * data = new float[datasize];

  ifstream f(file_name, ios::binary);

  // read in header, full of zeros.
  uint8_t temp_int;
  for (int i=0; i<hdrsize; i++) {
    f.read(reinterpret_cast<char*>(&temp_int), sizeof(uint8_t));
    hdr[i] = temp_int;
  }

  // now read in data
  float temp_float;
  for (int i=0; i<datasize; i++) {
    f.read(reinterpret_cast<char*>(&temp_float), sizeof(float));
    data[i] = temp_float;
  }
  f.close();

  for (int i=0; i<30; i++) {
    printf("%.2f ", data[i]);
  }
  printf("\n");

  vector<float[2]> pol1_series1(polsize);
  vector<float[2]> pol2_series1(polsize);
  get_series_idx(data, 0, 0, pol1_series1);
  get_series_idx(data, 0, 1, pol2_series1);

  const char * file_name_pol1 = "pol1.csv";
  const char * file_name_pol2 = "pol2.csv";
  ofstream f_pol1; f_pol1.open(file_name_pol1);
  ofstream f_pol2; f_pol2.open(file_name_pol2);
  for (int i=0; i<pol1_series1.size(); i++) {
    csv::write_row<float>(f_pol1, pol1_series1[i], 2);
    csv::write_row<float>(f_pol2, pol2_series1[i], 2);
  }
  f_pol1.close();
  f_pol2.close();

  delete [] hdr;
  delete [] data;
  return 0;
}
