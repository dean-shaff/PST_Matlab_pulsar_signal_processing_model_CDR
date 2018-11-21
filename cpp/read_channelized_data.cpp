#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string>
#include <vector>
#include <typeindex>
#include <typeinfo>
#include <fstream>

#include "csv.hpp"

using namespace std;

template<typename T>
T clock_t_to_num (clock_t t) {
  return ((T)t) / CLOCKS_PER_SEC;
}


/**
 * Data is structured like (n_pol, n_channel, n_bins, complex, n_series)
 * Data is structured like (n_series, n_pol, n_channel, n_bins, complex)
 */
void get_chan (
    float * arr,
    vector<float[2]> &vec,
    int i_series,
    int i_pol,
    int i_channel,
    const int n_pol=2,
    const int n_channel=8,
    const int n_series=2,
    const int n_bins=16384
  )
{
  int idx_real;
  int idx_imag;
  int incr = i_series + n_pol*n_series + i_channel*(n_series*n_pol);
  for (int i=0; i<n_bins; i++) {
    idx_real = incr + i*(n_series*n_pol*n_channel);
    idx_imag = idx_real + n_series*n_pol*n_channel*n_bins;
    vec[i][0] = arr[idx_real];
    vec[i][1] = arr[idx_imag];
  }
}


int main () {
  string file_name = "full_channelized_pulsar.dump";
  const int hdrsize = 4096;
  const int npol = 2;
  const int n_channel = 8;
  const int M = 7;
  const int Nin = M*pow(2, 14);
  const int n_series = 2;
  const int n_bins = Nin/M ;
  const int datasize = npol*n_channel*n_bins*2*n_series;

  printf("header size: %i\n", hdrsize);
  printf("data size: %i\n", datasize);

  uint8_t * hdr = new uint8_t[hdrsize];
  float * data = new float[datasize];

  ifstream f(file_name, ios::binary);

  clock_t t = clock();
  clock_t delta_t;
  // read in header, full of zeros.
  uint8_t temp_int;
  for (int i=0; i<hdrsize; i++) {
    f.read((char*)(&temp_int), sizeof(uint8_t));
    hdr[i] = temp_int;
  }

  // now read in data
  float temp_float;
  for (int i=0; i<datasize; i++) {
    // f.read(reinterpret_cast<char*>(&temp_float), sizeof(float));
    f.read((char*)(&temp_float), sizeof(float));
    data[i] = temp_float;
  }
  f.close();
  delta_t = clock() - t;
  printf("Took %.3f seconds to load data\n", clock_t_to_num<float>(delta_t));

  vector<string> file_names = {"pol1_channel1_series1.csv",
                               "pol2_channel1_series1.csv"};

  vector<float[2]> pol_data(n_bins) ;

  t = clock();
  for (int i=0; i<file_names.size(); i++) {
      ofstream f; f.open(file_names[i]);
      for (int c=2; c<3; c++) {
        get_chan(data, pol_data, 0, i, 2);
        for (int j=0; j<pol_data.size(); j++) {
          csv::write_row<float>(f, pol_data[j], 2);
        }
      }
      f.close();
  }
  delta_t = clock() - t;
  printf(
    "Took %.3f seconds to write channel data\n", clock_t_to_num<double>(delta_t));

  delete [] hdr;
  delete [] data;
  return 0;
}
