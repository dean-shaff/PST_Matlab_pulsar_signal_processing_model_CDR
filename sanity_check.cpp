#include <math.h>
#include <stdio.h>
#include <vector>
#include <fstream>

using namespace std;

int main () {
  const char * file_name = "sanity_check.dump";

  ifstream f(file_name, ios::binary);

  char * buffer = new char[sizeof(double)];
  double temp;
  vector<double> arr;


  // read is going to read in sizeof(double) bytes into
  // the address of temp, which we have recast as a pointer
  // to a char (array).
  while (f.read((char *)&temp, sizeof(double))) {
    arr.push_back(temp);
    printf("%.2f  ", arr[arr.size() - 1]);
  }

  printf("\narr.size(): %i\n", arr.size());

  delete [] buffer;
  return 0;
}
