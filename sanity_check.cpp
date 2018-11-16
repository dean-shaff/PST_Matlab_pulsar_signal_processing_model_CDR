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

  while (f.read(reinterpret_cast<char*>(&temp), sizeof(double))) {
    arr.push_back(temp);
    printf("%.2f  ", arr[arr.size() - 1]);
  }
  // for (int i=0; i<91; i++) {
  //   f >> temp;
  //   arr.push_back(temp);
  //   printf("%.2f  ", temp);
  // }

  printf("\narr.size(): %i\n", arr.size());

  delete [] buffer;
  return 0;
}
