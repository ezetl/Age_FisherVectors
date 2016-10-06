#include <vector>
#include <iostream>
#include <fstream>
#include <dlib/svm.h>

#define FEATURES_SIZE 7800 
#define PATHS "/media/ezetl/0C74D0DD74D0CB1A/Datasets/Faces/imdbwiki_fv/imdbwiki_fvpathssss.txt"

using namespace dlib;
using namespace std;

typedef float label_type;
typedef dlib::matrix<float,FEATURES_SIZE,1> sample_type;
typedef dlib::linear_kernel<sample_type> lin_kernel;

sample_type load_fisher_encoding(string filename);
void load_fisher_vectors(string list_filename, std::vector<sample_type>& X, std::vector<label_type>& y);

int main(){
    std::vector<sample_type> X;
    std::vector<label_type> y;
    load_fisher_vectors(PATHS, X, y);
    rr_trainer<lin_kernel> trainer = rr_trainer<lin_kernel>();
    std::cout << "Started training...\n";
    decision_function<lin_kernel> trained = trainer.train(X, y);

    serialize("rr_function.dat") << trained;
    deserialize("rr_function.dat") >> trained;

    // Test
    std::cout << "Testing:\n";
    label_type diff = 0.0;
    for (unsigned int i =0 ; i < X.size(); ++i){
        label_type predicted = trained(X[i]);
        diff += (predicted > y[i]) ? predicted - y[i] : y[i] - predicted;
        std::cout << "\rMAE: " << diff / (float) (i+1) << std::endl; 
    }
    return 0;
}

void load_fisher_vectors(string list_filename, std::vector<sample_type>& X, std::vector<label_type>& y) {
  ifstream infile;
  infile.open(list_filename.c_str());
  string path;
  float age; 
  while (infile >> path >> age) {
      string fv_path = path.substr(0, path.find_last_of(".")) + "_fv";
      sample_type s = load_fisher_encoding(fv_path);
      X.push_back(s);
      y.push_back(age);
  }
  infile.close();
}

sample_type load_fisher_encoding(string filename) {
  sample_type fv;
  ifstream infile;
  string fv_file = filename.substr(0, filename.find_last_of(".")) + "_fv";
  infile.open(fv_file.c_str());
  int i = 0;
  float element;
  while (infile >> element) {
    sample_type(i,0) = element; 
    ++i;
  }
  infile.close();
  return fv;
}
