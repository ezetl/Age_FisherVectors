#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <functional>
#include <numeric>
#include <sys/time.h>
#include "opencv2/xfeatures2d.hpp"

#define STRIDE 4
#define FACE_SIZE 100

// The VLFeat header files need to be declared external.
extern "C" {
#include <vl/generic.h>
#include <vl/gmm.h>
#include <vl/fisher.h>
#include <vl/dsift.h>
}

using namespace std;
using namespace cv;

#define TYPE float
#define VL_F_TYPE VL_TYPE_FLOAT

void save_image_descriptor(string filename, std::vector< std::vector<float> > feats) {
  /*
  string data_out = filename.substr(0,filename.find_last_of("."))+"_"+desc_name;
  cout << "Writing descriptor data in: " << data_out << endl;
  cv::FileStorage out_file(data_out, FileStorage::WRITE );
  out_file << "filename" << filename;
  out_file << "descriptor" << desc;
  out_file << "keypoints" << kpts;
  out_file.release();
  */
  string data_out = filename.substr(0, filename.find_last_of(".")) + "_sift";
  FILE *ofp;
  ofp = fopen(data_out.c_str(), "w");
  cout << "Writing descriptor data in: " << data_out << endl;

  int rows = feats.size();
  int cols = feats[0].size();
  fprintf(ofp, "%d %d \n", cols, rows);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      fprintf(ofp, "%f ", feats[i][j]);
    }
    fprintf(ofp, "\n");
  }
  fclose(ofp);
}

void load_image_descriptor(string filename, vector<KeyPoint> &loaded_kpts, Mat &desc) {
  /*
  string data_in = filename.substr(0,filename.find_last_of("."))+"_"+desc_name;
  cout << "Loading descriptor data from: " << data_in << endl;
  cv::FileStorage in_file(data_in, FileStorage::READ );
  in_file["descriptor"] >> loaded_desc;
  in_file["keypoints"] >> loaded_kpts;
  in_file.release();
  */

  FILE *ofp;

  string data_in = filename.substr(0, filename.find_last_of(".")) + "_sift";
  cout << "Loading descriptor data from: " << data_in << endl;
  ofp = fopen(data_in.c_str(), "r");

  int rows = 0;
  int cols = 0;
  int res = fscanf(ofp, "%d %d \n", &cols, &rows);
  if (res == EOF) {
    perror("There was an error reading the file.");
  }
  desc = Mat::zeros(rows, cols, CV_32F);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float a;
      int res = fscanf(ofp, "%f ", &a);
      if (res == EOF) {
        perror("There was an error reading the file.");
      }
      desc.at<float>(i, j) = a;
    }
    fprintf(ofp, "\n");
  }
  fclose(ofp);
}

void mat2float(cv::Mat img, std::vector<float>& img_float) {
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      img_float[i * img.rows + j] = img.at<unsigned char>(i, j) / 255.0f;
    }
  }
}

VlDsiftKeypoint normalize_kpoint(VlDsiftKeypoint const &orig_kp) {
  double x = orig_kp.x;
  double y = orig_kp.y;
  VlDsiftKeypoint new_point;
  new_point.x = x / (double)FACE_SIZE;
  // normalize between -1 and 1
  new_point.x = new_point.x * (1 - (-1)) + (-1);
  new_point.y = y / (double)FACE_SIZE;
  new_point.y = new_point.y * (1 - (-1)) + (-1);
  return new_point;
}

//void concatenate_features_kpoints(float const *descriptors1, VlDsiftKeypoint const *keypoints1, int nkps1,
//                                  float const *descriptors2, VlDsiftKeypoint const *keypoints2, int nkps2,
//                                  int desc_size, std::vector< std::vector<float> > &concat_feats) {
void concatenate_features_kpoints(float const *descriptors1, int nkps1,
                                  float const *descriptors2, int nkps2,
                                  int desc_size, std::vector< std::vector<float> > &concat_feats) {
    std::cout << "nkps1: " << nkps1 << std::endl;
    std::cout << "nkps2: " << nkps2 << std::endl;
  for (int i = 0; i < nkps1; ++i) {
    std::vector<float> tmp;
    for (int j = 0; j < desc_size; ++j) {
      tmp.push_back(descriptors1[i * desc_size + j]);
    }
    //VlDsiftKeypoint new_kp = normalize_kpoint(keypoints1[i]);
    //tmp.push_back(new_kp.x);
    //tmp.push_back(new_kp.y);
    concat_feats.push_back(tmp);
  }
  for (int i = 0; i < nkps2; ++i) {
    std::vector<float> tmp;
    for (int j = 0; j < desc_size; ++j) {
      tmp.push_back(descriptors2[i * desc_size + j]);
    }
    //VlDsiftKeypoint new_kp = normalize_kpoint(keypoints2[i]);
    //tmp.push_back(new_kp.x);
    //tmp.push_back(new_kp.y);
    concat_feats.push_back(tmp);
  }
}

bool compute_sift_descriptor(string filename) {
  // Read image
  Mat img = imread(filename, IMREAD_GRAYSCALE);

  // Check for invalid input
  if (!img.data) {
    cout << "Could not open or find the image" << std::endl;
    return false;
  }

  // Resize image
  Mat aux_img;
  //TODO: align image here
  resize(img, aux_img, Size(100, 100), INTER_CUBIC);
  std::vector<float> img_float(aux_img.rows * aux_img.cols);
  mat2float(aux_img, img_float);

  // Compute descriptor
  VlDsiftFilter* vlf = vl_dsift_new(aux_img.rows, aux_img.cols);
  vl_dsift_set_steps(vlf, STRIDE, STRIDE);

  // numBinT, numBinX, numBinY, binSizeX, binSizeY
  VlDsiftDescriptorGeometry vlf_geometry = {8, 4, 4, 4, 4};
  vl_dsift_set_geometry(vlf, &vlf_geometry);
  vl_dsift_process(vlf, &img_float[0]);
  int nkps1 = vl_dsift_get_keypoint_num(vlf);
  int sdesc = vl_dsift_get_descriptor_size(vlf);
  assert(sdesc == 128);
  VlDsiftKeypoint const* keypoints1 = vl_dsift_get_keypoints(vlf);
  float const* tmp_descs = vl_dsift_get_descriptors(vlf);
  float* descriptors1 = (float*)malloc((size_t)(nkps1 * sdesc * sizeof(float))); 
  descriptors1 = (float*)memcpy((void*)descriptors1, (void*)tmp_descs, (size_t)(nkps1*sdesc*sizeof(float)));


  vlf_geometry = {8, 4, 4, 8, 8};
  vl_dsift_set_geometry(vlf, &vlf_geometry);
  vl_dsift_process(vlf, &img_float[0]);
  int nkps2 = vl_dsift_get_keypoint_num(vlf);
  sdesc = vl_dsift_get_descriptor_size(vlf);
  assert(sdesc == 128);
  VlDsiftKeypoint const* keypoints2 = vl_dsift_get_keypoints(vlf);
  float const* descriptors2 = vl_dsift_get_descriptors(vlf);

  //TODO(ezetlopez): run PCA here, set sdesc to 66 (64 PCA and x,y coords)

  std::cout << "about to concatenate\n";
  std::cout << filename << std::endl;
  std::vector< std::vector<float> > concat_feats;
  //concatenate_features_kpoints(descriptors1, keypoints1, nkps1, descriptors2, keypoints2, nkps2, sdesc, concat_feats);
  concatenate_features_kpoints(descriptors1, nkps1, descriptors2, nkps2, sdesc, concat_feats);
  std::cout << "concat size: " << concat_feats.size() << std::endl;

  // Save descriptor
  save_image_descriptor(filename, concat_feats);

  vl_dsift_delete(vlf);
  free(descriptors1);
  return true;
}

std::vector<std::string> load_paths(string list_paths){
  ifstream infile;
  infile.open(list_paths.c_str());
  string path;
  float age; 
  std::vector<string> paths;
  while (infile >> path >> age) {
      paths.push_back(path);
  }
  infile.close();
  return paths;
}

void compute_descriptors_and_save(string list_paths) {
  vector<string> images_list = load_paths(list_paths);
  size_t count = images_list.size();
  cout << "Images found: " << " " << count << endl;
  for (size_t i = 0; i < count; i++) {
    compute_sift_descriptor(images_list[i]);
  }
}

void print_image_descriptor(Mat &desc, unsigned int MAX_DESC) {
  vl_size numData = desc.rows;
  vl_size dimension = desc.cols;

  TYPE *data = (TYPE *)vl_malloc(sizeof(TYPE) * numData * dimension);
  for (unsigned int row = 0; row < numData && row < MAX_DESC; row++) {
    for (unsigned int col = 0; col < dimension; col++) {
      data[row * dimension + col] = desc.at<float>(row, col) / 255.0f;
      // data[row*dimension+col] = (desc.at<float>(row, col));
      // VL_PRINT("%f ",(float)desc.at<float>(row,col));
      VL_PRINT("%f ", data[row * dimension + col]);
    }
    VL_PRINT("\n");
  }
  vl_free(data);
}

void save_gmm(VlGMM *gmm, string dataFileResults, vl_size numData) {
  FILE *ofp;

  vl_size d, cIdx;

  vl_size dimension = vl_gmm_get_dimension(gmm);
  vl_size numClusters = vl_gmm_get_num_clusters(gmm);
  vl_type dataType = vl_gmm_get_data_type(gmm);
  float const *sigmas = (float *)vl_gmm_get_covariances(gmm);
  float const *means = (float *)vl_gmm_get_means(gmm);
  float const *weights = (float *)vl_gmm_get_priors(gmm);
  int verb = vl_gmm_get_verbosity(gmm);
  vl_size iterations = vl_gmm_get_max_num_iterations(gmm);
  vl_size repetitions = vl_gmm_get_num_repetitions(gmm);

  ofp = fopen(dataFileResults.c_str(), "w");

  fprintf(ofp, "%d %llu %llu\n", verb, iterations, repetitions);
  fprintf(ofp, "%llu %llu %llu %u\n", numData, dimension, numClusters, dataType);

  for (cIdx = 0; cIdx < numClusters; cIdx++) {
    for (d = 0; d < dimension; d++) {
      // cout << means[cIdx*dimension+d] << " ";
      fprintf(ofp, "%f ", ((float *)means)[cIdx * dimension + d]);
    }
    for (d = 0; d < dimension; d++) {
      fprintf(ofp, "%f ", ((float *)sigmas)[cIdx * dimension + d]);
    }
    fprintf(ofp, "%f ", ((float *)weights)[cIdx]);
    fprintf(ofp, "\n");
  }
  fclose(ofp);
  cout << "GMM save finished..." << endl;
}

void load_gmm(VlGMM *&gmm, string dataFileResults) {
  FILE *ofp;

  vl_size d, cIdx;

  ofp = fopen(dataFileResults.c_str(), "r");

  vl_size dimension, numClusters, iterations, repetitions, numData;
  int verb;
  vl_type dataType;

  int res = fscanf(ofp, "%d %llu %llu", &verb, &iterations, &repetitions);
  if (res == EOF) {
    perror("There was an error reading the file.");
  }
  res = fscanf(ofp, "%llu %llu %llu %u", &numData, &dimension, &numClusters, &dataType);
  if (res == EOF) {
    perror("There was an error reading the file.");
  }

  float *means = (float *)vl_malloc(sizeof(float) * numClusters * dimension);
  float *weights = (float *)vl_malloc(sizeof(float) * numClusters);
  float *sigmas = (float *)vl_malloc(sizeof(float) * numClusters * dimension);

  for (cIdx = 0; cIdx < numClusters; cIdx++) {
    for (d = 0; d < dimension; d++) {
      res = fscanf(ofp, "%f ", &((float *)means)[cIdx * dimension + d]);
      if (res == EOF) {
        perror("There was an error reading the file.");
      }
    }
    for (d = 0; d < dimension; d++) {
      res = fscanf(ofp, "%f ", &((float *)sigmas)[cIdx * dimension + d]);
      if (res == EOF) {
        perror("There was an error reading the file.");
      }
    }
    res = fscanf(ofp, "%f ", &((float *)weights)[cIdx]);
    if (res == EOF) {
      perror("There was an error reading the file.");
    }
    res = fscanf(ofp, "\n");
    if (res == EOF) {
      perror("There was an error reading the file.");
    }
  }

  gmm = vl_gmm_new(VL_F_TYPE, dimension, numClusters);

  vl_gmm_set_covariances(gmm, sigmas);
  vl_gmm_set_means(gmm, means);
  vl_gmm_set_priors(gmm, weights);
  vl_gmm_set_verbosity(gmm, verb);
  vl_gmm_set_max_num_iterations(gmm, iterations);
  vl_gmm_set_num_repetitions(gmm, repetitions);

  fclose(ofp);
}

void train_gmm(void *data, vl_size numData, vl_size dimension, vl_size numClusters, VlGMM *&gmm) {
  cout << "\nTrain GMM__________________________________________________" << endl;
  // create a new instance of a GMM object for float data
  cout << "Creating GMM with: " << numData << " descriptors of length " << dimension << endl;
  gmm = vl_gmm_new(VL_F_TYPE, dimension, numClusters);
  // set the maximum number of EM iterations to 100
  vl_gmm_set_max_num_iterations(gmm, 100);
  // set the initialization to random selection
  vl_gmm_set_initialization(gmm, VlGMMRand);
  // cluster the data, i.e. learn the GMM
  vl_gmm_cluster(gmm, data, dimension);
}

void compute_fisher_encoding_and_save(string filename, std::vector<float> &vec_enc, VlGMM *&gmm,
                                      vl_size numClusters) {
  cout << "\nFV encoding__________________________________________________" << endl;

  // Load image descriptor
  vector<KeyPoint> l_kpts;
  Mat l_desc;
  load_image_descriptor(filename, l_kpts, l_desc);

  // Transform descriptors data to feed FV
  vl_size numData = l_desc.rows;
  vl_size dimension = l_desc.cols;
  TYPE *data_to_encode = (TYPE *)vl_malloc(sizeof(TYPE) * numData * dimension);
  for (unsigned int row = 0; row < numData; row++)
    for (unsigned int col = 0; col < dimension; col++) {
      // data_to_encode[row*dimension+col] = (l_desc.at<unsigned char>(row, col)/ 255.0f);
      // cout << l_desc.at<unsigned char>(row, col)/ 255.0f << endl;
      data_to_encode[row * dimension + col] = (l_desc.at<float>(row, col));
      // cout << l_desc.at<float>(row, col) << endl;
    }

  // Compute FV. Enc is a vector of size equal to twice the product of dimension and numClusters
  TYPE *enc = (TYPE *)vl_malloc(sizeof(TYPE) * 2 * dimension * numClusters);
  vl_fisher_encode(enc, VL_F_TYPE, vl_gmm_get_means(gmm), dimension, numClusters, vl_gmm_get_covariances(gmm),
                   vl_gmm_get_priors(gmm), data_to_encode, numData, VL_FISHER_FLAG_NORMALIZED);

  // Save FV into a file and convert enc to vector<float>
  FILE *ofp;
  string fv_data_out = filename.substr(0, filename.find_last_of(".")) + "_fv";
  cout << "Writing FV data in: " << fv_data_out << endl;
  ofp = fopen(fv_data_out.c_str(), "w");
  for (unsigned int i = 0; i < 2 * dimension * numClusters; i++) {
    fprintf(ofp, " %f ", (enc)[i]);
    vec_enc.push_back((enc)[i]);
  }
  fclose(ofp);

  // Free memory
  vl_free(enc);
  vl_free(data_to_encode);
}

void load_fisher_encoding(vector<float> &enc_vec, string filename) {
  ifstream infile;
  string fv_file = filename.substr(0, filename.find_last_of(".")) + "_fv";
  // cout << "Loading FV data from: " << fv_file << endl;
  infile.open(fv_file.c_str());
  while (!infile.eof()) {
    float element;
    infile >> element;
    if (infile.fail()) break;
    enc_vec.push_back(element);
  }
  infile.close();
}

vl_size load_all_descriptors(std::vector<std::string> &images_list, float *&data, int gmm_words, int count,
                             vl_size dimension) {
  // Concatenate descriptors of gmm_words images to train GMM
  int all_elements = 0;
  int all_words = 0;
  int all_descriptors = 0;

  // FILE *ofp = fopen("./GMMDescriptors", "w+");

  unsigned int GMM_size = min(gmm_words, (int)count);
  for (size_t i = 0; i < GMM_size; i++) {
    // Load descriptors
    Mat l_desc;
    vector<KeyPoint> l_kpts;
    load_image_descriptor(images_list[i], l_kpts, l_desc);

    // Concatenate descriptors
    vl_size numData = l_desc.rows;
    cout << "Descriptors to add: " << numData << endl;
    // fprintf(ofp, "%llu %llu ", numData, dimension);
    for (unsigned int row = 0; row < numData; row++) {
      for (unsigned int col = 0; col < dimension; col++) {
        data[row * dimension + col] = (l_desc.at<float>(row, col));
        // fprintf(ofp, "%f ",((float*) data)[all_elements+row*dimension+col]);
      }
    }
    all_words += 1;
    all_descriptors += numData;
    all_elements += numData * dimension;
    cout << "Words: " << all_words << " elements: " << all_elements << endl;
  }
  // fclose (ofp);

  return all_descriptors;
}

void example_get_indexing(string list_paths, int gmm_words, vl_size dimension, vl_size numClusters) {
  // Load images list from dataset and shuffle them
  std::vector<std::string> images_list = load_paths(list_paths);
  size_t count = images_list.size();
  random_shuffle(images_list.begin(), images_list.end());
  cout << "\nImages found: " << " " << count << endl;

  // Load all images descriptors
  TYPE *all_descriptors_data = (TYPE *)vl_malloc(sizeof(TYPE) * gmm_words * 30000 * dimension);
  vl_size desc_amount = load_all_descriptors(images_list, all_descriptors_data, gmm_words, count, dimension);

  // Train GMM
  VlGMM *gmm = NULL;
  train_gmm(all_descriptors_data, desc_amount, dimension, numClusters, gmm);
  assert(gmm != NULL);
  string file_out = "./gmm_object_sift";
  save_gmm(gmm, file_out, desc_amount);
  // Free memory
  vl_free(all_descriptors_data);

  // Generate FV for all images in the dataset and save them
  for (size_t i = 0; i < count; i++) {
    vector<float> enc;
    compute_fisher_encoding_and_save(images_list[i], enc, gmm, numClusters);
    cout << "Fisher vector size: " << enc.size() << endl;
  }

  // Free memory
  vl_gmm_delete(gmm);
}

void show_results(vector<int> &index, std::vector<std::string> &images_list, int k) {
  int width = 640;
  int height = 480;
  Mat to_show = Mat::zeros((k / 3) * height, 3 * width, CV_8UC3);
  for (unsigned int i = 0; i < index.size(); i++) {
    Mat img = imread(images_list[index[i]]);
    Mat res = Mat::zeros(img.cols, img.rows, CV_8UC3);
    resize(img, res, Size(width, height));
    Mat aux(to_show, Rect((i % 3) * width, (i / 3) * height, width, height));
    res.copyTo(aux);
  }

  // Save result as <timestamp>.jpg
  struct timeval tp;
  gettimeofday(&tp, NULL);
  long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
  ostringstream stream;
  stream << "./result_" << ms << "_sift.jpg";
  string path = stream.str();
  imwrite(path, to_show);
  cout << "Result saved at: " << path << endl;
}

void get_knn(string img_query, string list_paths, int k, VlGMM *&gmm, vl_size numClusters,
             vl_size dimension, bool load) {
  // Compute FV for query image
  // load_fisher_encoding(enc_query, img_query);
  vector<float> enc_query;
  compute_fisher_encoding_and_save(img_query, enc_query, gmm, numClusters);
  cout << "Fisher vector size: " << enc_query.size() << endl;
  Mat query = Mat(1, enc_query.size(), CV_32F);
  memcpy(query.data, enc_query.data(), enc_query.size() * sizeof(float));

  // Load dataset FVs
  std::vector<std::string> images_list;
  images_list = load_paths(list_paths);
  size_t count = images_list.size();
  cout << "\nImages found: " << " " << count << endl;

  // Concatenate FVs
  cout << "\nLoading FVs for knn:" << endl;
  Mat dataset(count, enc_query.size(), CV_32F);
  for (size_t i = 0; i < count; i++) {
    float *row = dataset.ptr<float>(i);
    vector<float> aux_enc;
    load_fisher_encoding(aux_enc, images_list[i]);
    memcpy(row, aux_enc.data(), aux_enc.size() * sizeof(float));
    cout << "Loading " << (i / (float)count) * 100 << " % \r";
  }

  cout << "\n\nKnn search working... " << endl;

  // Search
  vector<int> index(k);
  vector<float> dist(k);

  if (!load) {
    cout << "Saving kdtree index into memory..." << endl;
    // KdTree with "count" random trees
    flann::KDTreeIndexParams labels(count);
    // Constructs a nearest neighbor search index for a given dataset
    flann::Index kdtree(dataset, labels);
    string file_out = "./kdtree_sift";
    kdtree.save(file_out);
    kdtree.knnSearch(query, index, dist, k, cv::flann::SearchParams(32));
  } else {
    cout << "Loading kdtree index from memory..." << endl;
    string file_in = "./kdtree_sift";
    flann::Index kdtree(dataset, flann::SavedIndexParams(file_in));
    kdtree.knnSearch(query, index, dist, k, cv::flann::SearchParams(32));
  }

  // Print single search results
  cout << "\nTop " << k << " matches_____________________________ \n";
  for (unsigned int i = 0; i < index.size(); i++)
    cout << "(index, dist): " << index[i] << ",\t" << dist[i] << " \t" << images_list[index[i]] << endl;

  // Concatenate top-k images to show results
  show_results(index, images_list, k);
}

void encoding_similarity_test(string img1, string img2, vector<pair<double, int> > &l2,
                              vector<pair<double, int> > &euclidean, int index) {
  // Load FV
  vector<float> enc1, enc2;
  load_fisher_encoding(enc1, img1);
  load_fisher_encoding(enc2, img2);

  // Distance comparision
  double dist_l2 = norm(enc1, enc2, NORM_L2);
  l2.push_back(make_pair(dist_l2, index));
  cout << "L2 distance: " << dist_l2 << endl;

  double dist_euc = inner_product(enc1.begin(), enc1.end(), enc2.begin(), 0.0);
  euclidean.push_back(make_pair(dist_euc, index));
  cout << "Dot product: " << dist_euc << endl;
}

void no_encoding_similarity_test(Mat &desc1, Mat &desc2, vector<KeyPoint> &kpts1, vector<KeyPoint> &kpts2) {
  BFMatcher matcher(NORM_L2);
  vector<vector<DMatch> > nn_matches;
  matcher.knnMatch(desc1, desc2, nn_matches, 2);

  vector<KeyPoint> matched1, matched2;
  vector<DMatch> good_matches;
  for (size_t i = 0; i < nn_matches.size(); i++) {
    DMatch first = nn_matches[i][0];
    float dist1 = nn_matches[i][0].distance;
    float dist2 = nn_matches[i][1].distance;

    if (dist1 < 0.8f * dist2) {
      matched1.push_back(kpts1[first.queryIdx]);
      matched2.push_back(kpts2[first.trainIdx]);
    }
  }

  double inlier_ratio = matched1.size() * 1.0 / kpts1.size();
  cout << "A-KAZE Matching Results______________________________________" << endl;
  cout << "Keypoints 1:                        \t" << kpts1.size() << endl;
  cout << "Keypoints 2:                        \t" << kpts2.size() << endl;
  cout << "Matches:                            \t" << matched1.size() << endl;
  cout << "Ratio:                              \t" << inlier_ratio << endl;
}

static bool cmp_l2(pair<double, int> u, pair<double, int> v) { return u.first < v.first; }

void get_k_brute_force(string img_query, string list_paths, int k, VlGMM *&gmm, vl_size numClusters,
                       vl_size dimension) {
  // Compute FV for query image
  // load_fisher_encoding(enc_query, img_query);
  vector<float> enc_query;
  compute_fisher_encoding_and_save(img_query, enc_query, gmm, numClusters);
  cout << "Fisher vector size: " << enc_query.size() << endl;
  Mat query = Mat(1, enc_query.size(), CV_32F);
  memcpy(query.data, enc_query.data(), enc_query.size() * sizeof(float));

  // Load dataset FVs
  std::vector<std::string> images_list;
  images_list = load_paths(list_paths);
  size_t count = images_list.size();
  cout << "\nImages found: " << " " << count << endl;

  // Concatenate FVs
  vector<pair<double, int> > l2, euc;
  cout << "\nBrute force:" << endl;
  for (size_t i = 0; i < count; i++) encoding_similarity_test(img_query, images_list[i], l2, euc, i);

  sort(l2.begin(), l2.end(), cmp_l2);
  // sort(euc.begin(), euc.end(), cmp);

  // Print single search results
  std::vector<int> index;
  cout << "\nTop " << k << " matches_____________________________ \n";
  for (int i = 0; i < k; i++) {
    cout << "(l2, euc): " << l2[i].first << ",\t" << euc[i].first << " \t" << images_list[l2[i].second] << endl;
    index.push_back(l2[i].second);
  }
  // Concatenate top-k images to show results
  show_results(index, images_list, k);
}

void example_query(string query_img, string list_paths, vl_size numClusters, vl_size dimension,
                   bool load_index) {
  bool success = false;
  success = compute_sift_descriptor(query_img);

  if (success) {
    VlGMM *gmm = NULL;
    string file_in = "./gmm_object_sift";
    load_gmm(gmm, file_in);
    get_knn(query_img, list_paths, 6, gmm, numClusters, dimension, load_index);
    // get_k_brute_force(query_img, list_paths, 9, gmm, desc_name, numClusters, dimension);
    vl_gmm_delete(gmm);
  }
}

void help() {
  cout << "Modes: descriptors | indexing | knn.\nIf 'descriptors': \nYou must provide "
          "<images_list> as argument. \nIf 'indexing': \nYou must provide <images_list> as "
          "argument. \nIf 'knn': \nYou must provide <images_list> <query_image> {'save' | 'load'} "
          "as arguments where 'save' means store kdtree index (to be used first time), and 'load' will read it from "
          "memory. \n" << endl;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "Wrong arguments, please read the description above.\n" << endl;
    help();
    return 0;
  }

  string mode = argv[1];

  // Amount of images to be used to train GMM
  int gmm_words = 10;
  // Amount of clusters for GMM
  vl_size numClusters = 30;
  // Dimension for GMM
  vl_size dimension;

  // Test distance between two FVs
  // encoding_similarity_test("../dataset/holidays/105401.jpg", "../dataset/holidays/114400.jpg");

  dimension = 128;

  // Compute descriptors
  if (argc >= 2 && strcmp(mode.c_str(), "descriptors") == 0) {
    string list_paths = argv[2];
    compute_descriptors_and_save(list_paths);
    return 0;
  }

  // Example shows how to train GMM and index a database of images (Load descriptors, train GMM, create FVs)
  if (argc >= 2 && strcmp(mode.c_str(), "indexing") == 0) {
    string list_paths = argv[2];
    example_get_indexing(list_paths, gmm_words, dimension, numClusters);
    return 0;
  }

  // Example shows how to query an image and get N most similar images
  if ((argc >= 3) & (strcmp(mode.c_str(), "knn") == 0)) {
    string list_paths = argv[2];
    string query_img = argv[3];
    string load_kdtree = argv[4];

    if (strcmp(load_kdtree.c_str(), "load") == 0) {
      example_query(query_img, list_paths, numClusters, dimension, true);
    } else {
      if (strcmp(load_kdtree.c_str(), "save") == 0) {
        example_query(query_img, list_paths, numClusters, dimension, false);
      }
    }
    return 0;
  } else {
    cout << "Invalid arguments." << endl;
    return 0;
  }
}
