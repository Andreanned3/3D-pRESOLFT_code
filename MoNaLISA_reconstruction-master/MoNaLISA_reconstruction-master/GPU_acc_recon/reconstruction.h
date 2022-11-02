#pragma once

#include "cuda_runtime.h"
#include <opencv2/core/core.hpp>

#include <vector>

using namespace cv;
using namespace std;

class Reconstruction
{
public:
	Reconstruction(const int, const int, const int, float*, const int, float*, uchar**, vector<vector<Mat>>*);
	Reconstruction(const int, const int, const int, float*, const int, float*, uchar**, uchar***);
	~Reconstruction();

	void make_subsquares();
	void make_bases_im();
	void make_pinv_im();
	void extract_signal_GPU();
	void extract_signal_CPU();
	
	float elapsed;
	cudaError_t cudaStatus;

private:
	const int im_rows;
	const int im_cols;
	const int im_slices;

	int grid_rows;
	int grid_cols;

	float pattern[4];

	Mat ss_row;
	Mat ss_col;
	Mat basis_im;
	Mat pinv_im;

	vector<int> ss_row_start;
	vector<int> ss_row_end;
	vector<int> ss_col_start;
	vector<int> ss_col_end;

	//------Model-------
	int nr_bases;
	vector<float> sigmas;

	vector<Mat> bases_ims;
	vector<Mat> pinv_ims;

	uchar** raw_data_ptr;
	uchar*** coeffs_ptrs;
	vector<vector<Mat>> coeffs_mats;
};

