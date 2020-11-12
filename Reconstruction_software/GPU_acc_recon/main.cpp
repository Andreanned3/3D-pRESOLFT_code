#include "reconstruction.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

#define LIBDLL extern "C" __declspec(dllexport)

LIBDLL void calc_coeff_grid_size(int, int, int*, int*, float*);

int main() {

	Mat image_raw1 = imread("Substack_1.tif", CV_LOAD_IMAGE_ANYDEPTH);   // Read the file
	Mat image_raw2 = imread("Substack_1.tif", CV_LOAD_IMAGE_ANYDEPTH);   // Read the file

	Mat image1(image_raw1.rows, image_raw1.cols, CV_16U);
	Mat image2(image_raw2.rows, image_raw2.cols, CV_16U);

	image_raw1.convertTo(image1, CV_16U);
	image_raw2.convertTo(image2, CV_16U);

	if (!image1.data || !image2.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
	}
	if (image1.empty() || image2.empty())
		std::cout << "failed to open img.jpg" << std::endl;
	else
		std::cout << "img.jpg loaded OK" << std::endl;

	uchar* data_ptrs[2];
	data_ptrs[0] = image1.data;
	data_ptrs[1] = image2.data;
	
	const int im_rows = image1.rows;
	const int im_cols = image1.cols;
	const int im_slices = sizeof(data_ptrs) / sizeof(data_ptrs[0]);

	float pattern[4] = { 3.0f, 3.0f, 11.13f, 11.13f };
	
	float sigmas[2] = { 1.68, 0 };

	const int nr_bases = 2;

	int grid_rows, grid_cols;
	calc_coeff_grid_size(im_rows, im_cols, &grid_rows, &grid_cols, pattern);	
	
	uchar** coeffs_ptrs[nr_bases];
	//coeffs_ptrs.resize(nr_bases);
	vector<vector<Mat>> coeffs_mats_per_base;
	coeffs_mats_per_base.resize(nr_bases);

	//for (int q = 0; q < nr_bases; q++) {
	//	coeffs_mats_per_base[q].resize(im_slices);
	//	for (int i = 0; i < im_slices; i++) {
	//		coeffs_mats_per_base[q][i].create(grid_rows, grid_cols, CV_32F);
	//	}
	//}

	//Reconstruction reconstruction(im_rows, im_cols, im_slices, pattern, sigma, data_ptrs, &coeffs_mats_per_base);

	uchar* base1_coeffs[im_slices];
	uchar* base2_coeffs[im_slices];

	int q = 0;
	
	coeffs_mats_per_base[q].resize(im_slices);
	for (int i = 0; i < im_slices; i++) {
		coeffs_mats_per_base[q][i].create(grid_rows, grid_cols, CV_32F);
		base1_coeffs[i] = coeffs_mats_per_base[q][i].data;
	}
	coeffs_ptrs[q] = base1_coeffs;

	q = 1;

	coeffs_mats_per_base[q].resize(im_slices);
	for (int i = 0; i < im_slices; i++) {
		coeffs_mats_per_base[q][i].create(grid_rows, grid_cols, CV_32F);
		base2_coeffs[i] = coeffs_mats_per_base[q][i].data;
	}
	coeffs_ptrs[q] = base2_coeffs;

	Reconstruction reconstruction(im_rows, im_cols, im_slices, pattern, nr_bases, sigmas, data_ptrs, coeffs_ptrs);

	reconstruction.make_subsquares();
	reconstruction.make_bases_im();
	reconstruction.make_pinv_im();
	reconstruction.extract_signal_GPU();
	
	if (reconstruction.cudaStatus == cudaSuccess)
		printf("Cuda success!\n");

	printf("Elapsed time = %f\n", reconstruction.elapsed);

}

LIBDLL void calc_coeff_grid_size(int im_rows, int im_cols, int* grid_rows, int* grid_cols, float* pattern) {

	float pat_start_row = pattern[0];
	float pat_start_col = pattern[1];
	float pat_period_rows = pattern[2];
	float pat_period_cols = pattern[3];

	*grid_rows = (int)((im_rows - 1) / pat_period_rows + 1.0f / 2.0f - pat_start_row / pat_period_rows) + 1;
	*grid_cols = (int)((im_cols - 1) / pat_period_cols + 1.0f / 2.0f - pat_start_col / pat_period_cols) + 1;
}


LIBDLL void extract_signal_GPU(int im_rows, int im_cols, int im_slices, float* pattern, int nr_bases, float* sigmas, uchar** data_ptrs, uchar*** coeffs_ptrs) {

	Reconstruction reconstruction(im_rows, im_cols, im_slices, pattern, nr_bases, sigmas, data_ptrs, coeffs_ptrs);

	reconstruction.make_subsquares();
	reconstruction.make_bases_im();
	reconstruction.make_pinv_im();
	reconstruction.extract_signal_GPU();

}

LIBDLL void extract_signal_CPU(int im_rows, int im_cols, int im_slices, float* pattern, int nr_bases, float* sigmas, uchar** data_ptrs, uchar*** coeffs_ptrs) {

	Reconstruction reconstruction(im_rows, im_cols, im_slices, pattern, nr_bases, sigmas, data_ptrs, coeffs_ptrs);

	reconstruction.make_subsquares();
	reconstruction.make_bases_im();
	reconstruction.make_pinv_im();
	reconstruction.extract_signal_CPU();

}