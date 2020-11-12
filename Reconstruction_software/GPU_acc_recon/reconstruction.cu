#include "reconstruction.h"

#include "device_launch_parameters.h"
#include "device_functions.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;


// KERNEL FUNCIONS
//----------------------
__global__ void GPU_signal_extraction(uchar* dev_data,
	uchar* dev_coeff,
	uchar* dev_ss_row,
	uchar* dev_ss_col,
	uchar* dev_pinv_im,
	int im_rows,
	int im_cols,
	const int data_step,
	const int ss_row_step,
	const int ss_col_step,
	const int pinv_im_step,
	const int coeff_step) {


	float prod;
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int idy = threadIdx.y + blockIdx.y * blockDim.y;

	int data_elem_size = sizeof(uint16_t);
	int ss_elem_size = sizeof(uint16_t);
	int pinv_elem_size = sizeof(float);
	int coeff_elem_size = sizeof(float);

	if (idy < im_rows && idx < im_cols) {

		uint16_t* pix_loc_data = (uint16_t*)(dev_data + data_step*idy + data_elem_size*idx);
		uint16_t* pix_loc_ss_row = (uint16_t*)(dev_ss_row + ss_row_step*idy + ss_elem_size*idx);
		uint16_t* pix_loc_ss_col = (uint16_t*)(dev_ss_col + ss_col_step*idy + ss_elem_size*idx);
		float* pix_loc_pinv_im = (float*)(dev_pinv_im + pinv_im_step*idy + pinv_elem_size*idx);

		float* pix_loc_coeff = (float*)(dev_coeff + coeff_step * *pix_loc_ss_row + coeff_elem_size * *pix_loc_ss_col);
		prod = (float)*pix_loc_data * *pix_loc_pinv_im;
		atomicAdd(pix_loc_coeff, prod);
	}

}

//__global__ void GPU_signal_extraction(cuda::GpuMat* dev_data,
//	cuda::GpuMat* dev_coeff,
//	cuda::GpuMat* dev_ss_row,
//	cuda::GpuMat* dev_ss_col,
//	cuda::GpuMat* dev_pinv_im,
//	int im_rows,
//	int im_cols) {
//
//
//	float prod;
//	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
//	const int idy = threadIdx.y + blockIdx.y * blockDim.y;
//
//	int data_elem_size = 2;
//	int ss_elem_size = 2;
//	int pinv_elem_size = 4;
//	int coeff_elem_size = 4;
//
//	if (idy < im_rows && idx < im_cols) {
//
//		printf("data adress on GPU = %d", dev_data->data);
//	}
//
//}

//---------------------


//CLASS FUNCTIONS-------------
Reconstruction::Reconstruction(const int in_im_rows, const int in_im_cols, const int in_im_slices, float* in_pattern, const int in_nr_bases, float* in_sigmas, uchar** rawdata_ptr, vector<vector<Mat>>* in_coeffs_mats)
	: im_rows(in_im_rows), im_cols(in_im_cols), im_slices(in_im_slices), nr_bases(in_nr_bases)
{
	for (int i = 0; i < this->nr_bases; i++) {
		this->sigmas.push_back(in_sigmas[i]);
	}

	this->pattern[0] = in_pattern[0];
	this->pattern[1] = in_pattern[1];
	this->pattern[2] = in_pattern[2];
	this->pattern[3] = in_pattern[3];

	this->coeffs_mats = *in_coeffs_mats;

	this->raw_data_ptr = rawdata_ptr;
}

Reconstruction::Reconstruction(const int in_im_rows, const int in_im_cols, const int in_im_slices, float* in_pattern, const int in_nr_bases, float* in_sigmas, uchar** rawdata_ptr, uchar*** in_coeffs_ptrs)
	:im_rows(in_im_rows), im_cols(in_im_cols), im_slices(in_im_slices), nr_bases(in_nr_bases)
{
	for(int i = 0; i < this->nr_bases; i++) {
		this->sigmas.push_back(in_sigmas[i]);
	}

	this->sigmas[0] = in_sigmas[0];
	this->sigmas[1] = in_sigmas[1];

	this->pattern[0] = in_pattern[0];
	this->pattern[1] = in_pattern[1];
	this->pattern[2] = in_pattern[2];
	this->pattern[3] = in_pattern[3];

	this->coeffs_ptrs = in_coeffs_ptrs;

	this->raw_data_ptr = rawdata_ptr;
}

Reconstruction::~Reconstruction()
{
}

void Reconstruction::make_subsquares() {
	Mat subsquares_row(im_rows, im_cols, DataType<uint16_t>::type);
	Mat subsquares_col(im_rows, im_cols, DataType<uint16_t>::type);

	int ss;
	float pat_start_row = this->pattern[0];
	float pat_period_rows = this->pattern[2];
	int mem = 0;
	Mat aux;

	this->ss_row_start.push_back(0);
	for (int i = 0; i < im_rows; i++) {
		ss = (int)(i / pat_period_rows + 1.0f / 2.0f - pat_start_row / pat_period_rows); //Cast to int implies floor operation.
		subsquares_row.row(i).setTo((uint16_t)ss);
		if (ss != mem) {
			this->ss_row_start.push_back(i);
			this->ss_row_end.push_back(i - 1);
			mem = ss;
		}
	}
	this->ss_row_end.push_back(subsquares_row.rows - 1);

	float pat_start_col = this->pattern[1];
	float pat_period_cols = this->pattern[3];
	mem = 0;

	this->ss_col_start.push_back(0);

	for (int j = 0; j < im_cols; j++) {
		ss = (int)(j / pat_period_cols + 1.0f / 2.0f - pat_start_col / pat_period_cols);
		subsquares_col.col(j).setTo((uint16_t)ss);
		if (ss != mem) {
			this->ss_col_start.push_back(j);
			this->ss_col_end.push_back(j - 1);
			mem = ss;
		}
	}
	this->ss_col_end.push_back(subsquares_row.cols - 1);

	this->grid_rows = subsquares_row.at<uint16_t>(im_rows - 1, 0) + 1;
	this->grid_cols = subsquares_col.at<uint16_t>(0, im_cols - 1) + 1;

	this->ss_row = subsquares_row;
	this->ss_col = subsquares_col;
}


void Reconstruction::make_bases_im() {
	float ss_center_row;
	float ss_center_col;

	Mat ss;
	Mat temp_bases_im(this->im_rows, this->im_cols, CV_32F);
	for (int q = 0; q < this->nr_bases; q++) {
		for (int i = 0; i < this->grid_rows; i++) {
			for (int j = 0; j < this->grid_cols; j++) {
				ss = temp_bases_im(Range(this->ss_row_start.at(i), this->ss_row_end.at(i) + 1), Range(this->ss_col_start.at(j), this->ss_col_end.at(j) + 1));
				ss_center_row = this->pattern[0] + i * this->pattern[2] - this->ss_row_start.at(i);
				ss_center_col = this->pattern[1] + j * this->pattern[3] - this->ss_col_start.at(j);
				for (int k = 0; k < ss.rows; k++) {
					for (int l = 0; l < ss.cols; l++) {
						if (this->sigmas[q] >= 9999) { //Code for Constant bg
							ss.at<float>(k, l) = 1;
						}
						else if (this->sigmas[q] == 0) { //Code for Constant bg
							ss.at<float>(k, l) = 0;
						}
						else {
							ss.at<float>(k, l) = exp(-(pow(k - ss_center_row, 2) + pow(l - ss_center_col, 2)) / (2 * pow(this->sigmas[q], 2))) / (2*M_PI*pow(this->sigmas[q], 2));
						}
					}
				}
			}
		}
		if(q == this->nr_bases - 1)
			this->bases_ims.push_back(temp_bases_im);
		else
			this->bases_ims.push_back(temp_bases_im.clone());
	}
}

void Reconstruction::make_pinv_im() {

	//Mat ss_row_debug(this->im_cols, this->im_cols, CV_16U, this->ss_row.data);

	float ss_center_row;
	float ss_center_col;
	Mat ss;
	Mat ss_vec;
	Mat ss_mat_inv;
	Mat ss_mat;
	Mat temp_mat;
	for (int q = 0; q < this->nr_bases; q++) {
		temp_mat.create(this->im_rows, this->im_cols, CV_32F);
		this->pinv_ims.push_back(temp_mat.clone());
	}
	Range row_range;
	Range col_range;
	for (int i = 0; i < this->grid_rows; i++) {
		for (int j = 0; j < this->grid_cols; j++) {
			row_range = Range(this->ss_row_start.at(i), this->ss_row_end.at(i) + 1);
			col_range = Range(this->ss_col_start.at(j), this->ss_col_end.at(j) + 1);
			ss_mat.create(this->nr_bases, row_range.size()*col_range.size(), CV_32F);
			for (int q = 0; q < this->nr_bases; q++) {
				ss = this->bases_ims[q](row_range, col_range).clone();
				ss.reshape(0, 1).copyTo(ss_mat.row(q));
			}
			ss_mat_inv = ss_mat.inv(cv::DECOMP_SVD);
			for (int q = 0; q < this->nr_bases; q++) {
				ss_vec = ss_mat_inv.col(q).clone();
				ss_vec.reshape(0, ss.rows).copyTo(ss);
				ss.copyTo(this->pinv_ims[q](row_range, col_range));
			}
		}
	}
}

void Reconstruction::extract_signal_GPU() {
	cudaEvent_t start, stop;
	cudaError_t cudaStatus;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block(32, 32);
	dim3 grid((this->im_cols + block.x - 1) / block.x, (this->im_rows + block.y - 1) / block.y);

	cuda::GpuMat dev_ss_row;
	cuda::GpuMat dev_ss_col;
	cuda::GpuMat dev_pinv_im;

	cuda::GpuMat dev_data;
	cuda::GpuMat dev_coeff(this->grid_rows, this->grid_cols, CV_32F);
	
	dev_ss_row.upload(this->ss_row);
	dev_ss_col.upload(this->ss_col);

	for (int q = 0; q < this->nr_bases; q++) {
		dev_pinv_im.upload(this->pinv_ims[q]);

		for (int i = 0; i < this->im_slices; i++) {
			Mat rawdata(this->im_rows, this->im_cols, CV_16U, this->raw_data_ptr[i]);
			Mat coeffs(this->grid_rows, this->grid_cols, CV_32F, this->coeffs_ptrs[q][i]);
			dev_data.upload(rawdata);
			dev_coeff.setTo(Scalar(0));

			cudaEventRecord(start);
			GPU_signal_extraction << < grid, block >> > (dev_data.data, dev_coeff.data, dev_ss_row.data, dev_ss_col.data, dev_pinv_im.data, im_rows, im_cols, dev_data.step, dev_ss_row.step, dev_ss_col.step, dev_pinv_im.step, dev_coeff.step);

			cudaDeviceSynchronize();
			cudaEventRecord(stop);

			//dev_coeff.download(this->coeffs_mats[q][i]);
			dev_coeff.download(coeffs);
		}

	}
	this->cudaStatus = cudaEventElapsedTime(&this->elapsed, start, stop);
}

void Reconstruction::extract_signal_CPU() {
	uint16_t ci, cj;
	for (int q = 0; q < this->nr_bases; q++) {
		for (int s = 0; s < this->im_slices; s++) {
			Mat rawdata(this->im_rows, this->im_cols, CV_16U, this->raw_data_ptr[s]);
			Mat coeffs(this->grid_rows, this->grid_cols, CV_32F, this->coeffs_ptrs[q][s]);

			for (int i = 0; i < this->im_rows; i++) {
				for (int j = 0; j < this->im_cols; j++) {
					ci = ss_row.at<uint16_t>(i, j);
					cj = ss_col.at<uint16_t>(i, j);
					coeffs.at<float>(ci, cj) += this->pinv_ims[q].at<float>(i, j) * rawdata.at<uint16_t>(i, j);
				}
			}
		}
	}
}
