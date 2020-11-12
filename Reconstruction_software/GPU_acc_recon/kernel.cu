//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "device_functions.h"
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/cuda.hpp>
//
//
//#include <stdio.h>
//#include <iostream>
//#include <math.h>
//#include <vector>
//
//using namespace cv;
//using namespace std;
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//cudaEvent_t start, stop;
//#define LIBDLL extern "C" __declspec(dllexport)
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//	int i = threadIdx.x;
//	c[i] = a[i] + b[i];
//}
//
//void imshow_auto(string s, Mat im) {
//	double minVal, maxVal;
//	Point minLoc, maxLoc;
//
//	minMaxLoc(im, &minVal, &maxVal, &minLoc, &maxLoc);
//	Mat im_scaled = (im - minVal) * (256.0f / maxVal);
//	imshow(s, im_scaled);
//}
//
//void imshow_range(string s, Mat im, int* range) {
//	float max_of_type = (float)pow(2, 8*im.elemSize()); // 8 bits * size in bytes
//	Mat im_scaled = (im - range[0]) * (max_of_type / (range[1] - range[0]));
//	imshow(s, im_scaled);
//}
//
//void make_subsquares_(int im_rows, int im_cols, float* pattern, Mat* ret_Mat_row, Mat* ret_Mat_col, vector<int>* ss_row_start, vector<int>* ss_row_end, vector<int>* ss_col_start, vector<int>* ss_col_end) {
//	Mat subsquares_row(im_rows, im_cols, DataType<uint16_t>::type);
//	Mat subsquares_col(im_rows, im_cols, DataType<uint16_t>::type);
//
//	int ss;
//	float pat_start_row = pattern[0];
//	float pat_period_rows = pattern[2];
//	int mem = 0;
//	Mat aux;
//
//	ss_row_start->push_back(0);
//	for (int i = 0; i < subsquares_row.rows; i++) {
//		ss = (int)(i / pat_period_rows + 1.0f / 2.0f - pat_start_row / pat_period_rows); //Cast to int implies floor operation.
//		subsquares_row.row(i).setTo((uint16_t)ss);
//		if (ss != mem) {
//			ss_row_start->push_back(i);
//			ss_row_end->push_back(i - 1);
//			mem = ss;
//		}
//	}
//	ss_row_end->push_back(subsquares_row.rows - 1);
//
//	float pat_start_col = pattern[1];
//	float pat_period_cols = pattern[3];
//	mem = 0;
//
//	ss_col_start->push_back(0);
//
//	for (int j = 0; j < subsquares_col.cols; j++) {
//		ss = (int)(j / pat_period_cols + 1.0f / 2.0f - pat_start_col / pat_period_cols);
//		subsquares_col.col(j).setTo((uint16_t)ss);
//		if (ss != mem) {
//			ss_col_start->push_back(j);
//			ss_col_end->push_back(j - 1);
//			mem = ss;
//		}
//	}
//	ss_col_end->push_back(subsquares_row.cols - 1);
//
//	//int r[2] = { 0, 20 };
//	//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
//	//imshow_range("Display window", subsquares_row, r);                   // Show our image inside it.
//	//waitKey(0);
//
//	subsquares_row.copyTo(*ret_Mat_row);
//	subsquares_col.copyTo(*ret_Mat_col);
//
//}
//
//__global__ void Make_Zero(uchar* src, uchar* dst, const int step)
//{
//	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
//	const int idy = threadIdx.y + blockIdx.y * blockDim.y;
//	uchar* loc_src = src + step*idy + idx;
//	uchar* loc_dst = dst + step*idy + idx;
//	if (blockIdx.y == blockIdx.x)
//		*loc_dst = 0;
//	else
//		*loc_dst = *loc_src;
//}
//
//void make_pinv_im(float sigma, Mat* subsquares_row, Mat* subsquares_col, Mat* pinv_im, float* pattern, vector<int>* ss_row_start, vector<int>* ss_row_end, vector<int>* ss_col_start, vector<int>* ss_col_end) {
//	int im_rows = subsquares_row->rows;
//	int im_cols = subsquares_row->cols;
//	int ss_rows = subsquares_row->at<uint16_t>(subsquares_row->rows - 1, 0) + 1;
//	int ss_cols = subsquares_col->at<uint16_t>(0, subsquares_col->cols - 1) + 1;
//	float ss_center_row;
//	float ss_center_col;
//	Mat ss;
//	Mat ss_aux;
//	Mat ss_vec_inv;
//	Mat temp_pinv_im(im_rows, im_cols, DataType<float>::type);
//
//	for (int i = 0; i < ss_rows; i++) {
//		for (int j = 0; j < ss_cols; j++) {
//			ss = temp_pinv_im(Range(ss_row_start->at(i), ss_row_end->at(i) + 1), Range(ss_col_start->at(j), ss_col_end->at(j) + 1));
//			ss_center_row = pattern[0] + i * pattern[2] - ss_row_start->at(i);
//			ss_center_col = pattern[1] + j * pattern[3] - ss_col_start->at(j);
//			for (int k = 0; k < ss.rows; k++) {
//				for (int l = 0; l < ss.cols; l++) {
//					ss.at<float>(k, l) = exp(-(pow(k - ss_center_row, 2) + pow(l - ss_center_col, 2)) / (2 * pow(sigma, 2)));
//				}
//			}
//			ss.copyTo(ss_aux);
//			ss_vec_inv = ss_aux.reshape(0, 1).inv(cv::DECOMP_SVD);
//			ss_vec_inv.reshape(0, ss.rows).copyTo(ss);
//		}
//	}
//	temp_pinv_im.copyTo(*pinv_im);
//}
//
//__global__ void pixelwise_signal_extraction(uchar* dev_data, 
//	uchar* dev_coeff, 
//	uchar* dev_ss_row,
//	uchar* dev_ss_col,
//	uchar* dev_pinv_im,
//	int im_rows, 
//	int im_cols, 
//	const int data_step, 
//	const int ss_row_step, 
//	const int ss_col_step,
//	const int pinv_im_step, 
//	const int coeff_step) {
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
//		uint16_t* pix_loc_data = (uint16_t*)(dev_data + data_step*idy + data_elem_size*idx);
//		uint16_t* pix_loc_ss_row = (uint16_t*)(dev_ss_row + ss_row_step*idy + ss_elem_size*idx);
//		uint16_t* pix_loc_ss_col = (uint16_t*)(dev_ss_col + ss_col_step*idy + ss_elem_size*idx);
//		float* pix_loc_pinv_im = (float*)(dev_pinv_im + pinv_im_step*idy + pinv_elem_size*idx);
//
//		float* pix_loc_coeff = (float*)(dev_coeff + coeff_step * *pix_loc_ss_row + coeff_elem_size * *pix_loc_ss_col);
//		prod = (float)*pix_loc_data * *pix_loc_pinv_im;
//		//printf("Idx = %d, Idy = %d, coeff_idx = %d, coeff_idy = %d, data = %d, pinv_v = %f, proKCd in kernel is: %f\n", idx, idy, *pix_loc_ss_col, *pix_loc_ss_row, *pix_loc_data, *pix_loc_pinv_im, prod);
//		atomicAdd(pix_loc_coeff, prod);
//	}
//
//}
//
//
//
//LIBDLL void extract_signal(int im_rows, int im_cols, uchar* rawdata_ptr, uchar* ss_row_ptr, uchar* ss_col_ptr, uchar* pinv_im_ptr, uchar* coeffs_ptr, float* elapsed) {
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	Mat rawdata(im_rows, im_cols, CV_16U, rawdata_ptr);
//	Mat ss_row(im_rows, im_cols, CV_16U, ss_row_ptr);
//	Mat ss_col(im_rows, im_cols, CV_16U, ss_col_ptr);
//	Mat pinv_im(im_rows, im_cols, CV_32F, pinv_im_ptr);
//	int coeff_rows = ss_row.at<uint16_t>(ss_row.rows - 1, 0) + 1;
//	int coeff_cols = ss_col.at<uint16_t>(0, ss_col.cols - 1) + 1;
//	Mat coeffs(coeff_rows, coeff_cols, CV_32F, coeffs_ptr);
//	
//	dim3 block(32, 32);
//	dim3 grid((im_cols + block.x - 1) / block.x, (im_rows + block.y - 1) / block.y);
//
//	cuda::GpuMat dev_ss_row;
//	cuda::GpuMat dev_ss_col;
//	cuda::GpuMat dev_pinv_im;
//
//	dev_ss_row.upload(ss_row);
//	dev_ss_col.upload(ss_col);
//	dev_pinv_im.upload(pinv_im);
//
//	cuda::GpuMat dev_data;
//	cuda::GpuMat dev_coeff(coeff_rows, coeff_cols, CV_32F);
//	cudaEventRecord(start);
//	dev_data.upload(rawdata);
//	cudaDeviceSynchronize();
//	cudaEventRecord(stop);
//	pixelwise_signal_extraction << < grid, block >> > (dev_data.data, dev_coeff.data, dev_ss_row.data, dev_ss_col.data, dev_pinv_im.data, im_rows, im_cols, dev_data.step, dev_ss_row.step, dev_ss_col.step, dev_pinv_im.step, dev_coeff.step);
//	cudaDeviceSynchronize();
//	
//	dev_coeff.download(coeffs);
//
//	cudaEventElapsedTime(elapsed, start, stop);
//}
//
//LIBDLL void extract_signal_stack(int im_rows, int im_cols, int im_slices, uchar** rawdata_ptr, uchar* ss_row_ptr, uchar* ss_col_ptr, uchar* pinv_im_ptr, uchar** coeffs_ptr, float* elapsed) {
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	int ndims = 3;
//	int sizes[3] = { im_rows, im_cols, im_slices };
//
//	Mat ss_row(im_rows, im_cols, CV_16U, ss_row_ptr);
//	Mat ss_col(im_rows, im_cols, CV_16U, ss_col_ptr);
//	Mat pinv_im(im_rows, im_cols, CV_32F, pinv_im_ptr);
//	int coeff_rows = ss_row.at<uint16_t>(ss_row.rows - 1, 0) + 1;
//	int coeff_cols = ss_col.at<uint16_t>(0, ss_col.cols - 1) + 1;
//
//
//	dim3 block(32, 32);
//	dim3 grid((im_cols + block.x - 1) / block.x, (im_rows + block.y - 1) / block.y);
//
//	cuda::GpuMat dev_ss_row;
//	cuda::GpuMat dev_ss_col;
//	cuda::GpuMat dev_pinv_im;
//
//	dev_ss_row.upload(ss_row);
//	dev_ss_col.upload(ss_col);
//	dev_pinv_im.upload(pinv_im);
//
//	cuda::GpuMat dev_data;
//	cuda::GpuMat dev_coeff(coeff_rows, coeff_cols, CV_32F);
//
//
//	for (int i = 0; i < im_slices; i++) {
//		Mat rawdata(im_rows, im_cols, CV_16U, rawdata_ptr[i]);
//		Mat coeffs(coeff_rows, coeff_cols, CV_32F, coeffs_ptr[i]);
//		dev_data.upload(rawdata);
//
//		pixelwise_signal_extraction << < grid, block >> > (dev_data.data, dev_coeff.data, dev_ss_row.data, dev_ss_col.data, dev_pinv_im.data, im_rows, im_cols, dev_data.step, dev_ss_row.step, dev_ss_col.step, dev_pinv_im.step, dev_coeff.step);
//		cudaDeviceSynchronize();
//
//		dev_coeff.download(coeffs);
//	}
//	cudaEventElapsedTime(elapsed, start, stop);
//}
//
//LIBDLL void init_sig_extraction(int im_rows, int im_cols, int* coeff_rows_ptr, int* coeff_cols_ptr, uchar* ss_row_ptr, uchar* ss_col_ptr, uchar* pinv_im_ptr, float* pattern_ptr, float sigma_px) {
//
//	Mat ss_row(im_rows, im_cols, CV_16U, ss_row_ptr);
//	Mat ss_col(im_rows, im_cols, CV_16U, ss_col_ptr);
//
//	Mat pinv_im(im_rows, im_cols, CV_32F, pinv_im_ptr);
//
//	vector<int> ss_row_start;
//	vector<int> ss_row_end;
//	vector<int> ss_col_start;
//	vector<int> ss_col_end;
//
//	make_subsquares(im_rows, im_cols, pattern_ptr, &ss_row, &ss_col, &ss_row_start, &ss_row_end, &ss_col_start, &ss_col_end);
//	make_pinv_im(sigma_px, &ss_row, &ss_col, &pinv_im, pattern_ptr, &ss_row_start, &ss_row_end, &ss_col_start, &ss_col_end);
//
//	*coeff_rows_ptr = ss_row.at<uint16_t>(ss_row.rows - 1, 0) + 1;
//	*coeff_cols_ptr = ss_col.at<uint16_t>(0, ss_col.cols - 1) + 1;
//
//}
//
//
//LIBDLL unsigned char* GPU_image_proc(unsigned char *dataImg, unsigned char *resImg, int im_rows, int im_cols/*, float* pattern*/)
//{
//	//cout << cv::getBuildInformation() << endl;
//	//*out = in;
//	Mat image2;
//	Mat image = imread("Lena.jpg", CV_LOAD_IMAGE_ANYDEPTH);   // Read the file
//	if (!image.data)                              // Check for invalid input
//	{
//		cout << "Could not open or find the image" << std::endl;
//	}
//	if (image.empty())
//		std::cout << "failed to open img.jpg" << std::endl;
//	else
//		std::cout << "img.jpg loaded OK" << std::endl;
//
//	cuda::GpuMat src;
//	src.upload(image);
//	cuda::GpuMat dst(src.rows, src.cols, 0); //Initialize with same size as src, type is OpenCV specific numbers specifying data type, 0 is 8 bit uint.
//
//	dim3 block(32, 32);
//	dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
//
//	Make_Zero << <grid, block >> > (src.data, dst.data, src.step);
//	cudaDeviceSynchronize();
//
//	dst.download(image2);
//	Mat im(512, 512, CV_8UC1, resImg);
//	image2.copyTo(im);
//
//	//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
//	//imshow("Display window", image2);                   // Show our image inside it.
//
//	//waitKey(0);                                          // Wait for a keystroke in the window
//	return image.data;
//}
//
//int main_test() {
//
//	Mat image_raw = imread("04_pr_slice2.tif", CV_LOAD_IMAGE_ANYDEPTH);   // Read the file
//	Mat image(image_raw.rows, image_raw.cols, CV_16U);
//	image_raw.convertTo(image, CV_16U);
//	if (!image.data)                              // Check for invalid input
//	{
//		cout << "Could not open or find the image" << std::endl;
//	}
//	if (image.empty())
//		std::cout << "failed to open img.jpg" << std::endl;
//	else
//		std::cout << "img.jpg loaded OK" << std::endl;
//
//	int im_rows = image.rows;
//	int im_cols = image.cols;
//	int im_slices = 1;
//
//
//	//Mat res(im_rows, im_cols, CV_8UC1);
//	//Mat data(im_rows, im_cols, CV_16UC1, dataImg);
//	//data.copyTo(res);
//
//	Mat ss_row(im_rows, im_cols, CV_16U);
//	Mat ss_col(im_rows, im_cols, CV_16U);
//	Mat pinv_im(im_rows, im_cols, CV_32F);
//	vector<int> ss_row_start;
//	vector<int> ss_row_end;
//	vector<int> ss_col_start;
//	vector<int> ss_col_end;
//
//	int coeff_rows;
//	int coeff_cols;
//	float pattern[4] = { 0.0f, 0.0f, 16.0f, 16.0f };
//	float sigma = 3.2;
//	float elapsed;
//	
//	init_sig_extraction(im_rows, im_cols, &coeff_rows, &coeff_cols, (uchar*)ss_row.data, (uchar*)ss_col.data, (uchar*)pinv_im.data, pattern, sigma);
//
//	Mat coeffs(coeff_rows, coeff_cols, CV_32F);
//	
//	extract_signal_stack(im_rows, im_cols, im_slices, (uchar**)&image.data, (uchar*)ss_row.data, (uchar*)ss_col.data, (uchar*)pinv_im.data, (uchar**)&coeffs.data, &elapsed);
//	//coeffs.copyTo(res);
//	//for(int i = 0; i < res.rows; i++) {
//	//	for (int j = 0; j < res.cols; j++) {
//	//		printf("Coeff value is: %f\n", res.at<float>(i, j));
//	//	}
//	//}
////
////	Mat res_8bit(res.rows, res.cols, CV_8U);
////	res.convertTo(res_8bit, CV_8U);
////
//	int r[2] = { 0, 500 };
//	printf("Coeffs(1,1) = %f", coeffs.at<float>(1, 1));
//	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
//	imshow_range("Display window", coeffs, r);                   // Show our image inside it.
//	waitKey(0);
//	//int r[2] = { 0, 50 };
//	//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
//	//imshow_range("Display window", ss_col(Range(0,250), Range(0,250)), r);                   // Show our image inside it.
//
//	//waitKey(0);
//
//
//	//const int arraySize = 5;
//	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
//	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
//	//int c[arraySize] = { 0 };
//
//	//// Add vectors in parallel.
//	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//	//if (cudaStatus != cudaSuccess) {
//	//    fprintf(stderr, "addWithCuda failed!");
//	//    return 1;
//	//}
//
//	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//	//    c[0], c[1], c[2], c[3], c[4]);
//
//	//// cudaDeviceReset must be called before exiting in order for profiling and
//	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
//	//cudaStatus = cudaDeviceReset();
//	//if (cudaStatus != cudaSuccess) {
//	//    fprintf(stderr, "cudaDeviceReset failed!");
//	//    return 1;
//	//}
//
//	return 0;
//}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//// Helper function for using CUDA to add vectors in parallel.
////cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
////{
////    int *dev_a = 0;
////    int *dev_b = 0;
////    int *dev_c = 0;
////    cudaError_t cudaStatus;
////
////    // Choose which GPU to run on, change this on a multi-GPU system.
////    cudaStatus = cudaSetDevice(0);
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
////        goto Error;
////    }
////
////    // Allocate GPU buffers for three vectors (two input, one output)    .
////    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMalloc failed!");
////        goto Error;
////    }
////
////    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMalloc failed!");
////        goto Error;
////    }
////
////    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMalloc failed!");
////        goto Error;
////    }
////
////    // Copy input vectors from host memory to GPU buffers.
////    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMemcpy failed!");
////        goto Error;
////    }
////
////    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMemcpy failed!");
////        goto Error;
////    }
////
////    // Launch a kernel on the GPU with one thread for each element.
////    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
////
////    // Check for any errors launching the kernel
////    cudaStatus = cudaGetLastError();
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
////        goto Error;
////    }
////    
////    // cudaDeviceSynchronize waits for the kernel to finish, and returns
////    // any errors encountered during the launch.
////    cudaStatus = cudaDeviceSynchronize();
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
////        goto Error;
////    }
////
////    // Copy output vector from GPU buffer to host memory.
////    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
////    if (cudaStatus != cudaSuccess) {
////        fprintf(stderr, "cudaMemcpy failed!");
////        goto Error;
////    }
////
////Error:
////    cudaFree(dev_c);
////    cudaFree(dev_a);
////    cudaFree(dev_b);
////    
////    return cudaStatus;
////}
//
