/***************************************************************************
															blank.cu
-------------------
begin                : Mon Oct 20 2010
email                : biri@univ-mlv.fr
***************************************************************************/

/***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************/

// includes, system
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>

// includes, project
#include "cudaSimpleUtils.hpp"
#include "imageTexture.hpp"

// includes, kernels
#include "blankKernel.cu"

const unsigned int NT = 20;
const unsigned int NB = 20;
const unsigned int N = NT * NB;

/// /////////////////////////////////////////////////////////////////////////////
//! Run a blank test for CUDA
//! renvoie le temps d'execution du kernel
/// /////////////////////////////////////////////////////////////////////////////
float runTest(float * h_src, float * h_dest_gpu) {
	// Variable initialization
	float * d_src;
	float * d_dest_gpu;
	const size_t ARRAY_BYTES = sizeof(float) * size_t(N);
	
	cudaMalloc((void**)&d_src, ARRAY_BYTES);
	cudaMalloc((void**)&d_dest_gpu, ARRAY_BYTES);
	
	// Get data from CPU
	cudaMemcpy(d_src, h_src, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dest_gpu, h_dest_gpu, ARRAY_BYTES, cudaMemcpyHostToDevice);


	// setup execution parameters
	dim3  grid( 1, 1, 1);
	dim3  threads( 128, 1, 1);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// execute the kernel
	blankKernel<<< grid, threads>>>(d_src, d_dest_gpu);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float d_et;
	cudaEventElapsedTime(&d_et, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Send data to CPU
	cudaMemcpy(h_dest_gpu, d_dest_gpu, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(d_src);
	cudaFree(d_dest_gpu);

	return d_et;
}

float runLuminance(unsigned char* const h_rgbaImage, unsigned char* const h_greyImage, int numRows, int numCols) {

	unsigned char* d_src;
	unsigned char* d_res;

	const unsigned int threadDimX = 16;
	const unsigned int threadDimY = 16;
	const unsigned int blockDimX = numRows/threadDimX;
	const unsigned int blockDimY = numCols/threadDimY;

	const size_t ARRAY_BYTES = sizeof(unsigned char) * size_t(3*numRows*numCols);

	cudaMalloc((void**)&d_src, ARRAY_BYTES);
	cudaMalloc((void**)&d_res, ARRAY_BYTES);

	cudaMemcpy(d_src, h_rgbaImage, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, h_greyImage, ARRAY_BYTES, cudaMemcpyHostToDevice);

	dim3  grid( blockDimX, blockDimY, 1);
	dim3  threads( threadDimX, threadDimY, 1);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	luminanceKernel<<<grid, threads>>>(d_src, d_res, numRows, numCols);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float d_et;
	cudaEventElapsedTime(&d_et, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(h_greyImage, d_res, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_res);

	return d_et;
}

float* runAdditionOf2Matrix(float* h_mat1, float* h_mat2) {

	float* d_mat1;
	float* d_mat2;
	float* d_matRes;

	const unsigned int threadDimX = 3;
	const unsigned int threadDimY = 3;

	const size_t ARRAY_BYTES = sizeof(float) * 9;

	cudaMalloc((void**)&d_mat1, ARRAY_BYTES);
	cudaMalloc((void**)&d_mat2, ARRAY_BYTES);
	cudaMalloc((void**)&d_matRes, ARRAY_BYTES);

	cudaMemcpy(d_mat1, h_mat1, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat2, h_mat2, ARRAY_BYTES, cudaMemcpyHostToDevice);

	dim3  grid(1, 1, 1);
	dim3  threads(threadDimX, threadDimY, 1);
	additionOf2MatrixKernel<<<grid, threads>>>(d_mat1, d_mat2, d_matRes);

	float* h_matRes = new float[9];
	cudaMemcpy(h_matRes, d_matRes, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_mat1);
	cudaFree(d_mat2);
	cudaFree(d_matRes);

	return h_matRes;
}
