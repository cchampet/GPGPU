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

float runLuminance(ImageTex* img_src,ImageTex* img_res) {

}
