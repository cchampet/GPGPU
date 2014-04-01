/***************************************************************************
                              vectorTest.cu
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
#include <cudaSimpleUtils.hpp>

// includes, kernels
#include "MLAAKernel.cu"
#include "main.hpp"


/// /////////////////////////////////////////////////////////////////////////////
//! Run CUDA
/// /////////////////////////////////////////////////////////////////////////////
float cudaRunDiscontinuity(unsigned char *idata, unsigned char* odata, unsigned int sz_thread_per_block, unsigned int num_blocksx, unsigned int num_blocksy) {
	// compute size
	unsigned int width = num_blocksx * sz_thread_per_block ;
	unsigned int height = num_blocksy * sz_thread_per_block;

	unsigned int mem_size = 3*width*height*sizeof(unsigned char);
	unsigned int mem_size_buffer = 3*width*height*sizeof(float);
	// Génération des timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate device memory
	unsigned char* d_idata;
	if (!(cudaMalloc( (void**) &d_idata, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	// copy host memory to device
	if (!(cudaMemcpy( d_idata, idata, mem_size, cudaMemcpyHostToDevice) == cudaSuccess)) {
		std::cerr<<"Erreur copying memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// allocate device memory for result
	float* d_odataXYZ;
	if (!(cudaMalloc( (void**) &d_odataXYZ, mem_size_buffer) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	float* d_odataLab;
	if (!(cudaMalloc( (void**) &d_odataLab, mem_size_buffer) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	unsigned char* d_odata;
	if (!(cudaMalloc( (void**) &d_odata, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// setup execution parameters
	dim3  grid( num_blocksx, num_blocksy, 1);
	dim3  threads( sz_thread_per_block, sz_thread_per_block, 1);

	// execute the kernels
	cudaEventRecord(start, 0);
	DetectDiscontinuity<<< grid, threads >>>(d_idata, d_odataXYZ, d_odataLab, d_odata);
	cudaEventRecord(stop, 0);

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed","MLAA.cu",135);
	cudaEventSynchronize(stop);

	// copy result from device to host
	cudaMemcpy(odata, d_odata, mem_size, cudaMemcpyDeviceToHost);

	// GESTION DU TEMPS
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_idata);
	cudaFree(d_odataXYZ);
	cudaFree(d_odataLab);
	cudaFree(d_odata);

	cudaThreadExit();

	return elapsed_time;
}

float cudaRunHorizontalDoubling(unsigned char *idata, unsigned char* odata_l, unsigned char* odata_r, unsigned int sz_thread_per_blockx, unsigned int sz_thread_per_blocky, unsigned int num_blocksx, unsigned int num_blocksy) {
	// compute size
	unsigned int width = num_blocksx * sz_thread_per_blockx;
	unsigned int height = num_blocksy * sz_thread_per_blocky;

	unsigned int mem_size = 3*width*height*sizeof(unsigned char);
	// Génération des timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate device memory
	unsigned char* d_idata;
	if (!(cudaMalloc( (void**) &d_idata, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	// copy host memory to device
	if (!(cudaMemcpy( d_idata, idata, mem_size, cudaMemcpyHostToDevice) == cudaSuccess)) {
		std::cerr<<"Erreur copying memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// allocate device memory for result
	unsigned char* d_odata_l;
	if (!(cudaMalloc( (void**) &d_odata_l, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	unsigned char* d_odata_r;
	if (!(cudaMalloc( (void**) &d_odata_r, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// setup execution parameters
	dim3  grid( num_blocksx, num_blocksy, 1);
	dim3  threads( sz_thread_per_blockx, sz_thread_per_blocky, 1);

	// execute the kernels
	cudaEventRecord(start, 0);
	RecursiveDoublingHorizontal<<< grid, threads >>>(d_idata, d_odata_l, d_odata_r);
	cudaEventRecord(stop, 0);

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed","MLAA.cu",135);
	cudaEventSynchronize(stop);

	// copy result from device to host
	cudaMemcpy(odata_l, d_odata_l, mem_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(odata_r, d_odata_r, mem_size, cudaMemcpyDeviceToHost);

	// GESTION DU TEMPS
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_idata);
	cudaFree(d_odata_l);
	cudaFree(d_odata_r);

	cudaThreadExit();

	return elapsed_time;
}

float cudaRunVerticalDoubling(unsigned char *idata, unsigned char* odata_t, unsigned char* odata_b, unsigned int sz_thread_per_blockx, unsigned int sz_thread_per_blocky, unsigned int num_blocksx, unsigned int num_blocksy) {
	// compute size
	unsigned int width = num_blocksx * sz_thread_per_blockx;
	unsigned int height = num_blocksy * sz_thread_per_blocky;

	unsigned int mem_size = 3*width*height*sizeof(unsigned char);
	// Génération des timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate device memory
	unsigned char* d_idata;
	if (!(cudaMalloc( (void**) &d_idata, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	// copy host memory to device
	if (!(cudaMemcpy( d_idata, idata, mem_size, cudaMemcpyHostToDevice) == cudaSuccess)) {
		std::cerr<<"Erreur copying memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// allocate device memory for result
	unsigned char* d_odata_t;
	if (!(cudaMalloc( (void**) &d_odata_t, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	unsigned char* d_odata_b;
	if (!(cudaMalloc( (void**) &d_odata_b, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// setup execution parameters
	dim3  grid( num_blocksx, num_blocksy, 1);
	dim3  threads( sz_thread_per_blockx, sz_thread_per_blocky, 1);

	// execute the kernels
	cudaEventRecord(start, 0);
	RecursiveDoublingVertical<<< grid, threads >>>(d_idata, d_odata_t, d_odata_b);
	cudaEventRecord(stop, 0);

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed","MLAA.cu",135);
	cudaEventSynchronize(stop);

	// copy result from device to host
	cudaMemcpy(odata_t, d_odata_t, mem_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(odata_b, d_odata_b, mem_size, cudaMemcpyDeviceToHost);

	// GESTION DU TEMPS
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_idata);
	cudaFree(d_odata_t);
	cudaFree(d_odata_b);

	cudaThreadExit();

	return elapsed_time;
}

float cudaRunBlending(unsigned char *idata, unsigned char *idata_hl, unsigned char *idata_vt, unsigned char* odata, unsigned int sz_thread_per_blockx, unsigned int sz_thread_per_blocky, unsigned int num_blocksx, unsigned int num_blocksy) {
	// compute size
	unsigned int width = num_blocksx * sz_thread_per_blockx;
	unsigned int height = num_blocksy * sz_thread_per_blocky;

	unsigned int mem_size = 3*width*height*sizeof(unsigned char);
	unsigned int mem_size_weight = width*height*sizeof(float);
	// Génération des timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate device memory
	unsigned char* d_idata;
	if (!(cudaMalloc( (void**) &d_idata, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	// copy host memory to device
	if (!(cudaMemcpy( d_idata, idata, mem_size, cudaMemcpyHostToDevice) == cudaSuccess)) {
		std::cerr<<"Erreur copying memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// allocate device memory
	unsigned char* d_idata_hl;
	if (!(cudaMalloc( (void**) &d_idata_hl, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	// copy host memory to device
	if (!(cudaMemcpy( d_idata_hl, idata_hl, mem_size, cudaMemcpyHostToDevice) == cudaSuccess)) {
		std::cerr<<"Erreur copying memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	unsigned char* d_idata_vt;
	if (!(cudaMalloc( (void**) &d_idata_vt, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	// copy host memory to device
	if (!(cudaMemcpy( d_idata_vt, idata_vt, mem_size, cudaMemcpyHostToDevice) == cudaSuccess)) {
		std::cerr<<"Erreur copying memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// allocate device memory for result
	float* d_odata_wl;
	if (!(cudaMalloc( (void**) &d_odata_wl, mem_size_weight) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	float* d_odata_wt;
	if (!(cudaMalloc( (void**) &d_odata_wt, mem_size_weight) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	unsigned char* d_odata_blending_horizontal;
	if (!(cudaMalloc( (void**) &d_odata_blending_horizontal, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	unsigned char* d_odata;
	if (!(cudaMalloc( (void**) &d_odata, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// setup execution parameters
	dim3  grid( num_blocksx, num_blocksy, 1);
	dim3  threads( sz_thread_per_blockx, sz_thread_per_blocky, 1);

	// execute the kernels
	cudaEventRecord(start, 0);
	ComputeWeightHorizontal<<< grid, threads >>>(d_idata_hl, d_odata_wl);
	BlendingHorizontal<<< grid, threads >>>(d_idata, d_idata_hl, d_odata_wl, d_odata_blending_horizontal);
	
	ComputeWeightVertical<<< grid, threads >>>(d_idata_vt, d_odata_wt);
	BlendingVertical<<< grid, threads >>>(d_odata_blending_horizontal, d_idata_vt, d_odata_wt, d_odata);

	cudaEventRecord(stop, 0);

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed","MLAA.cu",135);
	cudaEventSynchronize(stop);

	// copy result from device to host
	cudaMemcpy(odata, d_odata, mem_size, cudaMemcpyDeviceToHost);

	// GESTION DU TEMPS
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_idata);
	cudaFree(d_idata_hl);
	cudaFree(d_idata_vt);
	cudaFree(d_odata_wl);
	cudaFree(d_odata_wt);
	cudaFree(d_odata_blending_horizontal);
	cudaFree(d_odata);

	cudaThreadExit();

	return elapsed_time;
}