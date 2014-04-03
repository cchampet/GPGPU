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
#include "SATKernel.cu"
#include "main.hpp"


/// /////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
/// /////////////////////////////////////////////////////////////////////////////
float runTest(unsigned int *idata, unsigned int* odata, unsigned int* oHisto, unsigned int sz_thread_per_block, unsigned int num_blocksx, unsigned int num_blocksy) {
	// TO DO : compute size...
	unsigned int width = num_blocksx * sz_thread_per_block ;
	unsigned int height = num_blocksy * sz_thread_per_block;

	unsigned int mem_size = 3*width*height*sizeof(unsigned int);
	// Génération des timers ...
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate device memory
	unsigned int* d_idata;
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
	unsigned int* d_odata;
	if (!(cudaMalloc( (void**) &d_odata, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// allocate device memory for historgramme result
	unsigned int* d_oHisto;
	if (!(cudaMalloc( (void**) &d_oHisto, 256*sizeof(unsigned int)) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}
	// copy host memory to device
	if (!(cudaMemcpy( d_oHisto, oHisto, 256*sizeof(unsigned int), cudaMemcpyHostToDevice) == cudaSuccess)) {
		std::cerr<<"Erreur copying memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// setup execution parameters
	dim3  grid( num_blocksx, num_blocksy, 1);
	dim3  threads( sz_thread_per_block, sz_thread_per_block, 1);

	// execute the kernels
	cudaEventRecord(start, 0);
	LumKernel<<< grid, threads >>>( d_idata, d_odata);
	HistoKernel<<< grid, threads >>>( d_odata, d_oHisto);
	cudaEventRecord(stop, 0);

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed","SAT.cu",135);
	cudaEventSynchronize(stop);

	// copy result from device to host
	cudaMemcpy( odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
	cudaMemcpy( oHisto, d_oHisto, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// GESTION DU TEMPS
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);cudaEventDestroy(stop);

	cudaFree(d_idata);
	cudaFree(d_odata);
	cudaFree(d_oHisto);

	cudaThreadExit();

	return elapsed_time;
}

float runGauss(unsigned int *idata, unsigned int* odata, unsigned int sz_thread_per_block, unsigned int num_blocksx, unsigned int num_blocksy) {
	// TO DO : compute size...
	unsigned int width = num_blocksx * sz_thread_per_block ;
	unsigned int height = num_blocksy * sz_thread_per_block;

	unsigned int mem_size = width*height*sizeof(unsigned int);
	// Génération des timers ...
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate device memory
	unsigned int* d_idata;
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
	unsigned int* d_odata;
	if (!(cudaMalloc( (void**) &d_odata, mem_size) == cudaSuccess)) {
		std::cerr<<"Erreur allocating memory in device ! Aborting!"<<std::endl;
		exit(1);
	}

	// setup execution parameters
	dim3  grid( num_blocksx, num_blocksy, 1);
	dim3  threads( sz_thread_per_block, sz_thread_per_block, 1);

	// execute the kernel
	cudaEventRecord(start, 0);
	GaussKernel<<< grid, threads >>>( d_idata, d_odata);
	cudaEventRecord(stop, 0);

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed","SAT.cu",135);
	cudaEventSynchronize(stop);

	// copy result from device to host
	cudaMemcpy( odata, d_odata, mem_size, cudaMemcpyDeviceToHost);

	// GESTION DU TEMPS
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);cudaEventDestroy(stop);

	cudaFree(d_idata);
	cudaFree(d_odata);

	cudaThreadExit();

	return elapsed_time;
}
