/***************************************************************************
                          vectorTestKernel.cu
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

#ifndef _SAT_KERNEL_H_
#define _SAT_KERNEL_H_

/// /////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! Each element is multiplied with the number of threads / array length
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
//! @param size     size of a row in data
/// /////////////////////////////////////////////////////////////////////////////
__global__ void SATKernelNaive( unsigned int* g_idata, unsigned int* g_odata,unsigned int size) {
	// TO DO
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	int sum = 0;
	for(unsigned int x = 0; x <= idx; ++x) {
		for(unsigned int y = 0; y <= idy; ++y) {
			sum += g_idata[x + y*size];
		}
	}
	g_odata[idx + idy*size] = sum;
}

/// /////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! Each element is multiplied with the number of threads / array length
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
//! @param size     size of a row in data
/// /////////////////////////////////////////////////////////////////////////////
__global__ void SATKernel( unsigned int* g_idata, unsigned int* g_odata,unsigned int size) {
	// shared memory
	__shared__  unsigned int sdata[32*32]; // Enough data must be stored
	// TO DO
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	sdata[idx + idy*size] = g_idata[idx + idy*size];
	__syncthreads();

	int sum = 0;
	for(unsigned int x = 0; x <= idx; ++x) {
		for(unsigned int y = 0; y <= idy; ++y) {
			sum += sdata[x + y*size];
		}
	}
	g_odata[idx + idy*size] = sum;
}

//get grey scale image from an RVB image
__global__ void LumKernel( unsigned int* g_idata, unsigned int* g_odata) {
	int idThreadX = threadIdx.x;
	int idThreadY = threadIdx.y;
	int idBlockX = blockIdx.x;
	int idBlockY = blockIdx.y;

	int nbBlockX = gridDim.x;
	int nbThreadPerBlockX = blockDim.x;
	int nbThreadPerBlockY = blockDim.y;

	int indiceX = 3 * ( idThreadX + idBlockX * nbThreadPerBlockX);
	int indiceY = 3 * ( idThreadY + idBlockY * nbThreadPerBlockY);
	int width = nbBlockX * nbThreadPerBlockX;
	int indice = indiceX + indiceY * width;

	if(indice > 0 || indice <= 3*1600*1064) {
		unsigned char r = g_idata[indice];
		unsigned char g = g_idata[indice + 1];
		unsigned char b = g_idata[indice + 2];
		// compute luminance
		unsigned char luminance = 0.3 * r + 0.59 * g + 0.11 * b;
		
		g_odata[indice] = luminance;
		g_odata[indice + 1] = luminance;
		g_odata[indice + 2] = luminance;
	}
}

//get histogramme of a grey scale image
//use atomic operation
__global__ void HistoKernel( unsigned int* g_idata, unsigned int* d_oHisto) {
	int idThreadX = threadIdx.x;
	int idThreadY = threadIdx.y;
	int idBlockX = blockIdx.x;
	int idBlockY = blockIdx.y;

	int nbBlockX = gridDim.x;
	int nbThreadPerBlockX = blockDim.x;
	int nbThreadPerBlockY = blockDim.y;

	int indiceX = idThreadX + idBlockX * nbThreadPerBlockX;
	int indiceY = idThreadY + idBlockY * nbThreadPerBlockY;
	int width = nbBlockX * nbThreadPerBlockX;
	int indice = indiceX + indiceY * width;

	unsigned char luminance = g_idata[indice]; //[0; 255]

	atomicAdd(&d_oHisto[luminance], 1);
}

//get gauss 5*5 from a grey scale image
//use shared memory
__global__ void GaussKernel( unsigned int* g_idata, unsigned int* g_odata) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int blockx = blockIdx.x;
	int blocky = blockIdx.y;
	int blockDimx = blockDim.x;
	int blockDimy = blockDim.y;

	__shared__ unsigned int sdata[16*16];
	
	int gaussSize = 2;

	sdata[idx + idy*blockDimx] = g_idata[(idx + blockx*(blockDimx-gaussSize*blockx)) + (idy*(blockDimy-gaussSize*blocky) + blocky*(blockDimy-gaussSize*blocky))];
	__syncthreads();

	int gauss = 0;
	int nbPixel = 0;
	for(unsigned int x = idx-gaussSize; x <= idx+gaussSize; ++x) {
		for(unsigned int y = idy-gaussSize; y <= idy+gaussSize; ++y) {
			if (x > 0 && x < blockDimx && y >0 && y < blockDimy) {
				gauss += sdata[x + y*blockDimx];
				++nbPixel;
			}
		}
	}
	g_odata[(idx + blockx*(blockDimx-2*blockx)) + (idy*(blockDimy-2*blocky) + blocky*(blockDimy-2*blocky))] = gauss / nbPixel;
}


#endif // #ifndef _SAT_KERNEL_H_
