/***************************************************************************
														blankKernek.cu
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

#ifndef _BLANK_KERNEL_H_
#define _BLANK_KERNEL_H_

#include "imageTexture.hpp"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! Each element is multiplied with the number of threads / array length
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

__global__ void blankKernel(float * d_src, float * d_dest_gpu) {
	//__shared__ float s[20];
	int idx = threadIdx.x;
	float f = d_src[idx];
	//__syncthreads();
	d_dest_gpu[idx] = f*sin(f);
}

__global__ void luminanceKernel(unsigned char* const d_rgbaImage, unsigned char* const d_greyImage, int numRows, int numCols) {
	// Luminance = 0.3 R + 0.59 G + 0.11 B
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

	unsigned char r = d_rgbaImage[indice];
	unsigned char g = d_rgbaImage[indice + 1];
	unsigned char b = d_rgbaImage[indice + 2];
	unsigned char luminance = 0.3 * r + 0.59 * g + 0.11 * b;
	
	d_greyImage[indice] = luminance;
	d_greyImage[indice + 1] = luminance;
	d_greyImage[indice + 2] = luminance;
}

__global__ void additionOf2MatrixKernel(float * d_mat1, float * d_mat2, float * d_matRes) {
	int idX = threadIdx.x;
	int idY = threadIdx.y;

	d_matRes[idX + idY * 3] = d_mat1[idX + idY * 3] + d_mat2[idX + idY * 3];
}

#endif // #ifndef _BLANK_KERNEL_H_
