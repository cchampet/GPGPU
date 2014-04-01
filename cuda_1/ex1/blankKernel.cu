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

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! Each element is multiplied with the number of threads / array length
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

__global__ void blankKernel(float * d_src, float * d_dest_gpu) {
	__shared__ float s[20];
	int idx = threadIdx.x;
	float f = d_src[idx];
	//__syncthreads();
	d_dest_gpu[idx] = f*sin(f);
}

#endif // #ifndef _BLANK_KERNEL_H_
