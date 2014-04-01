/***************************************************************************
												cudaSimpleUtils.hpp
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


inline void cutilCheckMsg( const char *errorMessage, const char *file, const int line ) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		std::cerr<<"cutilCheckMsg() CUTIL CUDA error : "<<errorMessage;
		std::cerr<<" : "<<cudaGetErrorString( err)<<std::endl;
		exit(1);
	}
	/*
	#ifdef _DEBUG_
	err = cudaThreadSynchronize();
	if( cudaSuccess != err) {
		std::cerr<<"cutilCheckMsg cudaThreadSynchronize error: ";
		std::cerr<<errorMessage" : "<< cudaGetErrorString( err)<<std::endl;
		exit(1);
	}
	#endif
	*/
}
