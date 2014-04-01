/***************************************************************************
                               main.cpp
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

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "main.hpp"

TimerManager* tman;
unsigned int id_timer_1;
unsigned int id_timer_2;

// TO DO : CHANGE THIS VALUE !!!
#define MAX_2D_ARRAY_SIZE	1024

/// /////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param size       size of one dimension of the square 2D array
/// /////////////////////////////////////////////////////////////////////////////
void computeGold(unsigned int* reference, unsigned int* idata, const unsigned int size) {
	size_t idx_i,idx_j,sum;
	for( unsigned int idx = 0; idx < size*size; ++idx)
	{
		idx_i = idx%size;
		idx_j = idx/size;
		sum = 0;
		for(size_t j = 0; j <= idx_j ; ++j) {
			for(size_t i = 0; i <= idx_i ; ++i) {
				sum += idata[j*size+i];
			}
		}
		reference[idx] = sum;
	}
}

/// /////////////////////////////////////////////////////////////////////////////
//! Compare two differents data set
//! @param reference  reference data, supposed to be the reference
//! @param data       input data to be compared
//! @param len        number of elements in reference / data
/// /////////////////////////////////////////////////////////////////////////////
bool compareData( const unsigned int* reference, const unsigned int* data, const unsigned int len) {
	bool result = true;
	unsigned int error_count = 0;
	float epsilon = 0.01f;
	float diff;
	bool comp;
	for( unsigned int i = 0; i < len; ++i) {
		if (reference[i]==0) {
			comp = (data[i]==0);
		}
		else {
			diff = (reference[i] - data[i])/reference[i];
			comp = (diff <= epsilon) && (diff >= -epsilon);
		}
		result &= comp;

		error_count += !comp;

		#ifdef _DEBUG
		if( ! comp)
		{
			std::cerr << "ERROR, i = " << i << ",\t "
			<< reference[i] << " / "
			<< data[i]
			<< " (reference / data)\n";
		}
		#endif
	}

	return result;
}

void blurSAT_CPU(unsigned int * tabRVB, unsigned int *outputData, int size, int offset) {
	int xmin,xmax,ymin,ymax;

	for(int j = 0; j < size; ++j) {
		for(int i = 0; i < size; ++i) {
			xmin = max(i-offset-1,0);
			xmax = min(size-1,i+offset);
			ymin = max(0,j-offset-1);
			ymax = min(size-1,j+offset);

			outputData[i + j*size] = - tabRVB[xmin+ymin*size] + tabRVB[xmax + ymin*size] + tabRVB[xmin + ymax*size] - tabRVB[xmax + ymax*size];
			outputData[i + j*size] /= (2 * offset)*(2 * offset);
		}

	}

}

unsigned int* histogramme(){
	ImageTex* data = new ImageTex(0,"data/halloween.jpg");

	unsigned int width = data->tailu;
	unsigned int height = data->tailv;

	unsigned int * histo = new unsigned int[256];

	for(unsigned int k = 0; k < 256; ++k) {
		histo[k] = 0;
	}

	for(unsigned int j = 0; j < height; ++j) {
		for(unsigned int i = 0; i < width; ++i) {

			unsigned int luminance = 0.3 * data->get(i, j, 0) + 0.59 * data->get(i, j, 1) + 0.11 * data->get(i, j, 2);
			histo[luminance]++;		

		}
	}
	return histo;
}


/// /////////////////////////////////////////////////////////////////////////////
/// Program main
/// /////////////////////////////////////////////////////////////////////////////
// ex2
int main( int argc, char** argv) {
	std::cerr<<"START ..."<<sizeof(unsigned int)<<std::endl;
	
	// Variable initialization
	unsigned int width = 0;
	unsigned int height = 0;
	// Reading the input data (using imageTexture)
	//ImageTex* data = new ImageTex(1,"data/Wink.pgm");
	ImageTex* data = new ImageTex(0,"data/halloween.jpg");

	width = data->tailu;
	height = data->tailv;
	std::cerr<<"Size of the image : "<< width << ", " << height << std::endl;
	
	// TO DO Transform data to unsigned int : OK
	unsigned int *inputData = new unsigned int[width*height*3];
	for(unsigned int i = 0; i < width*height*3; ++i) {
		inputData[i] = (unsigned int)data->getTab()[i];
	}

	delete(data);
	
	float timer_gpu = 0.;
	float timer_gpu_comp = 0.;
	float timer_cpu = 0.;
	tman = new TimerManager();
	id_timer_1 = tman->addOneTimer(1);
	id_timer_2 = tman->addOneTimer(1);
	// allocate host memory

	// initalize the memory

	// compute reference solution (on CPU)
	/*
	unsigned int *reference = new unsigned int[size*size];
	tman->startOneExecution(id_timer_1);
	computeGold( reference, inputData, size);
	tman->stopOneExecution(id_timer_1);
	timer_cpu = tman->getLastTime(id_timer_1);
	*/

	// Run on GPU
	// TO DO : compute sz_thread_per_block : OK
	//unsigned int sz_thread_per_block = size;
	unsigned int sz_thread_per_block = 8;
	unsigned int num_blocksx = width / sz_thread_per_block;
	unsigned int num_blocksy = height / sz_thread_per_block;
	// TO DO : allocate array for output data 
	unsigned int *outputData = new unsigned int[width*height*3];
	unsigned int *oHisto = new unsigned int[256];

	for(unsigned int k = 0; k < 256; ++k) {
		oHisto[k] = 0;
	}

	tman->startOneExecution(id_timer_2);
	timer_gpu_comp = runTest(inputData, outputData, oHisto, sz_thread_per_block, num_blocksx, num_blocksy);
	tman->stopOneExecution(id_timer_2);
	timer_gpu = tman->getLastTime(id_timer_2);

	unsigned int total = 0;
	for(unsigned int k = 0; k < 256; ++k) {
		std::cout << k << " : " << oHisto[k] << std::endl;
		total += oHisto[k];
	}
	std::cout << "total = " << total << std::endl; 

	// if (compareData(histogramme(), oHisto,256)) {
	// 	std::cerr<<"\tSUCCES !"<<std::endl;
	// 	// Print results...
	// }
	// else {
	// 	std::cerr<<"FAILURE !"<<std::endl;
	// }

	//apply blur CPU
	//unsigned int* outputDataBlur = new unsigned int[width*height];
	//blurSAT_CPU(outputData, outputDataBlur, size, 1);

	//save the result
	unsigned char* tabRVBRes = new unsigned char[width*height];

	for(unsigned int i = 0; i < width*height; i++) {
		tabRVBRes[i] = (unsigned char)outputData[3*i];
	}
	ImageTex* imageRes = new ImageTex(0, 2, width, height, tabRVBRes);
	imageRes->saveToPXM("halloween_lum.pgm");
	
	histogramme();

	//delete[](reference);
	if (inputData) delete[](inputData);
	if (outputData) delete[](outputData);

	std::cerr<<"... DONE"<<std::endl;
	std::cerr<<"TIMING ARE : "<<std::endl;
	std::cerr<<"\tGPU : "<<timer_gpu_comp<<" ("<<timer_gpu<<")"<<std::endl;
	std::cerr<<"\tCPU : "<<timer_cpu<<std::endl;
}

