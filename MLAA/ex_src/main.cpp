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
/// Program main
/// /////////////////////////////////////////////////////////////////////////////
// ex2
int main( int argc, char** argv) {
	std::cerr<<"START ..."<<sizeof(unsigned int)<<std::endl;
	
	// Variable initialization
	unsigned int width = 0;
	unsigned int height = 0;
	// Reading the input data (using imageTexture)
	ImageTex* data = new ImageTex(0,"data/reference_noAA.ppm");

	width = data->tailu;
	height = data->tailv;
	std::cerr<<"Size of the image : "<< width << ", " << height << std::endl;
	
	// Transform data to unsigned int : OK
	unsigned char *inputData = new unsigned char[width*height*3];

	for(unsigned int i=0; i<width*height*3; ++i) {
		inputData[i] = data->getTab()[i];
	}

	delete(data);
	
	float timer_gpu = 0.;
	float timer_gpu_comp = 0.;
	float timer_cpu = 0.;
	tman = new TimerManager();
	id_timer_1 = tman->addOneTimer(1);
	id_timer_2 = tman->addOneTimer(1);

	// Define threads and blocks
	// For discontinuity
	unsigned int sz_thread_per_block_discontinuity = 32;
	unsigned int num_blocksx_discontinuity = width / sz_thread_per_block_discontinuity;
	unsigned int num_blocksy_discontinuity = height / sz_thread_per_block_discontinuity;
	// For horizontal doubling
	unsigned int sz_thread_per_blockx_h_doubling = width;
	unsigned int sz_thread_per_blocky_h_doubling = 1;
	unsigned int num_blocksx_h_doubling = 1;
	unsigned int num_blocksy_h_doubling = height;
	// For vertical doubling
	unsigned int sz_thread_per_blockx_v_doubling = 1;
	unsigned int sz_thread_per_blocky_v_doubling = height;
	unsigned int num_blocksx_v_doubling = width;
	unsigned int num_blocksy_v_doubling = 1;

	// Allocate array for output data 
	unsigned char *outputDataDiscontinuity = new unsigned char[width*height*3];
	unsigned char *outputDataRightHorizontalDoubling = new unsigned char[width*height*3];
	unsigned char *outputDataLeftHorizontalDoubling = new unsigned char[width*height*3];
	unsigned char *outputDataTopVerticalDoubling = new unsigned char[width*height*3];
	unsigned char *outputDataBottomVerticalDoubling = new unsigned char[width*height*3];
	unsigned char *outputDataBlending = new unsigned char[width*height*3];

	// Run on GPU
	tman->startOneExecution(id_timer_2);
	timer_gpu_comp = cudaRunDiscontinuity(inputData, outputDataDiscontinuity, sz_thread_per_block_discontinuity, num_blocksx_discontinuity, num_blocksy_discontinuity);
	timer_gpu_comp += cudaRunHorizontalDoubling(outputDataDiscontinuity, outputDataLeftHorizontalDoubling, outputDataRightHorizontalDoubling, sz_thread_per_blockx_h_doubling, sz_thread_per_blocky_h_doubling, num_blocksx_h_doubling, num_blocksy_h_doubling);
	timer_gpu_comp += cudaRunVerticalDoubling(outputDataDiscontinuity, outputDataTopVerticalDoubling, outputDataBottomVerticalDoubling, sz_thread_per_blockx_v_doubling, sz_thread_per_blocky_v_doubling, num_blocksx_v_doubling, num_blocksy_v_doubling);
	timer_gpu_comp += cudaRunBlending(inputData, outputDataLeftHorizontalDoubling, outputDataTopVerticalDoubling, outputDataBlending, sz_thread_per_block_discontinuity, sz_thread_per_block_discontinuity, num_blocksx_discontinuity, num_blocksy_discontinuity);
	tman->stopOneExecution(id_timer_2);
	timer_gpu = tman->getLastTime(id_timer_2);

	// Save the results
	ImageTex* imageResDiscontinuity = new ImageTex(0, 0, width, height, outputDataDiscontinuity);
	imageResDiscontinuity->saveToPXM("data/reference_noAA_discontinuity.ppm");
	
	ImageTex* imageResRightHorizontalDoubling = new ImageTex(0, 0, width, height, outputDataRightHorizontalDoubling);
	imageResRightHorizontalDoubling->saveToPXM("data/reference_noAA_h_r_doubling.ppm");

	ImageTex* imageResLeftHorizontalDoubling = new ImageTex(0, 0, width, height, outputDataLeftHorizontalDoubling);
	imageResLeftHorizontalDoubling->saveToPXM("data/reference_noAA_h_l_doubling.ppm");

	ImageTex* imageResTopVerticalDoubling = new ImageTex(0, 0, width, height, outputDataTopVerticalDoubling);
	imageResTopVerticalDoubling->saveToPXM("data/reference_noAA_v_t_doubling.ppm");

	ImageTex* imageResBottomVerticalDoubling = new ImageTex(0, 0, width, height, outputDataBottomVerticalDoubling);
	imageResBottomVerticalDoubling->saveToPXM("data/reference_noAA_v_b_doubling.ppm");

	ImageTex* imageResBlending = new ImageTex(0, 0, width, height, outputDataBlending);
	imageResBlending->saveToPXM("data/reference_noAA_blending.ppm");

	//Test
	// for(unsigned int i=0; i<3*width*height;++i){
	// 	if(outputDataDoubling[i])
	// 		printf("outputDataDoubling : %d\n",(int)outputDataDoubling[i]);
	// }
	
	if (inputData) delete[](inputData);
	if (outputDataDiscontinuity) delete[](outputDataDiscontinuity);
	if (outputDataRightHorizontalDoubling) delete[](outputDataRightHorizontalDoubling);
	if (outputDataLeftHorizontalDoubling) delete[](outputDataLeftHorizontalDoubling);
	if (outputDataTopVerticalDoubling) delete[](outputDataTopVerticalDoubling);
	if (outputDataBottomVerticalDoubling) delete[](outputDataBottomVerticalDoubling);
	if (outputDataBlending) delete[](outputDataBlending);

	std::cerr<<"... DONE"<<std::endl;
	std::cerr<<"TIMING ARE : "<<std::endl;
	std::cerr<<"\tGPU : "<<timer_gpu_comp<<" ("<<timer_gpu<<")"<<std::endl;
	std::cerr<<"\tCPU : "<<timer_cpu<<std::endl;
}

