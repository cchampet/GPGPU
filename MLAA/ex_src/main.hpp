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


#ifndef __MAIN_HPP
#define __MAIN_HPP

#include <iostream>
#include <cstdlib>

#include "imageTexture.hpp"
#include "TimerManager.hpp"
#define _DEBUG
extern TimerManager* tman;
extern unsigned int id_timer_1;
extern unsigned int id_timer_2;

/// interface to compute fonction !
float cudaRunDiscontinuity(unsigned char *idata, unsigned char* odata, unsigned int sz_thread_per_block, unsigned int num_blocksx, unsigned int num_blocksy);
float cudaRunHorizontalDoubling(unsigned char *idata, unsigned char* odata_l, unsigned char* odata_r, unsigned int sz_thread_per_blockx, unsigned int sz_thread_per_blocky, unsigned int num_blocksx, unsigned int num_blocksy);
float cudaRunVerticalDoubling(unsigned char *idata, unsigned char* odata_t, unsigned char* odata_b, unsigned int sz_thread_per_blockx, unsigned int sz_thread_per_blocky, unsigned int num_blocksx, unsigned int num_blocksy);
float cudaRunBlending(unsigned char *idata, unsigned char *idata_hl, unsigned char *idata_vt, unsigned char* odata, unsigned int sz_thread_per_blockx, unsigned int sz_thread_per_blocky, unsigned int num_blocksx, unsigned int num_blocksy);

#endif
