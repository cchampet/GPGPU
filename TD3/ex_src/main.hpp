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

/// /////////////////////////////////////////////////////////////////////////////
/// interface to compute fonction !
float runTest(unsigned int *h_idata,unsigned int* h_odata,unsigned int* h_oHisto,unsigned int size,unsigned int size_one_block,unsigned int num_blocks);
float runGauss(unsigned int *idata, unsigned int* odata, unsigned int sz_thread_per_block, unsigned int num_blocksx, unsigned int num_blocksy);

/// /////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param size       size of one dimension of the square 2D array
/// /////////////////////////////////////////////////////////////////////////////
void computeGold(unsigned int* reference, unsigned int* idata, const unsigned int size);

/// /////////////////////////////////////////////////////////////////////////////
//! Compare two differents data set
//! @param reference  reference data, supposed to be the reference
//! @param data       input data to be compared
//! @param size       size of one dimension of the square 2D array
/// /////////////////////////////////////////////////////////////////////////////
bool compareData( const unsigned int* reference, const unsigned int* data, const unsigned int size);

#endif
