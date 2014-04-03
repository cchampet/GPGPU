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

#include <TimerManager.hpp>

//#define _DEBUG
extern TimerManager* tman;
extern unsigned int id_timer_1;
extern unsigned int id_timer_2;


/// /////////////////////////////////////////////////////////////////////////////
/// interface to compute fonction !
//! renvoie le temps d'execution du kernel
/// /////////////////////////////////////////////////////////////////////////////
float runTest(float *, float *);

#endif
