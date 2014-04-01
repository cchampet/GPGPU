/***************************************************************************
                          display.h  -  description
                             -------------------
    begin                : Tue 28 Jul 2010
    copyright            : (C) 2010 by Biri
    email                : biri@univ-mlv.fr
 ***************************************************************************/
/***************************************************************************
 *   Copyright (C) 2010 by Venceslas BIRI                                  *
 *   biri@univ-mlv.fr                                                      *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef ___DISPLAY_H___
#define ___DISPLAY_H___

#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>

#define USE_FBO			1

/// ///////////////////////////////////////////////////////////////////////////
/// variables globales
/// ///////////////////////////////////////////////////////////////////////////
extern int instantPrec;
extern float sepiaColor[3];

/// ///////////////////////////////////////////////////////////////////////////
/// Prototypes des fonctions de dessin
/// ///////////////////////////////////////////////////////////////////////////
void drawScene();
void initDisplay();
void initTexture();

/// ///////////////////////////////////////////////////////////////////////////
/// Prototypes des fonctions utilitaires
/// ///////////////////////////////////////////////////////////////////////////
void writeString(float x, float y,  char* s);

#endif
