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
// export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:usr/local/cuda/lib


#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cmath>

#include "main.hpp"
#include "display.hpp"
#include "interface.hpp"

TimerManager* tman;
unsigned int id_timer_1;
unsigned int id_timer_2;
const unsigned int NT = 20;
const unsigned int NB = 20;
const unsigned int N = NT * NB;

using namespace std;

/// /////////////////////////////////////////////////////////////////////////////
//! Compare two differents data set
//! @param reference  reference data, supposed to be the reference
//! @param data       input data to be compared
//! @param len        number of elements in reference / data
/// /////////////////////////////////////////////////////////////////////////////
bool compareData( const float* reference, const float* data, const unsigned int len) {
	bool result = true;
	unsigned int error_count = 0;
	float epsilon = 0.01f;
	float diff;
	bool comp;
	for( unsigned int i = 0; i < len; ++i) {
		if (reference[i]==0.f) {
			comp = (data[i]==0.f);
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

void computeSimpleVector(float * h_src, float * h_dest_cpu) {
	for(unsigned int i = 0; i < N; ++i) {
		h_dest_cpu[i] = i * sinf(h_src[i]);
	}
}

////////////////////////////      MAIN      ///////////////////////////
int main(int argc, char** argv)
{

	id_timer_1 = id_timer_2 = 0;

	// Variable initialization
	float * h_src = (float*)malloc(sizeof(float) * N);
	float * h_dest_cpu = (float*)malloc(sizeof(float) * N);
	float * h_dest_gpu = (float*)malloc(sizeof(float) * N);
	tman = new TimerManager();
	tman->addOneTimer(1);

	std::cerr<<"START ..."<<std::endl;

	tman->startOneExecution(id_timer_1);
	computeSimpleVector(h_src, h_dest_cpu);
	tman->stopOneExecution(id_timer_1);
	std::cout << "computeSimpleVector (CPU) : " << tman->getLastTime(id_timer_1) << std::endl;

	float h_timer_gpu = runTest(h_src, h_dest_gpu);
	std::cout << "computeSimpleVector (GPU) : " << h_timer_gpu << std::endl;

	std::cerr<<"... DONE"<<std::endl;

  // Parse the command line arguments
	if (argc<3) {
		help(argv[0]);
		exit(1);
	}

	for (int i=1;i<argc;i++) {
		if (strcmp(argv[i],"-h") == 0) {
			help(argv[0]);
			exit(1);
		}
		if (strcmp(argv[i],"-f") == 0) {
			// On a un image en niveau de gris a charger ...
			strcpy(nomFIC,argv[++i]);
		}
		/*
		if (strcmp(argv[i],"-f") == 0)
         fullscreen = true;
      if (strcmp(argv[i],"-s") == 0)
         solide = true;
		*/
	}

	cerr<<"Initialisation"<<endl;
	init();
	cerr<<"Fin initialisation"<<endl;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE | GLUT_STENCIL);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(image_base->tailu,image_base->tailv);
	if (glutCreateWindow("A Simple GLSL Program") == GL_FALSE) return 1;
	//if (fullscreen) glutFullScreen();

	cerr<<"Initialisation GL"<<endl;
	glewInit();
	initGL();
	cerr<<"Fin initialisation GL"<<endl;


	/* association de la fonction de dimensionnement */
	glutReshapeFunc(reshapeFunc);
	/* association de la fonction d'affichage */
	glutDisplayFunc(drawScene);
	/* association de la fonction d'événement souris */
	glutMouseFunc(clickMouse);
	/* association de la fonction de DRAG */
	//glutMotionFunc(motionFunc);
	/* association de la fonction d'événement du clavier */
	glutKeyboardFunc(kbdFunc);
	/* association de la fonction de traitement des touches*/
	/* spéciales du clavier                                */
	glutSpecialFunc(kbdSpFunc);
	/* fonction d'attente */
	glutIdleFunc(idleFunc);

	/* Ready to go! */
	glutMainLoop();

	return 0;
}
