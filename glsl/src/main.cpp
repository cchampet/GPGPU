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

#include <cstring>
#include <cstdlib>
#include <iostream>

#include "main.hpp"
#include "display.hpp"
#include "interface.hpp"

using namespace std;

////////////////////////////      MAIN      ///////////////////////////
int main(int argc, char** argv)
{
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
