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

#include "interface.hpp"
#include "display.hpp"
#include "rtfbo.hpp"
#include "imageComputation.hpp"
#include <iostream>

void __check_gl(int line, const char *filename) {
	int err = glGetError();
	if(err != GL_NO_ERROR) {
		std::cerr << "ERROR GL : erreur dans le fichier " << filename << " à la ligne " ;
		std::cerr << line << " : " << gluErrorString(err) << "\n";
		exit(0);
	}
}

#define CHECK_GL __check_gl(__LINE__, __FILE__)

/// ///////////////////////////////////////////////////////////////////////////
/// variables globales
/// ///////////////////////////////////////////////////////////////////////////
int instantPrec;
float sepiaColor[3];

/// ///////////////////////////////////////////////////////////////////////////
/// Corps des fonctions de dessin
/// ///////////////////////////////////////////////////////////////////////////

void initDisplay() {

	for(int i=0;i<10;i++) {
		flag_dessin[i] = false;
	}
	flag_dessin[0] = true;

	instantPrec = glutGet(GLUT_ELAPSED_TIME);

	sepiaColor[0] = 0.2;
	sepiaColor[1] = 0.1;
	sepiaColor[2] = 0;

	/// INITIALISATION DES TEXTURES ...
	initTexture();

	/// INITIALISATION DES SHADERS ...
	if(!RTShaders::areShadersSupported(true)) {
		std::cerr<<"[In initDisplay] : Shaders are not supported..."<<std::endl;
		exit(5);
	}
	testShader = RTShaders::loadShader("./shaders/tstshader.vert","./shaders/tstshader.frag",true);

	/// INITIALISATION DES FBOS ...
	if (USE_FBO) {
		if(initFBO()==false){
			cerr<<"FBO not supported ! Exiting"<<endl;
			exit(5);
		}

		glGenFramebuffersEXT(1, &fbo_handler);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,fbo_handler);
		GLuint depth_rb;
		glGenRenderbuffersEXT(1, &depth_rb);
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depth_rb);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT,GL_DEPTH_COMPONENT,image_base->tailu,image_base->tailv);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth_rb);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, image_base->indbind , 0);
		checkFramebufferStatus();
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);
	}

	/// INITIALISATION CLASSIQUE OPENGL ...
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glMatrixMode(GL_TEXTURE);
		glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);

	glShadeModel( GL_SMOOTH );
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_BLEND);
	glDisable(GL_FOG);
	CHECK_GL;


	glUseProgramObjectARB(0);
}

void initTexture() {
	glActiveTextureARB(GL_TEXTURE0_ARB);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);
	image_base->initTexture();
	image_base->setTextureWraping(GL_CLAMP,GL_CLAMP);
	image_cpu->initTexture();
	image_cpu->setTextureWraping(GL_CLAMP,GL_CLAMP);
	image_gpu->initTexture();
	image_gpu->setTextureWraping(GL_CLAMP,GL_CLAMP);
	glBindTexture(GL_TEXTURE_2D,0);
}

void drawScene(void) {
	// ****************** MISE AU POINT DE VUE ***************************************
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	// ********************************** CLEARING ***********************************
	glClearColor(0.0,0.0,0.0,0.0);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// ** Calcul du traitement
	if (flag_timer) timeManager->startOneExecution(0,true);
	switch(which_test) {
		case (0) :
			// Nothing to do...
			break;
		case (1) :
			if (flag_shaders) makeColorSwitchGPU(image_base,image_gpu);
			else  makeColorSwitch(image_base,image_cpu);
			break;
		case (2) :
			if (flag_shaders) makeLuminanceGPU(image_base,image_gpu);
			else  makeLuminance(image_base,image_cpu);
			break;
		case (3) :
			if (flag_shaders) makeSepiaGPU(image_base,image_gpu,sepiaColor);
			else  makeSepia(image_base,image_cpu,sepiaColor);
			break;
		case (4) :
			if (flag_shaders) makeBlurGPU(image_base,image_gpu);
			else  makeBlur(image_base,image_cpu);
			break;
		case (5) :
			if (flag_shaders) makeSobelFilterGPU(image_base,image_gpu);
			else  makeSobelFilter(image_base,image_cpu);
			break;
		case (8) :
			makeAdditionOf2MatrixCuda();
			break;
		case (9) :
			makeLuminanceCuda(image_base,image_cpu);
			break;
		default :
			std::cerr<<"Cas non traité (encore)"<<std::endl;
	}
	if (flag_timer) {
		timeManager->stopOneExecution(0,true);
		std::cerr<<"TEMPS ECOULE ";
		if (flag_shaders) std::cerr<<"GPU :";
		else std::cerr<<"CPU :";
		std::cerr<<timeManager->getLastTime(0)<<std::endl;
	}

	/// DESSIN DES TEXTURES
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glColor3f(1.0,1.0,1.0);
	if (flag_dessin[0]) image_base->chargeTexture();
	if (flag_dessin[1]) {
		image_cpu->chargeTexture();
		if (modified_tampon) {
			glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,image_cpu->tailu,image_cpu->tailv,
									 0,GL_RGB,GL_UNSIGNED_BYTE,image_cpu->getTab());
			glFinish();
			modified_tampon = false;
		}
	}
	if (flag_dessin[2]) image_gpu->chargeTexture();



	/// DESSIN DU DESSIN
	glBegin(GL_QUADS);
		glTexCoord2f(0,0);
		glVertex2f(0,0);
		glTexCoord2f(1.,0.);
		glVertex2f(width_ecran,0);
		glTexCoord2f(1.,1.);
		glVertex2f(width_ecran,height_ecran);
		glTexCoord2f(0.,1.);
		glVertex2f(0,height_ecran);
	glEnd();

	// DES DESSINS
	glColor3f(1.,1.,1.);
	if (flag_dessin[2]) {
		writeString(10,10,(char*)"IMAGE GPU");
	}
	else if (flag_dessin[1]) {
		writeString(10,10,(char*)"IMAGE CPU");
	}
	else if (flag_dessin[0]) {
		writeString(10,10,(char*)"IMAGE SOURCE");
	}

	if (flag_dessin[0]) image_base->dechargeTexture();
	if (flag_dessin[1]) image_cpu->dechargeTexture();
	if (flag_dessin[2]) image_gpu->dechargeTexture();

	glutSwapBuffers();
}

/// ///////////////////////////////////////////////////////////////////////////
/// Corps des fonctions utilitaires
/// ///////////////////////////////////////////////////////////////////////////
void writeString(float x, float y,  char* s)
{
	void* mafont=(void*)GLUT_BITMAP_HELVETICA_18;
	char* c;
	glRasterPos2f(x, y);                // set position to start drawing fonts
	glDisable(GL_TEXTURE_2D);
	for (c=s; *c != '\0'; c++)
		glutBitmapCharacter((void*)mafont, *c ); // draw the character to the screen
	glEnable(GL_TEXTURE_2D);
}
