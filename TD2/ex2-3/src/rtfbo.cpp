/***************************************************************************
                          rtfbo.cpp  -  description
                             -------------------
    begin                : Mon Mar 11 2010
    copyright            : (C) 2010 by Venceslas Biri
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
#include "rtfbo.hpp"
#include <GL/glext.h>


void checkFramebufferStatus() {
	GLenum status;
	status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	switch(status) {
	case GL_FRAMEBUFFER_COMPLETE_EXT:
		std::cout << "status:  GL_FRAMEBUFFER_COMPLETE_EXT  ...continuing" << std::endl;
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
		std::cout << "status:  GL_FRAMEBUFFER_UNSUPPORTED_EXT  ...exiting" << std::endl;
		exit(0);
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
		std::cout << "status : FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT  ...exiting" << std::endl;
		exit(0);
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
		std::cout << "status : FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT  ...exiting" << std::endl;
		exit(0);
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
		std::cout << "status : FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT  ...exiting" << std::endl;
		exit(0);
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
		std::cout << "status : FRAMEBUFFER_INCOMPLETE_FORMATS_EXT  ...exiting" << std::endl;
		exit(0);
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
		std::cout << "status : FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT  ...exiting" << std::endl;
		exit(0);
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
		std::cout << "status : FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT  ...exiting" << std::endl;
		exit(0);
		break;
	default:
		std::cout << "Shouldn't reach this point.  Something is very wrong.  ...exiting" << std::endl;
		exit(0);
	}
}


bool isGLExtensionSupported(const char* ext, bool v) {
  const GLubyte *start;
  GLubyte *where, *terminator;
  if(v)
    std::cout << "Extension " << ext << " [";

  where = (GLubyte *) strchr(ext, ' ');
  if(where || *ext == '\0'){
    if(v)
      std::cout << "OK]" << std::endl;
    return false;
  }

  const GLubyte *extensions = glGetString(GL_EXTENSIONS);

  start = extensions;
  for(;;) {
    where = (GLubyte*) strstr((const char*) start, ext);
    if(!where)
      break;
    terminator = where + strlen(ext);
    if(where == start || *(where - 1) == ' ')
      if(*terminator == ' ' || *terminator == '\0'){
	if(v)
	  std::cout << "OK]" << std::endl;
	return true;
      }
    start = terminator;
  }
  if(v)
    std::cout << "ECHEC]" << std::endl;
  return false;
}

bool initFBO() {

  // Si l'extension est presente
  if(!GLEW_EXT_framebuffer_object) {
		std::cerr<<"LES FBO ne sont pas supportés !"<<std::endl;
    exit(0);
  }

  checkFramebufferStatus();
  return true;
}

/*	*********************************************************************************
		*********************** EXEMPLE OF USE ******************************************
		********************************************************************************* */

/*
	////////// EXEMPLE OF USE
  // Activation du framebuffer object
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
  // Activation du buffer de rendu
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depth_rb);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushMatrix();

  glLightfv(GL_LIGHT0, GL_POSITION,lightPosition);
  glLightfv(GL_LIGHT0, GL_SPECULAR,white);
  glLightfv(GL_LIGHT0, GL_AMBIENT,white);
  glLightfv(GL_LIGHT0, GL_DIFFUSE,white);
  glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, spotDirection);

  // activation du programme de shader
  if(useshaders && sProgram) {
    glUseProgramObjectARB(sProgram);
	}
  else {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
  }

  glMaterialfv(GL_FRONT, GL_SPECULAR,white);
  glMaterialfv(GL_FRONT, GL_DIFFUSE,red);
  glMaterialfv(GL_FRONT, GL_AMBIENT,black);
  glMaterialf( GL_FRONT, GL_SHININESS, 60.0);
  glutSolidTeapot(1.0f);

  if(useshaders)
    glUseProgramObjectARB(0);
  else{
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
  }

  glPopMatrix();

	////////// EXEMPLE OF USE
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  float aratio = 1.0/ratio;
  float tratio1 = float(width)/float(TEX_SIZE);
  float tratio2 = float(height)/float(TEX_SIZE);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(-1.0,1.0,-aratio,aratio);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glBindTexture(GL_TEXTURE_2D,textureid);
  glBegin(GL_QUADS);
  glColor3f(1.0f,1.0f,1.0f);
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,tratio1,tratio2);
  glVertex2f(1.0f,aratio);
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,0.0f,tratio2);
  glVertex2f(-1.0f,aratio);
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,0.0f,0.0f);
  glVertex2f(-1.0f,-aratio);
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,tratio1,0.0f);
  glVertex2f(1.0f,-aratio);
  glEnd();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glutSwapBuffers();
}


void init() {

  glClearColor(1.0f,1.0f,0.0f,1.0f);
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glShadeModel(GL_SMOOTH);

  // Initialisation des fonctions des MRTs
  glDrawBuffersATI = (PFNGLDRAWBUFFERSATIPROC) glXGetProcAddressARB((const GLubyte*)"glDrawBuffersATI");
  if(glDrawBuffersATI==0){
    exit(0);
  }

  // Initialisation des textures
  glGenTextures(1, &color_tex1);
  glBindTexture(GL_TEXTURE_2D, color_tex1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TEX_SIZE, TEX_SIZE, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  // Initialisation des textures
  glGenTextures(1, &color_tex2);
  glBindTexture(GL_TEXTURE_2D, color_tex2);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TEX_SIZE, TEX_SIZE, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
  textureid = color_tex2;

  // Initialisation des FBOs
  // Recuperation d'un identifiant unique de Framebuffer object qui n'est pas encore utilise
  glGenFramebuffersEXT(1, &fb);
  // Initialisation et activation du framebuffer object
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
  // Association de deux textures au framebuffer object pour le rendu
  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D,color_tex2, 0);
  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D,color_tex1, 0);
  // Recuperation d'un identifiant unique de buffer de rendu qui n'est pas encore utilise
  glGenRenderbuffersEXT(1, &depth_rb);
  // Initialisation et activation du buffer de rendu
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depth_rb);
  // Definition du buffer de rendu (ici buffer de profondeur)
  glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, TEX_SIZE, TEX_SIZE);
  // Association du buffer de rendu au framebuffer object
  glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth_rb);
  // Activation des MRTs pour le contexte du framebuffer object
  glDrawBuffersATI(2, db);
  // Desactivation du framebuffer object et du buffer de rendu
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  timebase = glutGet(GLUT_ELAPSED_TIME);
}
*/
