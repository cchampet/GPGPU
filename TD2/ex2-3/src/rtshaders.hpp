/*********************************************************\
 * Nom du fichier   : RTShaders.hpp
 * Date de creation : jeu 24 fév 2010
 * Auteur :
 * Description :
 * Version : 1.1
 * Contact :
 * Copyright : libRT@2005
 * Dernière mise à jour : mer 30 mar 2010
\*********************************************************/

#ifndef RTSHADERS_HPP
#define RTSHADERS_HPP


#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <cstring>
#include <sys/stat.h>
#include <vector>


/*!
  \brief Le namespace RTShaders permet de creer et d'utiliser les shaders

  Les shaders sont des petits programmes ecrit GLSL <i>Graphic Library Shading Language</i> qui a une syntaxe proche du C. Ces petits programmes ont pour but de se soustraire à deux etapes du <i>pipeline</i> d'OpenGL :
  <ul>
  <li> Remplacement de la transformation des vertex : Programme sur les vertex;
  <li> Remplacement du texturage et de la coloration des fragments : Programme sur les fragments
  </ul>

*/
namespace RTShaders {

  GLhandleARB loadShader(const char *vertexFile, const char *fragmentFile, bool v);

  GLhandleARB loadShader(std::vector<char *>::iterator vstart, std::vector<char *>::iterator vsend, std::vector<char *>::iterator fstart, std::vector<char *>::iterator fsend, bool verbose);

  void printLog(GLhandleARB object, const char* str);

  bool linkProgram(GLhandleARB programObject, bool verbose);

  bool compileProgram(const char *filename, const GLenum shaderType, GLhandleARB programObject, bool verbose);

  bool compileVertex(const char *vertexFile, GLhandleARB& programObject, bool verbose);

  bool compileFragment(const char *fragmentFile, GLhandleARB& programObject, bool verbose);

  //bool createProgram(const char *vertexFile, const char *fragmentFile, GLhandleARB& programObject, bool verbose);

  void deleteProgram(GLhandleARB programObject);

  bool loadSource(const char* filename, char** source);

  bool areShadersSupported(bool v = false);

  bool isGLExtensionSupported(const char* ext, bool v);

};

#endif


