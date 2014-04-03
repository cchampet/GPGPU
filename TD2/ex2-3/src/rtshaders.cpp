/*********************************************************\
 * Nom du fichier   : RTShaders.cpp
 * Date de creation : jeu 24 fev 2010
 * Auteur :
 * Description :
 * Version : 1.0
 * Contact :
 * Copyright : libRT@2005
 * Dernière mise à jour : jeu 24 fév 2010
\*********************************************************/

// Inclusion de fichiers
#include "rtshaders.hpp"
#include <string>

/////////////////////////////
/// RTShaders::loadShader ///
/////////////////////////////
GLhandleARB RTShaders::loadShader(const char *vertexFile, const char *fragmentFile, bool verbose) {
	GLhandleARB programObject;
	if(verbose)
		std::cout << "Debut de l'initialisation des shaders" << std::endl;

	// ETAPE 1 : CREATION DU PROGRAM OBJECT : c'est un "handler" vers cet "objet" qui sera renvoyé par la fonction
	programObject = glCreateProgramObjectARB();
	if(verbose) std::cout << "Initialisation du programme sur les shaders ";
	if(!programObject){
		if(verbose) {
			std::cout << "[ECHEC]" << std::endl;
			printLog(programObject,0);
			std::cout << "Les shaders ne seront pas utilisés" << std::endl;
		}
		return 0;
	}
	if(verbose) {
		std::cout << "[OK]" << std::endl;
		//printLog(programObject,0);
	}

	// ETAPE 2, 3 & 4 : CREATION DU VERTEX SHADER OBJECT, CHARGEMENT DES SOURCES et COMPILATION
	if(!compileVertex(vertexFile, programObject, verbose)) {
		if(verbose) std::cout << "Les shaders ne seront pas utilisés" << std::endl;
		return 0;
	}

	// ETAPE 2, 3 & 4 : CREATION DU FRAGMENT SHADER OBJECT et COMPILATION
	if(!compileFragment(fragmentFile, programObject, verbose)) {
		if(verbose) std::cout << "Les shaders ne seront pas utilisés" << std::endl;
		return 0;
	}

	// ETAPE 5 : ETAPE DE LINK
	if(linkProgram(programObject, verbose)) {
		if(verbose) std::cout << "Fin de l'initialisation des shaders" << std::endl;
		return programObject;
	}
	else{
		if(verbose) std::cout << "Les shaders ne seront pas utilisés" << std::endl;
		return 0;
	}
}


/////////////////////////////
/// RTShaders::loadShader ///
/////////////////////////////
GLhandleARB RTShaders::loadShader(std::vector<char *>::iterator vstart,
																	std::vector<char *>::iterator vend,
																	std::vector<char *>::iterator fstart,
																	std::vector<char *>::iterator fend, bool verbose) {
	GLhandleARB programObject;
	if(verbose)
		std::cout << "Debut de l'initialisation des shaders" << std::endl;

	// Creation du programme sur les shaders
	programObject = glCreateProgramObjectARB();
	if(verbose)
		std::cout << "Initialisation du programme sur les shaders ";
	if(!programObject){
		if(verbose){
			std::cout << "[ECHEC]" << std::endl;
			printLog(programObject,0);
			std::cout << "Les shaders ne seront pas utilisés" << std::endl;
		}
		return 0;
	}
	if(verbose){
		std::cout << "[OK]" << std::endl;
		printLog(programObject,0);
	}

	std::vector<char *>::iterator it;

	// Compilation du vertex shader
	for(it = vstart; it != vend; ++it){
		if(!compileVertex((*it), programObject, verbose)){
			if(verbose) std::cout << "Les shaders ne seront pas utilisés" << std::endl;
			return 0;
		}
	}

	// Compilation du fragment shader
	for(it = fstart; it != fend; ++it){
		if(!compileFragment((*it), programObject, verbose)){
			if(verbose) std::cout << "Les shaders ne seront pas utilisés" << std::endl;
			return 0;
		}
	}

	if(linkProgram(programObject, verbose)){
		if(verbose) std::cout << "Fin de l'initialisation des shaders" << std::endl;
		return programObject;
	}
	else{
		if(verbose) std::cout << "Les shaders ne seront pas utilisés" << std::endl;
		return 0;
	}
}




///////////////////////////
/// RTShaders::printLog ///
///////////////////////////
void RTShaders::printLog(GLhandleARB object, const char* str)
{
	int length = 0;

	glGetObjectParameterivARB(object, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);

  if(length > 1) {
		// S'il ya quelque chose à afficher
		GLcharARB *logStr = new GLcharARB[length];
		// Recuperation du log
		glGetInfoLogARB(object, length, 0, logStr);

		if(str) std::cout << str << " ";
		std::cout << logStr << std::endl;
		std::string ch = logStr;
		// Pour eviter un plantage du pc...
		int i=ch.find("software");
		if(i>=0){
			std::cerr << "Impossible de continuer le programme en software." << std::endl;
			exit(0);
		}
		delete logStr;
	}
	else {
		std::cerr << "Pas de fichier de log a afficher." << std::endl;
	}
}


//////////////////////////////
/// RTShaders::linkProgram ///
//////////////////////////////
bool RTShaders::linkProgram(GLhandleARB programObject, bool verbose)
{
  if(verbose)
    std::cout << "Linkage du programme ";

  int linked = 0;   // Status du linkage

  // Linkage du programme
  glLinkProgramARB(programObject);
  // Recuperation de l'état du linkage
  glGetObjectParameterivARB(programObject, GL_OBJECT_LINK_STATUS_ARB, &linked);

  if(linked) {
    if(verbose){
      std::cout << "[OK]" << std::endl;
      printLog(programObject, 0);
    }
    return true;
  }

  if(verbose){
    std::cout << "[ECHEC]" << std::endl;
    printLog(programObject, 0);
  }
  return false;
}

/////////////////////////////////
/// RTShaders::compileProgram ///
/////////////////////////////////
bool RTShaders::compileProgram(const char *source, const GLenum shaderType, GLhandleARB programObject, bool verbose)
{
  int compiled = 0;  // status compilation de compilation
  // Creation d'un objet qui va contenir des sources
  GLhandleARB shaderObject = glCreateShaderObjectARB(shaderType);
  // Association des sources
  glShaderSourceARB(shaderObject, 1, &source, 0);
  // Compilation des sources
  glCompileShaderARB(shaderObject);

  // Recuperation de l'état de la compilation
  glGetObjectParameterivARB(shaderObject, GL_OBJECT_COMPILE_STATUS_ARB, &compiled);
  if(verbose)
    std::cout << "Compilation du programme ";

  if(compiled){
    // On attache l'objet compilé à notre programme
    glAttachObjectARB(programObject, shaderObject);
    if(verbose){
      std::cout << "[OK]" << std::endl;
      printLog(shaderObject,0);
    }
  }
  else{
    if(verbose){
      std::cout << "[ECHEC]" << std::endl;
      printLog(shaderObject,0);
    }
  }

	/*
	std::cerr<<"********************************************************"<<std::endl;
	//cerr<<listing<<endl;
	int taille_shader = -1;
	glGetObjectParameterivARB(shaderObject,GL_OBJECT_SHADER_SOURCE_LENGTH_ARB,&taille_shader);
	std::cerr<<"**************************"<<taille_shader<<std::endl;
	char* listing = new char[taille_shader];
	glGetShaderSourceARB(shaderObject,taille_shader,NULL,listing);
	std::cerr<<"********************************************************"<<std::endl;
	std::cerr<<listing<<std::endl;
	std::cerr<<"********************************************************"<<std::endl;
	*/

  glDeleteObjectARB(shaderObject);

  return compiled != 0;
}


////////////////////////////////
/// RTShaders::compileVertex ///
////////////////////////////////
bool RTShaders::compileVertex(const char *vertexFile, GLhandleARB& programObject, bool verbose)
{
  // Vertex shader
  char *vertexSource;
  if(vertexFile) {
    if(verbose)
      std::cout << "Programme sur les vertex : " << std::endl;
    if(!loadSource(vertexFile, &vertexSource)){
      if(verbose)
	std::cout << "Chargement du programme '" << vertexFile << "' [ECHEC]" << std::endl;
      return false;
    }
    if(verbose)
      std::cout << "Chargement du programme '" << vertexFile << "' [OK]" << std::endl;

    // Compilation du programme sur les vertex
    if(!compileProgram(vertexSource, GL_VERTEX_SHADER_ARB, programObject, verbose)) {
      delete vertexSource;
      return false;
    }
    delete vertexSource;
  }
  return true;
}


//////////////////////////////////
/// RTShaders::compileFragment ///
//////////////////////////////////
bool RTShaders::compileFragment(const char *fragmentFile, GLhandleARB& programObject, bool verbose)
{

  char *fragmentSource;
  if(fragmentFile) {
    if(verbose)
      std::cout << "Programme sur les fragments: " << std::endl;
    if(!loadSource(fragmentFile, &fragmentSource)){
      if(verbose)
	std::cout << "Chargement du programme '" << fragmentFile << "' [ECHEC]" << std::endl;
      return false;
    }
    if(verbose)
      std::cout << "Chargement du programme '" << fragmentFile << "' [OK]" << std::endl;

    // Compilation du programme sur les fragments
    if(!compileProgram(fragmentSource, GL_FRAGMENT_SHADER_ARB, programObject,verbose)) {
	delete fragmentSource;
	return false;
    }
    delete fragmentSource;
  }
  return true;
}



////////////////////////////////
/// RTShaders::deleteProgram ///
////////////////////////////////
void RTShaders::deleteProgram(GLhandleARB programObject)
{
  // S'il existe on supprime le programme GLSL
  if(programObject)
    glDeleteObjectARB(programObject);
}


/////////////////////////////
/// RTShaders::loadSource ///
/////////////////////////////
bool RTShaders::loadSource(const char* filename, char** source)
{

  std::ifstream fichier(filename);
  if(!fichier){
    std::cout << "Impossible d'ouvrir le fichier '" << filename << "'" << std::endl;
    return false;
  }

  std::stringstream buffer; // variable contenant l'intégralité du fichier
  // copier l'intégralité du fichier dans le buffer
  buffer << fichier.rdbuf();
  // nous n'avons plus besoin du fichier !
  fichier.close();

  std::string s = buffer.str();

  *source = new char[ buffer.str().size() + 1];
	unsigned int i;
  for(i =0; i < buffer.str().size(); ++i){
   (*source)[i] = s[i];
  }
  (*source)[i] = '\0';
  return true;
}


///////////////////////////////////////
/// RTShaders::initShaderExtensions ///
///////////////////////////////////////

//////////////////////////////////////
/// RTShaders::areShadersSupported ///
//////////////////////////////////////
bool  RTShaders::areShadersSupported(bool v)
{

  if (v) {
    std::cout << "Controle du support des extensions liées aux shaders :" << std::endl;
    std::cout << "Carte graphique: " << (char *)glGetString(GL_RENDERER) << std::endl;
    std::cout << "constructeur : " << (char *)glGetString(GL_VENDOR) << std::endl;
    std::cout << "Version d'OpenGL : " <<(char *)glGetString(GL_VERSION) << std::endl;
  }
  if(GLEW_ARB_shading_language_100 || GLEW_ARB_shader_objects ||
			GLEW_ARB_fragment_shader || GLEW_ARB_vertex_shader)
	{
    if (v) {
      std::cout << "Les shaders sont supportés" << std::endl;
		}
    return true;
  }
  if (v) {
    std::cout << "Les shaders ne sont pas supportés" << std::endl;
	}
  return false;
}

/////////////////////////////////////////
/// RTShaders::isGLExtensionSupported ///
/////////////////////////////////////////
bool  RTShaders::isGLExtensionSupported(const char* ext, bool v) {
  const GLubyte *start;
  GLubyte *where, *terminator;
	if(v)
    std::cout << "Extension " << ext << " [";

  where = (GLubyte *) strchr(ext,' ');
  if(*ext == '\0' || where) {
    if(v) {
      std::cout << "OK]" << std::endl;
		}
    return false;
  }

	char * extensions = (char*)glGetString(GL_EXTENSIONS);
	/*
	std::cout << "2>"<<std::endl;
	if (!extensions) {
		std::cout << "Extensions NULL"<<std::endl;
	}
	int nb = strlen(extensions);
	std::cout << "3>"<<nb<<">"<<std::endl;
  */
	start = (GLubyte*)extensions;
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
