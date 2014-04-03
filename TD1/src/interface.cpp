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

#include "display.hpp"
#include "interface.hpp"

#define COMPUTE_FPS					1
#define NOMBRE_SAMPLE_FPS		100
/*#define TIME_BASE						0
#define TIME_INIT						0*/

/// ///////////////////////////////////////////////////////////////////////////
/// variables globales
/// ///////////////////////////////////////////////////////////////////////////

/// L'ecran
unsigned int width_ecran = 800;
unsigned int height_ecran = 600;

/// Position de la camera ! VARIATION DE -100 a 100
unsigned int mytime;
unsigned int cptidletime;

/// Flag pour le dessin
bool flag_dessin[12];
bool flag_svg_fic;
bool flag_test;
bool flag_shaders;
bool flag_timer;
unsigned char *image_save;
bool modified_tampon = true;
unsigned int which_test;
TimerManager* timeManager;

/// La scene
char nomFIC[60] = {'\0'};
ImageTex* image_base = NULL;
ImageTex* image_gpu = NULL;
ImageTex* image_cpu = NULL;
unsigned int fbo_handler;
GLhandleARB testShader;

/// ///////////////////////////////////////////////////////////////////////////
/// Corps des fonctions IHM
/// ///////////////////////////////////////////////////////////////////////////

/// ///////////////////////////////////////////////////////////////////////////
/// fonction associée aux interruptions clavier
/// - c : caractère saisi
/// - x,y : coordonnée du curseur dans la fenêtre
void kbdFunc(unsigned char c, int , int )
{
 switch(c) {
		case 27: // quit
			exit(0);
			break;
		case '1': // print
		case '2': // print
		case '3': // print
		case '4': // print
		case '5': // print
		case '6': // print
		case '7': // print
			which_test = (int)(c-'0');
			std::cerr<<"Fixe test sur "<<which_test<<std::endl;
			break;
		case 'i': // print
			print();
			break;
		case '<': // aide
			help((char*)"./aSimpleGLSLProg");
			break;
		case 'x': // GPU On / Off
			cerr<<"Switch GPU ";
			flag_shaders = !flag_shaders;
			if (flag_shaders) cout<<" ON"<<endl;
			else  cout<<" OFF"<<endl;
			break;
		case 'f': // FPS computation On / Off
			cerr<<"Switch Test FPS compute "<<!flag_test<<endl;
			instantPrec = glutGet(GLUT_ELAPSED_TIME);
			flag_test = !flag_test;
			break;
		case 't': // Timer On / Off
			flag_timer = !flag_timer;
			cerr<<"Switch Timer "<<!flag_test<<endl;
			if (flag_shaders) cout<<" ON"<<endl;
			else  cout<<" OFF"<<endl;
			break;
		case 's': // Print image
			saveFromGL(width_ecran,height_ecran,"./essai.ppm");
			break;
		case 'o': // Improve red of sepiaColor
			sepiaColor[0] += 0.1;
			break;		
		case 'p': // Decrease red of sepiaColor
			sepiaColor[0] -= 0.1;
			break;
		case 'l': // Improve green of sepiaColor
			sepiaColor[1] += 0.1;
			break;		
		case 'm': // Decrease green of sepiaColor
			sepiaColor[1] -= 0.1;
			break;
		case ' ': // DO NOTHING...
			break;
		default :
			cerr<<"Touche non fonctionnelle"<<endl;
	}
	glutPostRedisplay();
}

void affiche_flag_dessin() {
	cout<<"FLAG AFFICHAGE : ";
	for(int i=0;i<12;i++) {
		cout<<flag_dessin[i]<<" ";
	}
	cout<<endl;
}

/// ///////////////////////////////////////////////////////////////////////////
/// fonction associée aux interruptions clavier speciales
/// - c : caractère saisi
/// - x,y : coordonnée du curseur dans la fenêtre
void kbdSpFunc(int c, int , int )
{
 switch(c)
    {
    case GLUT_KEY_F1 :
			flag_dessin[0] = true;
			flag_dessin[1] = false;
			flag_dessin[2] = false;
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F2 :
			flag_dessin[0] = false;
			flag_dessin[1] = true;
			flag_dessin[2] = false;
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F3 :
			flag_dessin[0] = false;
			flag_dessin[1] = false;
			flag_dessin[2] = true;
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F4 :
			flag_dessin[3] = !flag_dessin[3];
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F5 :
			flag_dessin[4] = !flag_dessin[4];
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F6 :
			flag_dessin[5] = !flag_dessin[5];
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F7 :
			flag_dessin[6] = !flag_dessin[6];
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F8 :
			flag_dessin[7] = !flag_dessin[7];
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F9 :
			flag_dessin[8] = !flag_dessin[8];
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F10 :
			flag_dessin[9] = !flag_dessin[9];
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F11 :
			flag_dessin[10] = !flag_dessin[10];
			affiche_flag_dessin();
			break;
    case GLUT_KEY_F12 :
			flag_dessin[11] = !flag_dessin[11];
			affiche_flag_dessin();
			break;
		case GLUT_KEY_UP :
			break;
    case GLUT_KEY_DOWN :
			break;
    case GLUT_KEY_LEFT :
			break;
    case GLUT_KEY_RIGHT :
			break;
    case GLUT_KEY_PAGE_UP :
			break;
    case GLUT_KEY_PAGE_DOWN :
			break;
		default :
			cerr<<"Touche speciale non fonctionnelle"<<endl;
	}
	glutPostRedisplay();
}

/// ///////////////////////////////////////////////////////////////////////////
/// fonction associée aux evenements souris
/// - button,state : état du bouton cliqué
/// - x,y : coordonnée de la souris dans la fenêtre
void clickMouse(int button,int state,int /*x*/,int /*y*/)
{
	if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
		//int h = glutGet(GLUT_WINDOW_HEIGHT);
		//int w = glutGet(GLUT_WINDOW_WIDTH);

		glutPostRedisplay();
	}
}

/// ///////////////////////////////////////////////////////////////////////////
/// Corps des fonctions de rendu
/// ///////////////////////////////////////////////////////////////////////////

/// ///////////////////////////////////////////////////////////////////////////
/// fonction de changement de dimension de la fenetre paramêtres :
/// - width  : largeur (x) de la zone de visualisation
/// - height : hauteur (y) de la zone de visualisation
void reshapeFunc(int width,int height)
{
	width_ecran = width;
	height_ecran = height;
	//cerr<<"Resize GL : "<<h << " , " << w << " , "<< r << "\n";
	glViewport( 0, 0, (GLint)width, (GLint)height );
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluOrtho2D(0,width,0,height);
/*	glFrustum(-LARG_ECRAN_PROJ*0.5,LARG_ECRAN_PROJ*0.5,-0.5*LARG_ECRAN_PROJ*r,
            0.5*LARG_ECRAN_PROJ*r,DIST_OEIL_ECRAN,MAX_DIST);*/
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	modified_tampon = true;
}

/// ///////////////////////////////////////////////////////////////////////////
/// fonction d'exectution de process en attendant un evenement
void idleFunc(void)
{
	if (flag_test) {
		cptidletime++;
		if (COMPUTE_FPS && (cptidletime%NOMBRE_SAMPLE_FPS == 0)) {
			int instant = glutGet(GLUT_ELAPSED_TIME);
			double t=(instant-instantPrec)/1000.0;
			cerr<<"FPS : "<<NOMBRE_SAMPLE_FPS/t<<"("<<instant<<")"<<" <"<<cptidletime<<"> "<<std::endl;
			instantPrec = instant;
		}
		glutPostRedisplay();
	}
}

void init()
{
	cptidletime = 0;

	image_save=NULL;
	flag_svg_fic = false;
	flag_test = false;
	flag_shaders = false;
	which_test = 0;
	modified_tampon = false;
	timeManager = new TimerManager();
	timeManager->addOneTimer(1);

	image_base = new ImageTex(0,nomFIC);
	image_gpu = new ImageTex(image_base,true);
	image_gpu->index = 1;
	image_cpu = new ImageTex(image_base,true);
// 	for(unsigned int i=0;i<image_cpu->tailv;i++) {
// 		for(unsigned int j=0;j<image_cpu->tailu;j++) {
// 			image_cpu->set(i,j,0,image_base->get(i,j,2));
// 			image_cpu->set(i,j,1,image_base->get(i,j,1));
// 			image_cpu->set(i,j,2,image_base->get(i,j,0));
// 		}
// 	}
	image_cpu->index = 2;
}

void initGL()
{
	initDisplay();
}

void help(char *cmd)
{
  cout <<endl;
  cout <<"**************************************"<<endl;
  cout <<"Usage: "<<cmd<<" [-h] -f <nom_fic> &"<<endl;
  cout <<"  -f <nom_fic> image à traiter"<<endl;
	cout <<"  -h   ce message "<<endl;
	cout <<" "<<endl;
  cout <<" "<<endl;
  cout <<"**************************************"<<endl;
}

void print()
{
  cout <<"\n**************************************"<<endl;
  cout <<"                INFORMATIONS            "<<endl;
	affiche_flag_dessin();
  cout <<"**************************************"<<endl;
}





/// ///////////////////////////////////////////////////////////////////////////
/// Fonctions utilitaires
/// ///////////////////////////////////////////////////////////////////////////
void saveFromGL(int w,int h,const char *nom_dest) {
	fstream fout;

	// Calculer deja la taille de l'image.
	fout.open(nom_dest,ios::out);
	if(!fout.is_open()) {
		cout << "Erreur lors de l'ouverture du fichier " << nom_dest << "\n";
	}
	else {
		//cout << "Sauver fichier : "<<nom_dest;
		//cout << "Width : "<<w<<" height : "<<h<<endl;
		fout<<"P6"<<endl;
		fout<<"# CREATOR : by Animated Zatz application created by M.Biri ";
		fout<<"(biri@univ-mlv.fr)"<<endl;
		fout<<w<<" "<<h<<endl;
		fout<<"255"<<endl;

		unsigned char *image = new unsigned char[h*w*3];
		if (image==NULL) exit(5);
		glPixelStorei(GL_UNPACK_ALIGNMENT,1);
		glReadPixels(0,0,(GLint)w,(GLint)h,GL_RGB,GL_UNSIGNED_BYTE,image);
		for(int i=h-1;i>=0;i--) {
			fout.write((char*)(image+i*w*3),w*3);
		}
		delete(image);
	}
	fout.close();
}

