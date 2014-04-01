/***************************************************************************
                          texture.h  -  description
                             -------------------
    begin                : Aug 21 2010
    copyright            : (C) 2010 by Biri
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


#ifndef __IMAGETEX_H__
#define __IMAGETEX_H__

/** \addtogroup Macros */
/*@{*/
/** \defgroup type_texture Types de texture */
/*@{*/
/** Texture NULL. Utilisé à des fins de vérification uniquement. */
#define TEX_TYPE_NULL		99
/** Texture RVB. Texture classique (la plus utilisée). */
#define TEX_TYPE_RVB		0
/** Texture RVBA. Texture avec transparence. */
#define TEX_TYPE_RVBA		1
/** Texture LUM. Texture avec uniquement une luminance. */
#define TEX_TYPE_LUM		2
/*@}*/
/*@}*/


/** \addtogroup Macros Macros prédéfinies */
/*@{*/
/** \defgroup type_image_jpg Type d'image JPG */
/*@{*/
/// Type d'image NULL.
/// Utilisé à des fins de vérification.
#define TYPE_JPG_NULL		-1
/// Type d'image RVB.
/// Image en Rouge Vert Bleu. Le plus utilise.
#define TYPE_JPG_RVB		0
/// Type d'image RVBA
/// Image en RVB plus transparence. N'existe pas encore mais pourrait exister.
#define TYPE_JPG_RVBA		1
/// Type d'image L
/// Image en luminance.
#define TYPE_JPG_L			2
/// Type d'image CMYK
/// Image en Cyan Magenta Yellow et Noir. N'existe pas encore mais pourrait exister.
#define TYPE_JPG_CMYK		3
/*@}*/
/*@}*/



#include <GL/glew.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>

using namespace std;

/** Classe permettant le chargement et l'utilisation d'une texture en OpenGL.
	* Cette classe permet d'une part le chargement d'une texture à
	* partir d'un nom de fichier ou bien des données elles mêmes.
	* Ensuite cette classe permet leur chargement dans le contexte
	* OpenGL choisi avec les réglages de base.
  * @author Venceslas BIRI (biri@univ-mlv.fr)
  * \todo Miroir horizontal pour la lecture TGA
  */
class ImageTex {
public:
	// CONSTRUCTEURS ET DESTRUCTEURS
	/// Constructeur vide.
	/// A ne pas utiliser.
	ImageTex();
	/// Constructeur par index. Seul l'index de texture est spécifié.
	/// \param indexin l'index
	ImageTex(int indexin);
	/// Constructeur par nom de fichier.
	/// \param indexin l'index \param fic le nom du fichier
	ImageTex(int indexin,const char* fic);
	/// Constructeur par nom de fichier.
	/// \param indexin l'index
	/// \param fic le nom du fichier
	/// \param path path jusqu'au nom de fichier
	ImageTex(int indexin,const char *fic,const char *path);
	/** Constructeur par copie
		* Constructeur par copie. Seule ne seront pas recopiés :
		* - L'index index
		* - L'index OpenGL indbind
		* - L'unite de texture active : unite_tex_charge
		* Si l'argument \a copie est à vrai (valeur par défaut) alors les données sont recopiées
		* \param texsource ImageTex source
		* \param copie Vrai par défaut. Recopie ou non les données images
		*/
	ImageTex(const ImageTex* texsource,bool copie=true);
	/** Constructeur de texture avec reservation de mémoire (données = 0)  ou non.
		* Constructeur de texture avec reservation de mémoire ou non. Si il y a
		* réservation de mémoire (suivant \a reserve ), les données sont nulles et la 'texture' est donc noire.
		* Utile pour un travail ultérieur sur texture.
		* Si il n'y a pas réservation, le pointeur image est à NULL. Utile pour les textures FBO
		* \param indexin l'index
		* \param type type de l'image a choisir dans les types prédéfinis de texture
		* \param tailuin,tailvin dimension de l'image
		* \param reserve Réservation ou non de la mémoire (vrai par défaut)
		* \see type_texture
		*/
	ImageTex(int indexin,int type,int tailuin,int tailvin,bool reserve=true);
	/// Constructeur par chargement de données.
	/// \param indexin l'index
	/// \param type type de l'image a choisir dans les types prédéfinis de texture
	/// \param tailuin,tailvin dimension de l'image
	/// \param imag_src données image à reconstruire
	/// \see type_texture
	ImageTex(int indexin,int type,int tailuin,int tailvin,unsigned char* imag_src);
	/// Destructeur.
	/// \warning Attention détruit les données de l'image
	~ImageTex();

	/** Initialisation OpenGL de la texture.
		* Cette initialisation crée une texture OpenGL avec en paramètre :
		* - GL_TEXTURE_WRAP_R fixé à GL_REPEAT
		* - GL_TEXTURE_WRAP_S fixé à GL_REPEAT
		* - GL_TEXTURE_MAG_FILTER fixé à GL_LINEAR
		* - GL_TEXTURE_MIN_FILTER fixé à GL_LINEAR
		*
		* \todo Faire des fonctions et ajouter des infos internes pour changer ces paramètres
		*/
	void initTexture();
	/** Fixation des paramètres de filtering de la texture.
		* Cette fonction fixe les deux paramètres de wrapping GL_TEXTURE_MAG_FILTER et GL_TEXTURE_MIN_FILTER
		* aux valeurs à choisir classiquement dans
		* - GL_NEAREST
		* - GL_LINEAR
		*
		* \param param_min Filtering en min : choix dans GL_LINEAR, GL_NEAREST, GL_NEAREST_MIPMAP_NEAREST,
		*   GL_LINEAR_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_LINEAR
		* \param param_max Filtering en max : choix dans GL_LINEAR, GL_NEAREST
		*/
	void setTextureFiltering(GLenum param_min,GLenum param_max);
	/** Fixation des paramètres de wrapping de la texture.
		* Cette fonction fixe les deux paramètres de wrapping GL_WRAP_S et GL_WRAP_T
		* aux valeurs à choisir dans
		* - GL_CLAMP
		* - GL_CLAMP_TO_EDGE
		* - GL_REPEAT
		*
		* \param param_s Wrapping sur les coordonnées s
		* \param param_t Wrapping sur les coordonnées t
		*/
	void setTextureWraping(GLenum param_s,GLenum param_t);
	/** Chargement de la texture dans le moteur OpenGL
		* \param target_tex Index de contexte de texture (type GL_TEXTURE0_ARB qui est le paramètre par défaut)
		*/
	void chargeTexture(GLuint target_tex = GL_TEXTURE0_ARB);
	/** Dechargement de la texture dans le moteur OpenGL
		* \param target_tex Index de contexte de texture (type GL_TEXTURE0_ARB qui est le paramètre par défaut)
		*/
	void dechargeTexture(GLuint target_tex = GL_TEXTURE0_ARB);
	
	
	// *** TOOLS
	/** Flip image vertically
		*/
	void flip();
	/** Save image to a ppm or pgm file
		*/
	void saveToPXM(const char* nameFile);

	/// Accès aux données images.
	/// \return pointeur sur les données images
	inline unsigned char* getTab() const;

	inline void set(unsigned int i,unsigned int j,unsigned int depth,unsigned char val) {
		tabRVB[nbChannel*j*tailu+nbChannel*i+depth] = val;
	}

	inline unsigned char get(unsigned int i,unsigned int j,unsigned int depth) {
		return tabRVB[nbChannel*j*tailu+nbChannel*i+depth];
	}

	/** Modifie la texture pour que l'image soit une puissance de 2.
		* Modifie la taille de l'image afin que la largeur et la hauteur soient des puissances
		* de 2. Si \a flag_closer est fixé à 0 (valeur par défaut) la puissance de 2
		* utilisée est la puissance la plus grande qui soit inférieure aux largeur et
		* hauteur d'origine. Si \a flag_closer est fixé à 1 c'est la puissance de 2 la plus petite qui soit supérieure
		* aux largeur et hauteur d'origine. Si \a flag_closer est fixé à 2 alors c'est la
		* puissance de 2 la plus proche.\n
		* Note : Largeur et hauteur sont traités séparément.
		* \param new_w,new_h Nouvelles tailles de l'image
		* \param flag_closer Choix d'une puissance de 2 inférieur (0 valeur par défaut), supérieur (1) ou
		* au plus proche (2)
		*/
	void makePowerOfTwo(int& new_w,int& new_h,int flag_closer=0);

	int tailu;								///< Largeur de la texture (width)
	int tailv;								///< Hauteur de la texture (height)
	/** Index absolu dans la liste des textures utilisées.
		* Cet index peut s'utiliser lorsque l'on gère un pool de texture
		* qui sont donc toutes indexées. Cet index pouvant donc effectivement
		* être différent de l'indexation OpenGL. */
	unsigned int index;
	unsigned int indbind;			///< Index dans le binder de OpenGL
	unsigned int typetext;		///< Type de texture RVB 0 RVBA 1 L 2 \see type_texture
	unsigned int nbChannel;		///< Type de texture RVB 0 RVBA 1 L 2 \see type_texture
	unsigned int unite_tex_charge;
														///< Index dans le binder de OpenGL

private:
	void chargeImage(string* nmfile); // Chargement d'une image par nom de fichier
	void lecturePpm(FILE*f);					// Lecture d'un PPM
	void lectureJpg(string* nmfile);	// Lecture d'un JPG
	void lectureRgba(int sizex,int sizey,FILE* f);	// Lecture d'un RGBA
	void lectureTga(FILE* f);					// Lecture d'un TGA
	void lectureTgaNoCompress(FILE *file,int d);	// Lecture d'un TGA non compresse
	void lectureTgaCompress(FILE *file,int d);		// Lecture d'un TGA compresse
	bool lectureBmp(FILE* f);					// Lecture d'un BMP (non compressé)

	unsigned char* tabRVB; 		///< Donnees sources de l'image
};

/// Affichage de la texture sur un flux
extern ostream& operator<<(ostream& os,const ImageTex& src);

inline unsigned char* ImageTex::getTab() const {
	return tabRVB;
}

#endif
