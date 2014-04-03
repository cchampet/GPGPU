/***************************************************************************
                          texture.cpp  -  description
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

#include "imageTexture.hpp"

#include <cstring>
#include <string>
#include <cstdlib>
#include <jpeglib.h>
#include <setjmp.h>
#include <iostream>



/* *************************************************************************************
 * ********** STRUCTURES DE DONNEES POUR CHARGEMENT IMAGE (TGA/JPG/BMP)
 * ************************************************************************************* */

typedef struct s_bmp_fh
{
short sType; // Deux caractères B et M
int iSize; // Taille total du fichier
short sReserved1; // 0
short sReserved2; // 0
int iOffBits; // Offset des bits du bitmap dans le fichier

} t_bmp_fh;


typedef struct s_bmp_sh
{
	int iSize; // Taille de cette structure en octets
	int iWidth; // Largeur du bitmap en pixel
	int iHeight; // Hauteur du bitmap en pixel
	short sPlanes; // 1
	short sBitCount; // Bits couleurs par pixel
	int iCompression; // Schéma de compactage (0 pour aucun)
	int iSizeImage; // Taille de l'image en octets (utile pour le compactage)
	int iXpelsPerMeter; // Résolution horizontale en pixels par mètre
	int iYpelsPerMeter; // Résolution verticale en pixels par mètre
	int iClrUsed; // Nombre de couleurs utilisées dans l'image
	int iClrImportant; // Nombre de couleurs importantes
} t_bmp_sh;


typedef struct s_bmp_header
{
t_bmp_fh first_header;
t_bmp_sh second_header;

} t_bmp_header;


typedef struct s_bmp
{
t_bmp_header header;
int width;
int width_useless;
int height;
} t_bmp;


typedef struct
{
	unsigned char header[6];
	unsigned int bytesPerPixel;
	unsigned int imageSize;
	unsigned int temp;
	unsigned int type;
	unsigned int Height;
	unsigned int Width;
	unsigned int Bpp;
} TGA;

typedef struct
{
	unsigned char Header[18];
} TGAHeader;

struct my_error_mgr {
  struct jpeg_error_mgr pub;	/* "public" fields */

  jmp_buf setjmp_buffer;	/* for return to caller */
};

typedef struct my_error_mgr * my_error_ptr;


/* *************************************************************************************
 * ********** CONSTRUCTEUR & DESTRUCTEURS
 * ************************************************************************************* */

ImageTex::ImageTex() {
	index = unite_tex_charge = 0;
	tabRVB = NULL;
	indbind = 0;
	tailu = 0;
	tailv = 0;
	nbChannel = 0;
	typetext = TEX_TYPE_NULL;
}

ImageTex::ImageTex(int indexin) {
	index = indexin;
	tabRVB = NULL;
	unite_tex_charge = 0;
	tailu = 0;
	tailv = 0;
	indbind = 0;
	nbChannel = 0;
	typetext = TEX_TYPE_NULL;
}

ImageTex::ImageTex(int indexin,const char* fic) {
	string *nomfic;

	nomfic = new string((char*)fic);
	unite_tex_charge = 0;
	indbind = 0;
	index = indexin;
	tabRVB = NULL;
	typetext = TEX_TYPE_NULL;
	nbChannel = 0;
	chargeImage(nomfic);
}


ImageTex::ImageTex(int indexin,const char* fic,const char* path) {
	string *nomfic;

	nomfic = new string(path);
	nomfic->append(fic);
	unite_tex_charge = 0;
	indbind = 0;
	index = indexin;
	nbChannel = 0;
	tabRVB = NULL;
	typetext = TEX_TYPE_NULL;
	chargeImage(nomfic);
}

ImageTex::ImageTex(const ImageTex* texsource,bool copie) {
	unite_tex_charge = 0;
	indbind = 0;
	index = 0;
	nbChannel = texsource->nbChannel;
	typetext = texsource->typetext;
	tailu = texsource->tailu;
	tailv = texsource->tailv;
	if (nbChannel > 0 && copie) {
		tabRVB = new unsigned char[tailu*tailv*nbChannel];
		memcpy(tabRVB,texsource->getTab(),tailu*tailv*nbChannel);
	}
	else {
		tabRVB = NULL;
	}
}

ImageTex::ImageTex(int indexin,int type,int tailuin,int tailvin,bool reserve) {
	indbind = 0;
	unite_tex_charge = 0;
	index = indexin;
	tailu = tailuin;
	tailv = tailvin;
	typetext = type;
	switch (type) {
		case (TEX_TYPE_NULL) :
			nbChannel = 0;
			tabRVB = NULL;
			break;
		case (TEX_TYPE_LUM) :
			nbChannel = 1;
			break;
		case (TEX_TYPE_RVB) :
			nbChannel = 3;
			break;
		case (TEX_TYPE_RVBA) :
			nbChannel = 4;
			break;
		default :
			cerr<<"[VBVector3D : ImageTex] : Chargement d'un mauvais type "<<type<<" dans constructeur"<<endl;
	}
	if (nbChannel > 0 && reserve) {
		tabRVB = new unsigned char[tailu*tailv*nbChannel];
		if (tabRVB == NULL) {
			cerr<<"[VBVector3D : ImageTex] Probleme allocation memoire constructeur ImageTex !"<<endl;
			exit(5);
		}
		memset(tabRVB,0,tailu*tailv*nbChannel);
	}
	else {
		tabRVB = NULL;
	}
}

ImageTex::ImageTex(int indexin,int type,int tailuin,int tailvin,unsigned char* imag_src) {
	indbind = 0;
	unite_tex_charge = 0;
	index = indexin;
	tailu = tailuin;
	tailv = tailvin;
	typetext = type;
	switch (type) {
		case (TEX_TYPE_NULL) :
			nbChannel = 0;
			tabRVB = NULL;
			break;
		case (TEX_TYPE_LUM) :
			nbChannel = 1;
			break;
		case (TEX_TYPE_RVB) :
			nbChannel = 3;
			break;
		case (TEX_TYPE_RVBA) :
			nbChannel = 4;
			break;
		default :
			cerr<<"[VBVector3D : ImageTex] : Chargement d'un mauvais type "<<type<<" dans constructeur"<<endl;
	}
	if (nbChannel > 0) {
		tabRVB = new unsigned char[tailu*tailv*nbChannel];
		if (tabRVB == NULL) {
			cerr<<"[VBVector3D : ImageTex] : Probleme d'allocation memoire constructeur ImageTex !"<<endl;
			exit(5);
		}
		memcpy(tabRVB,imag_src,tailu*tailv*nbChannel);
	}
}

ImageTex::~ImageTex(){
	if (tabRVB != NULL) {
		delete[](tabRVB);
	}
	tabRVB = NULL;
}

/* *************************************************************************************
 * ********** FONCTIONS UTILITAIRES
 * ************************************************************************************* */

void ImageTex::makePowerOfTwo(int& new_w,int& new_h,int flag_closer) {
	new_w = new_h = 1;
	int tmp_w,tmp_h;
	tmp_w = tailu;
	tmp_h = tailv;
	while(tmp_w > 1) {
		new_w *= 2;
		tmp_w = tmp_w/2;
	}
	while(tmp_h > 1) {
		new_h *= 2;
		tmp_h = tmp_h/2;
	}
	if ((new_w == tailu) && (new_h == tailv)) return;

	if (flag_closer == 1) {
		if (new_w != tailu) new_w *= 2;
		if (new_h != tailv) new_h *= 2;
	}
	if (flag_closer == 2) {
		if ((tailu - new_w) > (2*new_w - tailu)) new_w *= 2;
		if ((tailv - new_h) > (2*new_h - tailv)) new_h *= 2;
	}

	unsigned char* newimage;
	int res = 1;
	int size = 0;
	GLenum frmt = GL_LUMINANCE;
	size = nbChannel*new_w*new_h;
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glPixelStorei(GL_PACK_ALIGNMENT,1);
	newimage = new GLubyte[size];
	if (newimage==NULL) {
		cerr<<"[VBVector3D] : Impossible d'allouer la nouvelle image ! Unscalable !"<<endl;
		return;
	}
	res = gluScaleImage(frmt,tailu,tailv,GL_UNSIGNED_BYTE,
											tabRVB,new_w,new_h,GL_UNSIGNED_BYTE,newimage);
	if (res != 0) {
		cerr<<"[VBVector3D] : Impossible d'allouer la nouvelle image ! Unscalable !"<<endl;
		return;
	}
	if (tabRVB!=NULL) delete(this->tabRVB);
	tailu = new_w;
	tailv = new_h;
	tabRVB = newimage;
	//tabRVB = NULL;
}

/* *************************************************************************************
 * ********** FONCTIONS D'INTERACTION AVEC OPENGL
 * ************************************************************************************* */

void ImageTex::initTexture() {
	glGenTextures(1,&indbind);
	glBindTexture(GL_TEXTURE_2D,indbind);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	if (typetext==TEX_TYPE_RVB) {
		glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,tailu,tailv,0,GL_RGB,GL_UNSIGNED_BYTE,getTab());
	}
	else if (typetext==TEX_TYPE_RVBA) {
		glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,tailu,tailv,
									0,GL_RGBA,GL_UNSIGNED_BYTE,getTab());
	}
	else if (typetext==TEX_TYPE_LUM){
		glTexImage2D(GL_TEXTURE_2D,0,GL_LUMINANCE,tailu,tailv,
									0,GL_LUMINANCE,GL_UNSIGNED_BYTE,getTab());
	}
	else {
		cerr<<"[VBVector3D : ImageTex ] Impossible d'initialiser une texture de type NULL"<<endl;
	}
	glBindTexture(GL_TEXTURE_2D,0);
	//cout<<"Fin initialisation ImageTex : "<<*this<<endl;
}

void ImageTex::setTextureWraping(GLenum param_s,GLenum param_t) {
	glBindTexture(GL_TEXTURE_2D,indbind);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,param_s);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,param_t);
	glBindTexture(GL_TEXTURE_2D,0);
}

void ImageTex::setTextureFiltering(GLenum param_min,GLenum param_max) {
	glBindTexture(GL_TEXTURE_2D,indbind);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,param_max);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,param_min);
	glBindTexture(GL_TEXTURE_2D,0);
}

void ImageTex::chargeTexture(GLuint target_tex) {
	unite_tex_charge = target_tex;
	glActiveTextureARB(target_tex);
	glBindTexture(GL_TEXTURE_2D,indbind);
}

void ImageTex::dechargeTexture(GLuint target_tex) {
	unite_tex_charge = target_tex;
	glActiveTextureARB(target_tex);
	glBindTexture(GL_TEXTURE_2D,0);
}

/* *************************************************************************************
 * ********** FONCTIONS DE CHARGEMENT
 * ************************************************************************************* */

void ImageTex::chargeImage(string* nmfile) {
	int posfin;
	FILE* ffic = fopen(nmfile->c_str(),"rb");
	if (ffic==NULL) {
		cerr<<"[VBVector3D] Lecture image : Impossible de lire le fichier de texture "<<*nmfile<<endl;
		exit(5);
	}

	string *indictype = new string(*nmfile);
	posfin = indictype->find_last_of('.',indictype->size());
	*indictype = indictype->erase(0,posfin+1);
	if ((indictype->compare("ppm")==0) || (indictype->compare("pgm")==0)) {
		typetext = TEX_TYPE_RVB;
		nbChannel = 3;
		lecturePpm(ffic);
		fclose(ffic);
	}
	else if (indictype->compare("rgba")==0) {
		cerr<<"LECTURE D'UN FICHIER RGBA : On fixe arbitrairement la taille à 128 !"<<endl;
		typetext = TEX_TYPE_RVBA;
		nbChannel = 4;
		lectureRgba(128,128,ffic);
		fclose(ffic);
	}
	else if ((indictype->compare("tga")==0) || (indictype->compare("TGA")==0)) {
		lectureTga(ffic);
		fclose(ffic);
	}
	else if ((indictype->compare("jpg")==0) || (indictype->compare("JPG")==0)) {
		typetext = TEX_TYPE_RVB;
		lectureJpg(nmfile);
	}
	else if ((indictype->compare("bmp")==0) || (indictype->compare("BMP")==0)) {
		if (!lectureBmp(ffic)) exit(5);
		fclose(ffic);
	}
	else {
		cerr<<"[VBVector3D] Lecture image : Erreur format fichier. Reconnu : .ppm / .rgba / .jpg / .tga "<<endl;
		exit(5);
	}
}

void ImageTex::lecturePpm(FILE *f) {
	int maxrvb;
	bool flagp5 = false;
	char tamp[50] = {'\0'};
	int cpttmp;

	cpttmp = fscanf(f,"%s",tamp);
	for (int i=0;i<50;i++) if (tamp[i] == ' ') tamp[i] = '\0';
	if (strcmp(tamp,"P6") && strcmp(tamp,"P5")) {
		cerr << "[VBVector3D] Lecture PPM : Fichier de texture not a .ppm nor a .pgm";
		exit(5);
	}
	if (!strcmp(tamp,"P5")) flagp5 = true;
	if (flagp5) {
	}
	else {
		nbChannel = 3;
	}
	while(fgetc(f)!='\n');
	cpttmp = fscanf(f,"%d %d",&tailu,&tailv);
	while(fgetc(f)!='\n');
	//while(fgetc(f)!='\n');
	cpttmp = fscanf(f,"%d",&maxrvb);

	while(fgetc(f)!='\n');
	if (maxrvb != 255) {
		cerr << "[VBVector3D] Lecture PPM : Nombre de couleur texture different de 255 #"<<tamp<<","<<maxrvb<<"! Aborting !";
		//exit(5);
	}
	if (flagp5) {
		typetext = TEX_TYPE_LUM;
		nbChannel = 1;
	}
	else {
		nbChannel = 3;
	}
	tabRVB = new unsigned char[tailu*tailv*nbChannel];
	if (tabRVB == NULL) {
		cerr<<"[VBVector3D] Lecture PPM : Probleme d'allocation memoire ImageTex::lecturePpm !"<<endl;
		exit(5);
	}
	for(int i=0;i<tailv;i++) {
		cpttmp = fread(tabRVB+(tailv-i-1)*nbChannel*tailu,sizeof(unsigned char),tailu*nbChannel,f);
	}
}

inline unsigned char ptitfit(unsigned char x) {
	int t = 4*(int)x;
	if (t>255) t=255;
	return (unsigned char)t;
}

void ImageTex::lectureRgba(int sizex,int sizey,FILE* f) {
	int cpttmp;

	tailu = sizex;
	tailv = sizey;
	nbChannel = 4;
	unsigned char* tst = new unsigned char[tailu*tailv*nbChannel];
	if (!tst) {
		cerr<<"[VBVector3D] Lecture RGBA : Pb allocation memoire !"<<endl;
		exit(5);
	}
	cpttmp = fread((char*)tst,sizeof(unsigned char),tailu*tailv*nbChannel,f);

	tabRVB =  new unsigned char[tailu*tailv*nbChannel];
	if (!tabRVB) {
		cerr<<"[VBVector3D] Lecture RGBA : Pb allocation memoire !"<<endl;
		exit(5);
	}

	unsigned char* ptim = tst;
	// Rouge...
	for(int i=0;i<tailu*tailv;i++) {
		tabRVB[4*i]   = ptitfit(*ptim++);
	}
	// Vert ...
	for(int i=0;i<tailu*tailv;i++) {
		tabRVB[4*i+1] = ptitfit(*ptim++);
	}
	// Bleu ...
	for(int i=0;i<tailu*tailv;i++) {
		tabRVB[4*i+2] = ptitfit(*ptim++);
	}
	// Transp...
	for(int i=0;i<tailu*tailv;i++) {
		tabRVB[4*i+3] = ptitfit(*ptim++);
	}
	delete[](tst);
}

void ImageTex::lectureTga(FILE *file) {
	TGAHeader tgaheader;
	unsigned char uTGAcompare = 2;
	unsigned char cTGAcompare = 10;
	bool vertical_flip = false;
	bool horizont_flip = false;
	char descriptor;
	int d;

	if (tabRVB != NULL) {
		delete[](tabRVB);
		tabRVB = NULL;
	}
	if(fread(&tgaheader, sizeof(TGAHeader), 1, file) == 0) {
		cerr<<"[VBVector3D] Lecture TGA : Error could not read file header"<<endl;
		if(file != NULL) {
			fclose(file);
		}
		exit(5);
		return;
	}
	int xorigine = tgaheader.Header[9]*256 + tgaheader.Header[8];
	int yorigine = tgaheader.Header[11]*256 + tgaheader.Header[10];
	tailu = *((unsigned short*) (tgaheader.Header+12));
	tailv = *((unsigned short*) (tgaheader.Header+14));
	descriptor = tgaheader.Header[17];

	if (descriptor && 0x20) vertical_flip = true;
	if (descriptor && 0x10) horizont_flip = true;

	if (horizont_flip) {
		cerr<<"[VBVector3D] Lecture TGA : mirroir horizontal non supporte "<<endl;
	}

	cerr<<"Taille : "<<tailu<<" / "<<tailv<<endl;
	d = tgaheader.Header[16];
	cerr<<"Origine "<<xorigine<<" / "<<yorigine<<endl;
	if ((xorigine != 0) || (yorigine != 0)) {
	cerr<<"[VBVector3D] Lecture TGA : pas de gestion du decalage ... ! Aborting !"<<endl;
	fclose(file);
	exit(5);
	}
	if (tgaheader.Header[2] == uTGAcompare) {
		lectureTgaNoCompress(file,d);
	}
	else if (tgaheader.Header[2] == cTGAcompare) {
		lectureTgaCompress(file,d);
	}
	else {
		cerr<<"[VBVector3D] Lecture TGA : Error TGA file be type 2 (colour no compress) or type 10 (colour compress)"<<endl;
		fclose(file);
		exit(5);
	}
}

void ImageTex::lectureTgaNoCompress(FILE *file,int d) {
	unsigned int cswap;
	nbChannel = 3;

	if((tailu <= 0) || (tailv <= 0) || ((d != 24) && (d !=32)))	{
		cerr<<"[VBVector3D] Lecture TGA : Error invalid texture information"<<endl;
		if(file != NULL) fclose(file);
		exit(5);
	}

	if(d == 32) {
		typetext = TEX_TYPE_RVBA;
		nbChannel = 4;
	}
	else {
		typetext = TEX_TYPE_RVB;
	}

	unsigned int size = nbChannel * tailu * tailv;
	tabRVB	= new unsigned char[size];

	if(tabRVB == NULL) {
		cerr<<"[VBVector3D] Lecture TGA : Error could not allocate memory for image"<<endl;
		fclose(file);
		exit(5);
	}

	if(fread(tabRVB, 1, size, file) != size) {
		cerr<<"[VBVector3D] Lecture TGA : Error could not read image data"<<endl;
		if (tabRVB != NULL) delete[](tabRVB);
		tabRVB = NULL;
		fclose(file);
		exit(5);
	}

	for(cswap = 0; cswap < size; cswap += tailu*tailv)
		tabRVB[cswap] ^= tabRVB[cswap+2] ^= tabRVB[cswap] ^= tabRVB[cswap+2];

}

void ImageTex::lectureTgaCompress(FILE *file,int d) {
	unsigned int pixelcount;
	unsigned int currentpixel	= 0;
	unsigned int currentbyte	= 0;
	unsigned char* colorbuffer;
	int size;
	int bpp;


	pixelcount = tailu * tailv;

	if((tailu <= 0) || (tailv <= 0) || ((d != 24) && (d !=32)))	{
		cerr<<"Error invalid texture information"<<endl;
		if(file != NULL) fclose(file);
		exit(5);
	}

	bpp	= (d / 8);
	size	= (bpp * tailu * tailv);
	tabRVB = new unsigned char[size];
	colorbuffer = (unsigned char *)malloc(bpp);

	if(bpp == 4) {
		typetext = TEX_TYPE_RVBA;
		nbChannel = 4;
	}
	else {
		typetext = TEX_TYPE_RVB;
		nbChannel = 3;
	}

	if(tabRVB == NULL) {
		cerr<<"[VBVector3D] Lecture TGA : Error could not allocate memory for image"<<endl;
		fclose(file);
		exit(5);
	}

	do {
		unsigned char chunkheader = 0;
		if(fread(&chunkheader, sizeof(unsigned char), 1, file) == 0) {
	  	cerr<<"[VBVector3D] Lecture TGA : Error could not read RLE header"<<endl;
	  	if(file != NULL) fclose(file);
			if(tabRVB != NULL) delete[](tabRVB);
			tabRVB = NULL;
			exit(5);
		}

		if(chunkheader < 128) {
			short counter;
			chunkheader++;
			for(counter = 0; counter < chunkheader; counter++) {
				if(fread(colorbuffer, 1, bpp, file) != (unsigned int)bpp ) {
					cerr<<"[VBVector3D] Lecture TGA : Error could not read image data"<<endl;
					if(file != NULL) fclose(file);
					if(colorbuffer != NULL) free(colorbuffer);
					if(tabRVB != NULL) delete[](tabRVB);
					tabRVB = NULL;
					exit(5);
				}
				/* write to memory */
				tabRVB[currentbyte		]   = colorbuffer[2];
				tabRVB[currentbyte + 1	] = colorbuffer[1];
				tabRVB[currentbyte + 2	] = colorbuffer[0];

				if(bpp == 4)
				{
					tabRVB[currentbyte + 3] = colorbuffer[3];
				}

				currentbyte += bpp;
				currentpixel++;

				if(currentpixel > pixelcount) {
					cerr<<"[VBVector3D] Lecture TGA : Error too many pixels read"<<endl;
					if(file != NULL) fclose(file);
					if(colorbuffer != NULL) free(colorbuffer);
					if(tabRVB != NULL) delete[](tabRVB);
					tabRVB = NULL;
					exit(5);
				}
			}
		}
		else {
			short counter;
			chunkheader -= 127;
			if(fread(colorbuffer, 1, bpp, file) != (unsigned int)bpp){
				cerr<<"[VBVector3D] Lecture TGA : Error could not read image data"<<endl;
				if(file != NULL) fclose(file);
				if(colorbuffer != NULL) free(colorbuffer);
				if(tabRVB != NULL) delete[](tabRVB);
				tabRVB = NULL;
				exit(5);
			}

			for(counter = 0; counter < chunkheader; counter++) {
				tabRVB[currentbyte		] = colorbuffer[2];
				tabRVB[currentbyte + 1	] = colorbuffer[1];
				tabRVB[currentbyte + 2	] = colorbuffer[0];
				if(bpp == 4) tabRVB[currentbyte + 3] = colorbuffer[3];

				currentbyte += bpp;
				currentpixel++;
				if(currentpixel > pixelcount){
					cerr<<"[VBVector3D] Lecture TGA : Error too many pixels read"<<endl;
					if(file != NULL) fclose(file);
					if(colorbuffer != NULL) free(colorbuffer);
					if(tabRVB != NULL) delete[](tabRVB);
					tabRVB = NULL;
					exit(5);
				}
			}
		}
	} while(currentpixel < pixelcount);
}

unsigned char* loadFromJPG(string* nmfile,int& size_w,int& size_h,unsigned int& typeimg) {
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	FILE * infile;      	/* source file */
	JSAMPARRAY buffer;  	/* Output row buffer */
	int row_stride;	    	/* physical row width in output buffer */
	int typtmp;
	int nb_of_composant;
	unsigned char* image_buffer;

	if ((infile = fopen(nmfile->c_str(), "rb")) == NULL) {
		std::cerr<<"Impossible d'ouvrir "<<nmfile<<std::endl;
		exit(1);
	}

	cinfo.err = jpeg_std_error(&jerr);
	/* Now we can initialize the JPEG decompression object. */
	jpeg_create_decompress(&cinfo);

	/* Step 2: specify data source (eg, a file) */
	jpeg_stdio_src(&cinfo, infile);

	/* Step 3: read file parameters with jpeg_read_header() */
	(void) jpeg_read_header(&cinfo, TRUE);

	size_h = cinfo.image_height;
	size_w = cinfo.image_width;
	nb_of_composant = cinfo.num_components;
	typtmp = cinfo.jpeg_color_space;
	switch(typtmp) {
		case JCS_GRAYSCALE :
			if (nb_of_composant != 1) {
				std::cerr<<"Format JPG differe du nombre de composant par pixel !"<<std::endl;
				exit(7);
			}
			typeimg = TYPE_JPG_L;
			break;
		case JCS_RGB :
		case JCS_YCbCr :
			if (nb_of_composant != 3) {
				std::cerr<<"Format JPG differe du nombre de composant par pixel !"<<std::endl;
				exit(7);
			}
			typeimg = TYPE_JPG_RVB;
			break;
		case JCS_CMYK :
			if (nb_of_composant != 4) {
				std::cerr<<"Format JPG differe du nombre de composant par pixel !"<<std::endl;
				exit(7);
			}
			typeimg = TYPE_JPG_RVBA;
			break;
		case JCS_YCCK :
			std::cerr<<"Format JPG non traite pour le moment ("<<typtmp<<")!"<<std::endl;
			exit(7);
		case JCS_UNKNOWN :
		default :
			std::cerr<<"Format Inconnu de la librairie !!!"<<std::endl;
	}

	/* Step 4: set parameters for decompression */
	/* In this example, we don't need to change any of the defaults set by
	* jpeg_read_header(), so we do nothing here.
	*/

	/* Step 5: Start decompressor */

	(void) jpeg_start_decompress(&cinfo);

	//std::cerr<<"We got : "<<size_w<<" / "<<size_h<<" vs "<<cinfo.output_width<<"/"<<cinfo.output_height<<std::endl;

	if ((cinfo.output_width != (unsigned int)size_w) ||
		(cinfo.output_height != (unsigned int)size_h) ||
		(cinfo.out_color_components != nb_of_composant)) {
		std::cerr<<"Difference entre l'entete JPG et le fichier JPG ! Aborting loading !";
		std::cerr<<std::endl;
	}

	/* CREATION DE L'IMAGE */
	image_buffer = new unsigned char[size_w*size_h*nb_of_composant];

	/* We may need to do some setup of our own at this point before reading
	* the data.  After jpeg_start_decompress() we have the correct scaled
	* output image dimensions available, as well as the output colormap
	* if we asked for color quantization.
	* In this example, we need to make an output work buffer of the right size.
	*/
	/* JSAMPLEs per row in output buffer */
	row_stride = size_w * nb_of_composant;
	/* Make a one-row-high sample array that will go away when done with image */
	buffer = (*cinfo.mem->alloc_sarray) ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

	/* Step 6: while (scan lines remain to be read) */
	/*           jpeg_read_scanlines(...); */

	/* Here we use the library's state variable cinfo.output_scanline as the
	* loop counter, so that we don't have to keep track ourselves.
	*/
	unsigned char *ptr = image_buffer+(cinfo.output_height-1)*row_stride;
	while (cinfo.output_scanline < cinfo.output_height) {
		/* jpeg_read_scanlines expects an array of pointers to scanlines.
		* Here the array is only one element long, but you could ask for
		* more than one scanline at a time if that's more convenient.
		*/
		(void) jpeg_read_scanlines(&cinfo, buffer, 1);
		/* Assume put_scanline_someplace wants a pointer and sample count. */
		memcpy(ptr,buffer[0],row_stride);
		ptr -= row_stride;
	}

	/* Step 7: Finish decompression */

	(void) jpeg_finish_decompress(&cinfo);

	/* Step 8: Release JPEG decompression object */

	/* This is an important step since it will release a good deal of memory. */
	jpeg_destroy_decompress(&cinfo);

	/* SI LE TYPE ETAIT YUV alors faire la conversion ! */
	//CompImage tool;
	//image_buffer = tool.convertYUVtoRVB(image_buffer,image_width,image_height);

	fclose(infile);

	return image_buffer;
}

void ImageTex::lectureJpg(string* nmfile) {
	unsigned int atype;
	tabRVB = loadFromJPG(nmfile,tailu,tailv,atype);
	switch(atype) {
		case TYPE_JPG_NULL :
			typetext = TEX_TYPE_NULL;
			std::cerr<<"[VBVector3D : ImageTex] Lecture JPG : Le format d'image jpg pour ("<<nmfile<<") est vide ! Aborting"<<endl;
			exit(5);
		case TYPE_JPG_RVB :
			typetext = TEX_TYPE_RVB;
			nbChannel = 3;
			break;
		case TYPE_JPG_RVBA :
		case TYPE_JPG_CMYK :
			typetext = TEX_TYPE_RVBA;
			nbChannel = 4;
			break;
		case TYPE_JPG_L :
			typetext = TEX_TYPE_LUM;
			nbChannel = 1;
			break;
		default :
			cerr<<"[VBVector3D] Lecture JPG : Unable to find a correct type image in loading jpg ! Aborting !"<<endl;
			exit(1);
	}
}

bool ImageTex::lectureBmp(FILE *fd)
{
	t_bmp bmp;
	int cpttmp;
	// LECTURE PREMIER HEADER
	cpttmp = fread(&(bmp.header.first_header.sType), sizeof(short), 1, fd); // 0 - 1
	cpttmp = fread(&(bmp.header.first_header.iSize), sizeof(int), 1, fd); // 2 - 5
	cpttmp = fread(&(bmp.header.first_header.sReserved1), sizeof(short), 1, fd); // 6 - 7
	cpttmp = fread(&(bmp.header.first_header.sReserved2), sizeof(short), 1, fd); // 8 - 9
	cpttmp = fread(&(bmp.header.first_header.iOffBits), sizeof(int), 1, fd); // 10 - 13
	/*
	if (bmp.header.first_header.iOffBits > 0) {
		cerr<<"[VBVector3D] Chargement BMP : OffBits > 0. Non géré... Aborting !"<<bmp.header.first_header.iOffBits<<endl;
		return false;
	}
	*/

	// LECTURE SECOND HEADER
	cpttmp = fread(&(bmp.header.second_header.iSize), sizeof(int), 1, fd); // 14 - 17
	cpttmp = fread(&(bmp.header.second_header.iWidth), sizeof(int), 1, fd); // 18 - 21
	cpttmp = fread(&(bmp.header.second_header.iHeight), sizeof(int), 1, fd); // 22 - 25
	cpttmp = fread(&(bmp.header.second_header.sPlanes), sizeof(short), 1, fd); // 26 - 27
	cpttmp = fread(&(bmp.header.second_header.sBitCount), sizeof(short), 1, fd); // 28 - 29
	cpttmp = fread(&(bmp.header.second_header.iCompression), sizeof(int), 1, fd); // 30 - 33
	cpttmp = fread(&(bmp.header.second_header.iSizeImage), sizeof(int), 1, fd); // 34 - 37
	cpttmp = fread(&(bmp.header.second_header.iXpelsPerMeter), sizeof(int), 1, fd); // 38 - 41
	cpttmp = fread(&(bmp.header.second_header.iYpelsPerMeter), sizeof(int), 1, fd); // 42 - 45
	cpttmp = fread(&(bmp.header.second_header.iClrUsed), sizeof(int), 1, fd); // 46 - 49
	cpttmp = fread(&(bmp.header.second_header.iClrImportant), sizeof(int), 1, fd); // 50 - 53

	if (bmp.header.second_header.sPlanes != 1) {
		cerr<<"[VBVector3D] Chargement BMP : Erreur interne du fichier image BMP"<<endl;
		return false;
	}
	if (bmp.header.second_header.iCompression != 0) {
		if (bmp.header.second_header.iCompression > 3)
			cerr<<"[VBVector3D] Chargement BMP : Erreur interne du fichier image BMP"<<endl;
		if (bmp.header.second_header.iCompression == 1)
			cerr<<"[VBVector3D] Chargement BMP : Compression RLE de 8 bits par pixel non gérée"<<endl;
		if (bmp.header.second_header.iCompression == 2)
			cerr<<"[VBVector3D] Chargement BMP : Compression RLE de 4 bits par pixel non gérée"<<endl;
		if (bmp.header.second_header.iCompression == 3)
			cerr<<"[VBVector3D] Chargement BMP : Compression par palette non géré"<<endl;
		return false;
	}
	if (bmp.header.second_header.iClrUsed > 0) {
		cerr<<"[VBVector3D] Chargement BMP : Compression par palette non géré"<<endl;
		return false;
	}

	tailu = bmp.width = bmp.header.second_header.iWidth;
	tailv = bmp.height = bmp.header.second_header.iHeight;
	bmp.width_useless = bmp.width % 4;
	if (bmp.width_useless != 0) {
		cerr<<"[VBVector3D] : Erreur interne du fichier image BMP width_useless != 0"<<endl;
		exit(5);
	}

	//cerr<<"TAILLE D'UN UNSIGNED CHAR "<<sizeof(unsigned char)<<endl;
	unsigned int size = (bmp.header.second_header.sBitCount * tailu * tailv)/(8*sizeof(unsigned char));
	if ((size*sizeof(unsigned char)) != (unsigned int)bmp.header.second_header.iSizeImage) {
		cerr<<"[VBVector3D] Chargement BMP : Erreur interne du fichier image BMP ";
		cerr<<"entre taille calcule "<<bmp.header.second_header.sBitCount<<"*"<<tailu<<"*"<<tailv<<" et ";
		cerr<<"la taille stockée "<<bmp.header.second_header.iSizeImage<<endl;
		//return false;
	}
	unsigned int size_one_pixel = 0;
	switch (bmp.header.second_header.sBitCount) {
		case (1) :
			cerr<<"[VBVector3D] Chargement BMP : Format binaire non géré"<<endl;
			return false;
			break;
		case (4) :
			cerr<<"[VBVector3D] Chargement BMP : Format 16 couleur non géré"<<endl;
			return false;
			break;
		case (8) :
			size_one_pixel = 1;
			typetext = TEX_TYPE_LUM;
			nbChannel = 1;
			break;
		case (16) :
			cerr<<"[VBVector3D] Chargement BMP : Formats 16/24 bits/pixel non gérés"<<endl;
			return false;
			break;
		case (24) :
			size_one_pixel = 3;
			nbChannel = 3;
			typetext = TEX_TYPE_RVB;
			break;
		case (32) :
			size_one_pixel = 4;
			nbChannel = 4;
			typetext = TEX_TYPE_RVBA;
			break;
		default :
			cerr<<"[VBVector3D] Chargement BMP : Erreur interne du fichier image BMP"<<endl;
			return false;
	}
/*	cerr<<"Taille un pixel : "<<size_one_pixel<<endl;
	cerr<<"Taille de l'image : "<<tailu<<"*"<<tailv<<endl;
	cerr<<"Taille de l'image en octet : "<<size_one_pixel*tailu*tailv<<endl;
	cerr<<"Taille préindiquée dans le fichier : "<<(unsigned int)bmp.header.second_header.iSizeImage<<endl;*/
	tabRVB = new unsigned char[size_one_pixel*tailu*tailv];
	if (!tabRVB) {
		cerr<<"[VBVector3D] Chargement BMP : No more memory !"<<endl;
		return false;
	}
	/*
	for(int i=0;i<tailu*tailv;i++) {
		tabRVB[3*i+2] = (unsigned char) fgetc(fd);
		tabRVB[3*i+1] = (unsigned char) fgetc(fd);
		tabRVB[3*i]   = (unsigned char) fgetc(fd);
	}
	*/
	cpttmp = fread(tabRVB,sizeof(unsigned char),tailu*tailv*size_one_pixel,fd);
	// ATTENTION C'EST DU BGR...
	/*
	if (size_one_pixel>1)
		CompImage::convertBGRtoRGB(tabRVB,tailu,tailv,size_one_pixel);
	*/

	//char nammme[150];
	//sprintf(nammme,"./test%d.ppm",(int)fd);
	//CompImage::makePpmP6(tailu,tailv,tabRVB,nammme);

	return true;
}


/* *************************************************************************************
 * ********** TOOL FUNCTIONS
 * ************************************************************************************* */
void ImageTex::flip() {
	size_t ind_row;
	size_t size_a_row = nbChannel*tailu;
	unsigned char* a_row = new unsigned char[size_a_row];
	for(ind_row = 0;ind_row < (unsigned int)tailv/2;++ind_row) {
		memcpy(a_row,tabRVB+ind_row*size_a_row,size_a_row*sizeof(unsigned char));
		memcpy(tabRVB+ind_row*size_a_row,tabRVB+(tailv-1-ind_row)*size_a_row,size_a_row*sizeof(unsigned char));
		memcpy(tabRVB+(tailv-1-ind_row)*size_a_row,a_row,size_a_row*sizeof(unsigned char));
	}
}

void ImageTex::saveToPXM(const char* nameFile) {
	FILE* f = fopen(nameFile,"wb");
	if (!f) {
		std::cerr<<"Error opening file for writing "<<nameFile<<std::endl;
		return; 
	}
	
	if (nbChannel == 1) fprintf(f,"P5\n#Saving from imageTexture\n%d %d\n255\n",tailu,tailv);
	else fprintf(f,"P6\n#Saving from imageTexture\n%d %d\n255\n",tailu,tailv);
	fwrite(tabRVB,tailu*tailv*nbChannel,sizeof(unsigned char),f);
	fclose(f);
}
