/***************************************************************************
                      imageComputation.cpp  -  description
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

#include "imageComputation.hpp"
#include "interface.hpp"
#include "display.hpp"

/// ///////////////////////////////////////////////////////////////////////////
/// Corps des fonctions utilitaires
/// ///////////////////////////////////////////////////////////////////////////
void drawOneFrontQuad() {
	glBegin(GL_QUADS);
		glMultiTexCoord2fARB(GL_TEXTURE0,0.0, 0.0);
		glVertex2f(0.0, 0.0);
		glMultiTexCoord2fARB(GL_TEXTURE0,0.0, 1.0);
		glVertex2f(0.0, 1.0);
		glMultiTexCoord2fARB(GL_TEXTURE0,1.0, 1.0);
		glVertex2f(1.0, 1.0);
		glMultiTexCoord2fARB(GL_TEXTURE0,1.0, 0.0);
		glVertex2f(1.0, 0.0);
	glEnd();
}

void startFBOComputation(ImageTex* img_src,ImageTex* img_res) {
	// FIXATION OFFLINE RENDERING
	glBindFramebuffer(GL_FRAMEBUFFER,fbo_handler);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,img_res->indbind,0);
	glClearColor(1.0,1.0,0.0,0.0);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// CALCUL DE LA PROJECTION
	glViewport( 0, 0, (GLint)img_src->tailu, (GLint)img_src->tailv );
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0,-1.,1.);
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	// FIXATION DE QQ PARAMETRE GL
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	// CHARGEMENT DU VERTEX & FRAGMENT SHADER
	glUseProgram(testShader);

}

void endFBOComputation() {
	// DECHARGEMENT VERTEX & FRAGMENT SHADER
	glUseProgram(0);
	// RETOUR PARAMETRE GL
	glEnable(GL_DEPTH_TEST);

	// RETOUR PROJECTION
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();

	// RETOUR FBO CLASSIQUE
	glBindFramebuffer(GL_FRAMEBUFFER,0);
}

/// ///////////////////////////////////////////////////////////////////////////
/// Corps des fonctions de calcul
/// ///////////////////////////////////////////////////////////////////////////
void makeColorSwitch(ImageTex* img_src,ImageTex* img_res) {
	std::cerr<<"makeColorSwitch (CPU) !"<<std::endl;
	
	for(int i = 0; i < img_res->tailv; i++){
		for(int j = 0; j < img_res->tailu; j++){
			unsigned char pixel_r = img_src->get(i, j, 0);
			unsigned char pixel_g = img_src->get(i, j, 1);
			unsigned char pixel_b = img_src->get(i, j, 2);
			img_res->set(i, j, 0, pixel_b);
			img_res->set(i, j, 1, pixel_g);
			img_res->set(i, j, 2, pixel_r);
		}
	}
	
	modified_tampon = true;
}

void makeColorSwitchGPU(ImageTex* img_src,ImageTex* img_res) {
	// Activation du FBO...
	startFBOComputation(img_src,img_res);

	// DO SOME WORK...
	std::cerr<<"makeColorSwitch (GPU) !"<<std::endl;
	
	// CHARGEMENT DE LA TEXTURE SOURCE...
	img_src->chargeTexture();

	// FIXATION DES PARAMETRES SHADERS (basetex ici)
	GLint location = glGetUniformLocation(testShader, "usampler");
    glUniform1i(location, 0);
	
	// Affichage d'un quadrilatère devant l'écran de la caméra virtuelle
	drawOneFrontQuad();

	// DECHARGEMENT DE LA TEXTURE SOURCE...
	img_src->dechargeTexture();

	// DECHARGEMENT DU FBO
	endFBOComputation();
}

void makeLuminance(ImageTex* img_src,ImageTex* img_res) {
	// Luminance = 0.3 R + 0.59 G + 0.11 B
	std::cerr<<"makeLuminance (CPU) !"<<std::endl;
	
	for(int i = 0; i < img_res->tailv; i++){
		for(int j = 0; j < img_res->tailu; j++){
			unsigned char pixel_r = img_src->get(i, j, 0);
			unsigned char pixel_b = img_src->get(i, j, 1);
			unsigned char pixel_g = img_src->get(i, j, 2);
			unsigned char luminance = 0.3 * pixel_r + 0.59 * pixel_g + 0.11 * pixel_b;
			img_res->set(i, j, 0, luminance);
			img_res->set(i, j, 1, luminance);
			img_res->set(i, j, 2, luminance);
		}
	}

	modified_tampon = true;
}

void makeLuminanceGPU(ImageTex* img_src,ImageTex* img_res) {
	// Activation du FBO...
	startFBOComputation(img_src,img_res);

	// DO SOME WORK...
	std::cerr<<"makeLuminance (GPU) !"<<std::endl;
	
	// CHARGEMENT DE LA TEXTURE SOURCE...
	img_src->chargeTexture();
	
	// FIXATION DES PARAMETRES SHADERS
	GLint location = glGetUniformLocation(testShader, "usampler");
    glUniform1i(location, 0);
	//glUniform1i(glGetUniformLocation(testShader, "usampler"), 0);
	
	// Affichage d'un quadrilatère devant l'écran de la caméra virtuelle
	drawOneFrontQuad();

	// DECHARGEMENT DE LA TEXTURE SOURCE...
	img_src->dechargeTexture();
	
	// DECHARGEMENT DU FBO
	endFBOComputation();
}

void makeSepia(ImageTex* img_src,ImageTex* img_res,float* sepiaColor) {
	// L = 0.3 R + 0.59 G + 0.11 B
	// R = (1 + S_r) * L
	// G = (1 + S_g) * L
	// G = (1 + S_b) * L
	std::cerr<<"makeSepia (CPU) !"<<std::endl;

	for(int i = 0; i < img_res->tailv; i++){
		for(int j = 0; j < img_res->tailu; j++){
			unsigned char pixel_r = img_src->get(i, j, 0);
			unsigned char pixel_b = img_src->get(i, j, 1);
			unsigned char pixel_g = img_src->get(i, j, 2);
			float luminance = 0.3 * pixel_r + 0.59 * pixel_g + 0.11 * pixel_b;
			unsigned int r = (1 + sepiaColor[0]) * luminance;
			if(r > 255)
			{
				r = 255;
			}
			unsigned int g = (1 + sepiaColor[1]) * luminance;
			if(g > 255)
			{
				g = 255;
			}
			unsigned int b = (1 + sepiaColor[2]) * luminance;
			if(b > 255)
			{
				b = 255;
			}
			img_res->set(i, j, 0, (char)r);
			img_res->set(i, j, 1, (char)g);
			img_res->set(i, j, 2, (char)b);
		}
	}

	modified_tampon = true;
}

void makeSepiaGPU(ImageTex* img_src,ImageTex* img_res,float* sepiaColor) {
	// Activation du FBO...
	startFBOComputation(img_src,img_res);

	// DO SOME WORK... Ne pas oublier de transmettre sepiaColor
	std::cerr<<"makeSepia (GPU) !"<<std::endl;
	// CHARGEMENT DE LA TEXTURE SOURCE...
	img_src->chargeTexture();
	
	// FIXATION DES PARAMETRES SHADERS
    glUniform1i(glGetUniformLocation(testShader, "usampler"), 0);
    glUniform3fv(glGetUniformLocation(testShader, "sepiaColor"), 3, sepiaColor);
	
	// Affichage d'un quadrilatère devant l'écran de la caméra virtuelle
	drawOneFrontQuad();

	// DECHARGEMENT DE LA TEXTURE SOURCE...
	img_src->dechargeTexture();

	// DECHARGEMENT DU FBO
	endFBOComputation();
}

void makeBlur(ImageTex* img_src,ImageTex* img_res) {
	// Simple flou : on additionne au pixel ses 8 pixels voisins (et on divise par 9 bien sûr)
	std::cerr<<"makeBlur (CPU) !"<<std::endl;

	float tmp_blur_r = 0;
	float tmp_blur_g = 0;
	float tmp_blur_b = 0;

	for(int i = 1; i < img_res->tailv; i++){

		for(int j = 1; j < img_res->tailu; j++){
			
			for( int k = i-1; k <= i+1; k++) {

				for (int l = j-1; l<= j+1; l++) {

					tmp_blur_r += img_src->get(k, l, 0);
					tmp_blur_g += img_src->get(k, l, 1);
					tmp_blur_b += img_src->get(k, l, 2);

				}

			}

			img_res->set(i, j, 0, (char)(tmp_blur_r/9.));
			img_res->set(i, j, 1, (char)(tmp_blur_g/9.));
			img_res->set(i, j, 2, (char)(tmp_blur_b/9.));

			tmp_blur_r = 0;
			tmp_blur_g = 0;
			tmp_blur_b = 0;

		}

	}

	modified_tampon = true;
}

void makeBlurGPU(ImageTex* img_src,ImageTex* img_res) {
	// Activation du FBO...
	startFBOComputation(img_src,img_res);
	std::cerr<<"makeBlur (GPU) !"<<std::endl;

	// CHARGEMENT DE LA TEXTURE SOURCE...
	img_src->chargeTexture();
	
	// FIXATION DES PARAMETRES SHADERS
    glUniform1i(glGetUniformLocation(testShader, "usampler"), 0);
	glUniform1f(glGetUniformLocation(testShader, "width"), img_src->tailv);
	glUniform1f(glGetUniformLocation(testShader, "height"), img_src->tailu);

	// Affichage d'un quadrilatère devant l'écran de la caméra virtuelle
	drawOneFrontQuad();

	// DECHARGEMENT DE LA TEXTURE SOURCE...
	img_src->dechargeTexture();

	// DECHARGEMENT DU FBO
	endFBOComputation();
}

void makeSobelFilter(ImageTex* img_src,ImageTex* img_res) {
	std::cerr<<"makeSobelFilter (CPU) !"<<std::endl;

	float tabLuminance[3][3];

	for(int i = 1; i < img_res->tailv; i++){
		for(int j = 1; j < img_res->tailu; j++){
			
			// matrice A
			unsigned int index_x = 0;
			unsigned int index_y = 0;
			for(int k = i-1; k < i+1; k++) {
				for(int l = j-1; l< j+1; l++) {					
					unsigned char pixel_r = img_src->get(k, l, 0);
					unsigned char pixel_b = img_src->get(k, l, 1);
					unsigned char pixel_g = img_src->get(k, l, 2);
					float luminance = 0.3 * pixel_r + 0.59 * pixel_g + 0.11 * pixel_b;
					tabLuminance[index_x][index_y] = luminance;
					index_y += 1;
				}
				index_x += 1;
				index_y = 0;
			}

			// compute masks (horizontal and vertical)
			float gx = 2. * tabLuminance[1][0] - 2. * tabLuminance[1][1];
			float gy = 0.;
			
			// norme of the gradient
			float g = sqrt(gx * gx + gy * gy);
			if(g > 255.){g = 255.;}

			// set pixel
			img_res->set(i, j, 0, (unsigned char)g);
			img_res->set(i, j, 1, (unsigned char)g);
			img_res->set(i, j, 2, (unsigned char)g);
		}
	}

	modified_tampon = true;
}

void makeSobelFilterGPU(ImageTex* img_src,ImageTex* img_res) {
	// Activation du FBO...
	startFBOComputation(img_src,img_res);
	std::cerr<<"makeSobelFilter (GPU) !"<<std::endl;

	// CHARGEMENT DE LA TEXTURE SOURCE...
	img_src->chargeTexture();
	
	// FIXATION DES PARAMETRES SHADERS
   	glUniform1i(glGetUniformLocation(testShader, "usampler"), 0);
	glUniform1f(glGetUniformLocation(testShader, "width"), img_src->tailv);
	glUniform1f(glGetUniformLocation(testShader, "height"), img_src->tailu);

	// Affichage d'un quadrilatère devant l'écran de la caméra virtuelle
	drawOneFrontQuad();

	// DECHARGEMENT DE LA TEXTURE SOURCE...
	img_src->dechargeTexture();

	// DECHARGEMENT DU FBO
	endFBOComputation();
}

void makeConvolution3x3(ImageTex* img_src,ImageTex* img_res) {
	std::cerr<<"makeConvolution3x3 (CPU) !"<<std::endl;

	/*float kernel[3][3] = {
			{-1.,-1., -1.},
			{-1., 9., -1.},
			{-1.,-1., -1.}
		};
	*/
	
	// repoussage
	float kernel[3][3] = {
			{-2., -1., 0.},
			{-1., 1.,  1.},
			{0.,  1.,  2.}
		};

	float res_r = 0;
	float res_g = 0;
	float res_b = 0;

	unsigned int index_x = 0;
	unsigned int index_y = 0;

	for(int i = 1; i < img_res->tailv; i++){//height
		for(int j = 1; j < img_res->tailu; j++){//width
			
			for(int k = i-1; k <= i+1; k++) {//height
				for (int l = j-1; l <= j+1; l++) {//width
					res_r += img_src->get(k, l, 0) * kernel[index_x][index_y];
					res_g += img_src->get(k, l, 1) * kernel[index_x][index_y];
					res_b += img_src->get(k, l, 2) * kernel[index_x][index_y];

					index_y++;
				}
				index_x++;
				index_y = 0;
			}
			index_x = 0;

			if(res_r < 0.) res_r = 0.;
			else if(res_r > 255.) res_r = 255.;
			if(res_g < 0.) res_g = 0.;
			else if(res_g > 255.) res_g = 255.;
			if(res_b < 0.) res_b = 0.;
			else if(res_b > 255.) res_b = 255.;

			img_res->set(i, j, 0, (unsigned char)(res_r));
			img_res->set(i, j, 1, (unsigned char)(res_g));
			img_res->set(i, j, 2, (unsigned char)(res_b));

			res_r = 0.;
			res_g = 0.;
			res_b = 0.;
		}
	}

	modified_tampon = true;
}

void makeConvolution3x3GPU(ImageTex* img_src,ImageTex* img_res) {
	// Activation du FBO...
	startFBOComputation(img_src,img_res);
	std::cerr<<"makeConvolution3x3 (GPU) !"<<std::endl;

	// CHARGEMENT DE LA TEXTURE SOURCE...
	img_src->chargeTexture();
	
	// FIXATION DES PARAMETRES SHADERS
   	glUniform1i(glGetUniformLocation(testShader, "usampler"), 0);
	glUniform1f(glGetUniformLocation(testShader, "width"), img_src->tailv);
	glUniform1f(glGetUniformLocation(testShader, "height"), img_src->tailu);
	// repoussage
	float kernel[3][3] = {
			{-2., -1., 0.},
			{-1., 1.,  1.},
			{0.,  1.,  2.}
		};
	glUniformMatrix3fv(glGetUniformLocation(testShader, "conv3x3_kernel"), 1, GL_FALSE, &kernel[0][0]);

	// Affichage d'un quadrilatère devant l'écran de la caméra virtuelle
	drawOneFrontQuad();

	// DECHARGEMENT DE LA TEXTURE SOURCE...
	img_src->dechargeTexture();

	// DECHARGEMENT DU FBO
	endFBOComputation();
}


void makeConvolution5x5(ImageTex* img_src,ImageTex* img_res) {
	std::cerr<<"makeConvolution5x5 (CPU) !"<<std::endl;
	
	// augmenter le contraste
	float kernel[5][5] = {
			{0.,  0.,  0.,  0., 0.},
			{0.,  0., -1.,  0., 0.},
			{0., -1.,  5., -1., 0.},
			{0.,  0., -1.,  0., 0.},
			{0.,  0.,  0.,  0., 0.}
		};

	float res_r = 0;
	float res_g = 0;
	float res_b = 0;

	unsigned int index_x = 0;
	unsigned int index_y = 0;

	for(int i = 1; i < img_res->tailv; i++){//height
		for(int j = 1; j < img_res->tailu; j++){//width
			
			for(int k = i-2; k <= i+2; k++) {//height
				for (int l = j-2; l <= j+2; l++) {//width
					res_r += img_src->get(k, l, 0) * kernel[index_x][index_y];
					res_g += img_src->get(k, l, 1) * kernel[index_x][index_y];
					res_b += img_src->get(k, l, 2) * kernel[index_x][index_y];

					index_y++;
				}
				index_x++;
				index_y = 0;
			}
			index_x = 0;

			if(res_r < 0.) res_r = 0.;
			else if(res_r > 255.) res_r = 255.;
			if(res_g < 0.) res_g = 0.;
			else if(res_g > 255.) res_g = 255.;
			if(res_b < 0.) res_b = 0.;
			else if(res_b > 255.) res_b = 255.;

			img_res->set(i, j, 0, (unsigned char)(res_r));
			img_res->set(i, j, 1, (unsigned char)(res_g));
			img_res->set(i, j, 2, (unsigned char)(res_b));

			res_r = 0.;
			res_g = 0.;
			res_b = 0.;
		}
	}

	modified_tampon = true;
}

void makeConvolution5x5GPU(ImageTex* img_src,ImageTex* img_res) {
	// Activation du FBO...
	startFBOComputation(img_src,img_res);
	std::cerr<<"makeConvolution5x5 (GPU) !"<<std::endl;

	// CHARGEMENT DE LA TEXTURE SOURCE...
	img_src->chargeTexture();
	
	// FIXATION DES PARAMETRES SHADERS
   	glUniform1i(glGetUniformLocation(testShader, "usampler"), 0);
	glUniform1f(glGetUniformLocation(testShader, "width"), img_src->tailv);
	glUniform1f(glGetUniformLocation(testShader, "height"), img_src->tailu);
	// augmenter le contraste
	// ATTENTION : impossible d'envoyer une matrice 5x5 en variable uniform !
	float kernel[25] = {
			0.,  0.,  0.,  0., 0.,
			0.,  0., -1.,  0., 0.,
			0., -1.,  5., -1., 0.,
			0.,  0., -1.,  0., 0.,
			0.,  0.,  0.,  0., 0.
		};
	glUniform1fv(glGetUniformLocation(testShader, "conv5x5_kernel"), 25, kernel);

	// Affichage d'un quadrilatère devant l'écran de la caméra virtuelle
	drawOneFrontQuad();

	// DECHARGEMENT DE LA TEXTURE SOURCE...
	img_src->dechargeTexture();

	// DECHARGEMENT DU FBO
	endFBOComputation();
}

