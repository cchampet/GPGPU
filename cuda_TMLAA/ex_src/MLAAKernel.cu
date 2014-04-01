/***************************************************************************
                          vectorTestKernel.cu
                          -------------------
begin                : Mon Oct 20 2010
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

#ifndef _KERNEL_H_
#define _KERNEL_H_


//best structure : Map (one-to-one)
__device__ void RGBtoXYZ(unsigned char* g_idata, float* g_odata) {
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int indice = 3*indiceX + 3*indiceY * width;

	unsigned char R = g_idata[indice];
	unsigned char G = g_idata[indice + 1];
	unsigned char B = g_idata[indice + 2];

	// R between 0 and 255 
	float var_R = ( R / 255.f);
	// G between 0 and 255
	float var_G = ( G / 255.f);
	// B between 0 and 255
	float var_B = ( B / 255.f);

	if (var_R > 0.04045f) { var_R = pow((var_R * 0.055f) / 1.055f, 2.4f); }
	else				  { var_R = var_R / 12.92f; }

	if (var_G > 0.04045f) { var_G = pow((var_G * 0.055f) / 1.055f, 2.4f); }
	else 				  { var_G = var_G / 12.92f; }

	if (var_B > 0.04045f) { var_B = pow((var_B * 0.055f) / 1.055f, 2.4f); }
	else 				  { var_B = var_B / 12.92f; }

	var_R = var_R * 100;
	var_G = var_G * 100;
	var_B = var_B * 100;

	float X = var_R * 0.412453f + var_G * 0.357580f + var_B * 0.180423f;
	float Y = var_R * 0.212671f + var_G * 0.715160f + var_B * 0.072169f;
	float Z = var_R * 0.019334f + var_G * 0.119193f + var_B * 0.950227f;

	g_odata[indice] = X;
	g_odata[indice+1] = Y;
	g_odata[indice+2] = Z;
}

__device__ void XYZtoLAB(float* g_idata, float* g_odata) {
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int indice = 3*indiceX + 3*indiceY * width;

	float X = g_idata[indice];
	float Y = g_idata[indice + 1];
	float Z = g_idata[indice + 2];

	X = X / 95.047;
	Y = Y / 100.000;
	Z = Z / 108.883;

	if ( X > 0.008856f ) { X = pow(X, 0.33f ); }
	else                { X = ( 7.787 * X ) + ( 16 / 116.0 ); }

	if ( Y > 0.008856f ) { Y = pow(Y, 0.33f ); }
	else                { Y = ( 7.787 * Y ) + ( 16 / 116.0 ); }

	if ( Z > 0.008856f ) { Z = pow(Z, 0.33f ); }
	else                { Z = ( 7.787 * Z ) + ( 16 / 116.0 ); }

	float L = ( 116 * Y ) - 16;
	float a = 500 * ( X - Y );
	float b = 200 * ( Y - Z );

	g_odata[indice] = L;
	g_odata[indice+1] = a;
	g_odata[indice+2] = b;
}

__device__ float getDistance(float* pixel1, float* pixel2) {
	float newVector[3];
	newVector[0] = pixel2[0]-pixel1[0];
	newVector[1] = pixel2[1]-pixel1[1];
	newVector[2] = pixel2[2]-pixel1[2];
	return sqrt(pow(newVector[0], 2)+pow(newVector[1], 2)+pow(newVector[2], 2));
}

__device__ void DetectDiscontinuityFromLab(float* g_idata, unsigned char* g_odata) {
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int indice = 3*indiceX + 3*indiceY * width;

	//seuil of discontinuity
	float seuil = 0.1f; //1.3f

	//current pixel
	float currentPixel[3];
	currentPixel[0] = g_idata[indice];
	currentPixel[1] = g_idata[indice + 1];
	currentPixel[2] = g_idata[indice + 2];

	//pixel to the right
	float rightPixel[3];
	rightPixel[0] = g_idata[indice + 3];
	rightPixel[1] = g_idata[indice + 4];
	rightPixel[2] = g_idata[indice + 5];

	//pixel to the bottom
	float bottomPixel[3];
	bottomPixel[0] = g_idata[indice + width*3];
	bottomPixel[1] = g_idata[indice + width*3 + 1];
	bottomPixel[2] = g_idata[indice + width*3 + 2];

	float distanceToRight = getDistance(currentPixel, rightPixel);
	float distanceToBottom = getDistance(currentPixel, bottomPixel);

	g_odata[indice] = 0;
	g_odata[indice+1] = 0;
	g_odata[indice+2] = 0;

	__syncthreads();

	//bottom => green
	if(distanceToBottom > seuil)// && indice+width*3 < width*height*3
		g_odata[indice+1] = 255;
	//right => red
	if(distanceToRight > seuil)
		g_odata[indice] = 255;
}

__global__ void DetectDiscontinuity(unsigned char* g_idata, float* g_odataXYZ, float* g_odataLab, unsigned char* g_odata) {
	RGBtoXYZ(g_idata, g_odataXYZ);
	__syncthreads();
	XYZtoLAB(g_odataXYZ, g_odataLab);
	__syncthreads();
	DetectDiscontinuityFromLab(g_odataLab, g_odata);
}

__global__ void RecursiveDoublingHorizontal(unsigned char* g_idata, 
	unsigned char* g_odata_hl, unsigned char* g_odata_hr) {
	
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int indice = 3*indiceX + 3*indiceY * width;

	//set shared data
	__shared__ unsigned char bReadyRight[1024];
	__shared__ unsigned char bReadyLeft[1024];
	bReadyRight[indiceX] = 0;
	bReadyLeft[indiceX] = 0;
	bReadyLeft[0]=1;
	bReadyRight[width-1]=1;

	if(indiceX==0){
		g_odata_hl[indice] = 0;
		g_odata_hl[indice+1] = 0;
		g_odata_hl[indice+2] = 0;
	}

	if(indiceX==width-1){
		g_odata_hr[indice] = 0;
		g_odata_hr[indice+1] = 0;
		g_odata_hr[indice+2] = 0;
	}

	__syncthreads();
	
	//horizontal, to the left
	while(bReadyLeft[indiceX]==0){
		if(g_idata[indice+1]>0){ //check green
			if(bReadyLeft[indiceX-1]==1){
				g_odata_hl[indice] = g_odata_hl[indice-3]+1;
				g_odata_hl[indice+1] = g_odata_hl[indice+1-3]+1;
				g_odata_hl[indice+2] = g_odata_hl[indice+2-3]+1;
				bReadyLeft[indiceX] = 1;
			}
		}
		else{
			g_odata_hl[indice] = 0;
			g_odata_hl[indice+1] = 0;
			g_odata_hl[indice+2] = 0;
			bReadyLeft[indiceX] = 1;	
		}
		__syncthreads();
	}

	//horizontal, to the right
	while(bReadyRight[indiceX]==0){
		if(g_idata[indice+1]>0){ //check green
			if(bReadyRight[indiceX+1]==1){
				g_odata_hr[indice] = g_odata_hr[indice+3]+1;
				g_odata_hr[indice+1] = g_odata_hr[indice+1+3]+1;
				g_odata_hr[indice+2] = g_odata_hr[indice+2+3]+1;
				bReadyRight[indiceX] = 1;
			}
		}
		else{
			g_odata_hr[indice] = 0;
			g_odata_hr[indice+1] = 0;
			g_odata_hr[indice+2] = 0;
			bReadyRight[indiceX] = 1;	
		}
		__syncthreads();
	}
}

__global__ void ComputeWeightHorizontal(unsigned char* g_idata_h, float* g_odata_w) {
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int indice = 3*indiceX + 3*indiceY * width;

	if(g_idata_h[indice] == 0) {
		g_odata_w[indiceX + indiceY*width] = 0;
		return;
	}

	int ddroite = 0;
	int dgauche = 0;

	int tmp_indice = indice;
	while(g_idata_h[tmp_indice] !=0) {
		++dgauche;
		--tmp_indice;
	}

	tmp_indice = 0;
	while(g_idata_h[tmp_indice] !=0) {
		++ddroite;
		++tmp_indice;
	}	

	unsigned char p = min(ddroite, dgauche);
	unsigned char T = ddroite + dgauche - 1;

	// Weight
	float w = (1/2.) * (1- ((2*p+1) / T));
	g_odata_w[indiceX + indiceY*width] = w;
}

__global__ void BlendingHorizontal(unsigned char* g_idata, unsigned char* g_idata_h, float* g_idata_w, unsigned char* g_odata) {
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int height = gridDim.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int indice = 3*indiceX + 3*indiceY * width;

	g_odata[indice] = g_idata[indice];
	g_odata[indice+1] = g_idata[indice+1];
	g_odata[indice+2] = g_idata[indice+2];

	__syncthreads();

	// If border pixel
	if(indiceX-1 < 0 || indiceX+1 >= width || indiceY+1 >= height || indiceY-1 < 0)
		return;

	//Detect L shape
	if(g_idata_h[indice] != 0){


		// HORIZONTAL LEFT

		float w_top = g_idata_w[indiceX + indiceY*width - width];
		float w_bottom = g_idata_w[indiceX + indiceY*width + width];
		float w_current = g_idata_w[indiceX + indiceY*width];

		int ddroite_h = 0;
		int dgauche_h = 0;

		int tmp_indice_h = indice;

		while(g_idata_h[tmp_indice_h] !=0) {
			++dgauche_h;
			--tmp_indice_h;
		}

		tmp_indice_h = 0;

		while(g_idata_h[tmp_indice_h] !=0) {
			++ddroite_h;
			++tmp_indice_h;
		}

		//  ___ 			___
		// |		or 		   |
		//
		if(g_idata_h[indice-(3*width)] == 0) {
			//Current pixel
			g_odata[indice] = w_top*g_idata[indice] + g_idata[indice-3*width]*(1-w_top);
			g_odata[indice+1] = w_top*g_idata[indice+1] + g_idata[indice-(3*width)+1]*(1-w_top);
			g_odata[indice+2] = w_top*g_idata[indice+2] + g_idata[indice-(3*width)+2]*(1-w_top);
			//Top pixel
			g_odata[indice-(3*width)] = w_current*g_idata[indice-(3*width)] + g_idata[indice]*(1-w_current);
			g_odata[indice-(3*width)+1] = w_current*g_idata[indice-(3*width)+1] + g_idata[indice+1]*(1-w_current);
			g_odata[indice-(3*width)+2] = w_current*g_idata[indice-(3*width)+2] + g_idata[indice+2]*(1-w_current);

		}

		// 
		// |___ 	or 		___|
		//
		if(g_idata_h[indice+(3*width)] == 0) {
			//Current pixel
			g_odata[indice] = w_bottom*g_idata[indice] + g_idata[indice+3*width]*(1-w_bottom);
			g_odata[indice+1] = w_bottom*g_idata[indice+1] + g_idata[indice+(3*width)+1]*(1-w_bottom);
			g_odata[indice+2] = w_bottom*g_idata[indice+2] + g_idata[indice+(3*width)+2]*(1-w_bottom);
			//Bottom pixel
			g_odata[indice+(3*width)] = w_current*g_idata[indice+(3*width)] + g_idata[indice]*(1-w_current);
			g_odata[indice+(3*width)+1] = w_current*g_idata[indice+(3*width)+1] + g_idata[indice+1]*(1-w_current);
			g_odata[indice+(3*width)+2] = w_current*g_idata[indice+(3*width)+2] + g_idata[indice+2]*(1-w_current);

		}

		//  ___
		//     |		or 		___|
		//
		if(ddroite_h == 0) {
			//Right pixel
			g_odata[indice+3] = w_current*g_idata[indice+3] + g_idata[indice]*(1-w_current);
			g_odata[indice+3+1] = w_current*g_idata[indice+3+1] + g_idata[indice+1]*(1-w_current);
			g_odata[indice+3+2] = w_current*g_idata[indice+3+2] + g_idata[indice+2]*(1-w_current);
		}

		//  ___
		// |			or 		|___
		//
		if(dgauche_h == 0) {
			//Left pixel
			g_odata[indice-3] = w_current*g_idata[indice-3] + g_idata[indice]*(1-w_current);
			g_odata[indice-3+1] = w_current*g_idata[indice-3+1] + g_idata[indice+1]*(1-w_current);
			g_odata[indice-3+2] = w_current*g_idata[indice-3+2] + g_idata[indice+2]*(1-w_current);
		}
	}
}


__global__ void RecursiveDoublingVertical(unsigned char* g_idata, unsigned char* g_odata_vt, unsigned char* g_odata_vb) {
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int height = gridDim.y * blockDim.y;
	int indice = 3*indiceX + 3*indiceY * width;

	//set shared data
	__shared__ unsigned char bReadyTop[768];
	__shared__ unsigned char bReadyBottom[768];
	bReadyTop[indiceY] = 0;
	bReadyBottom[indiceY] = 0;
	bReadyTop[0]=1;
	bReadyBottom[height-1]=1;

	if(indiceY==0){
		g_odata_vt[indice] = 0;
		g_odata_vt[indice+1] = 0;
		g_odata_vt[indice+2] = 0;
	}

	if(indiceY==height-1){
		g_odata_vb[indice] = 0;
		g_odata_vb[indice+1] = 0;
		g_odata_vb[indice+2] = 0;
	}

	__syncthreads();
	
	//vertical, to the top
	while(bReadyTop[indiceY]==0){
		if(g_idata[indice]>0){ //check red
			if(bReadyTop[indiceY-1]==1){
				g_odata_vt[indice] = g_odata_vt[indice-(width*3)]+1;
				g_odata_vt[indice+1] = g_odata_vt[indice+1-(width*3)]+1;
				g_odata_vt[indice+2] = g_odata_vt[indice+2-(width*3)]+1;
				bReadyTop[indiceY] = 1;
			}
		}
		else{
			g_odata_vt[indice] = 0;
			g_odata_vt[indice+1] = 0;
			g_odata_vt[indice+2] = 0;
			bReadyTop[indiceY] = 1;	
		}
		__syncthreads();
	}

	//vertical, to the bottom
	while(bReadyBottom[indiceY]==0){
		if(g_idata[indice]>0){ //check red
			if(bReadyBottom[indiceY+1]==1){
				g_odata_vb[indice] = g_odata_vb[indice+(width*3)]+1;
				g_odata_vb[indice+1] = g_odata_vb[indice+1+(width*3)]+1;
				g_odata_vb[indice+2] = g_odata_vb[indice+2+(width*3)]+1;
				bReadyBottom[indiceY] = 1;
			}
		}
		else{
			g_odata_vb[indice] = 0;
			g_odata_vb[indice+1] = 0;
			g_odata_vb[indice+2] = 0;
			bReadyBottom[indiceY] = 1;	
		}
		__syncthreads();
	}
}

__global__ void ComputeWeightVertical(unsigned char* g_idata_v, float* g_odata_w) {
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int indice = 3*indiceX + 3*indiceY * width;

	if(g_idata_v[indice] == 0) {
		g_odata_w[indiceX + indiceY*width] = 0;
		return;
	}

	int dhaut = 0;
	int dbas = 0;

	int tmp_indice = indice;
	while(g_idata_v[tmp_indice] !=0) {
		++dhaut;
		tmp_indice -= width;
	}

	tmp_indice = 0;
	while(g_idata_v[tmp_indice] !=0) {
		++dbas;
		tmp_indice += width;
	}	

	unsigned char p = min(dbas, dhaut);
	unsigned char T = dbas + dhaut - 1;

	// Weight
	float w = (1/2.) * (1- ((2*p+1) / T));
	g_odata_w[indiceX + indiceY*width] = w;
}

__global__ void BlendingVertical(unsigned char* g_idata, unsigned char* g_idata_v, float* g_idata_w, unsigned char* g_odata) {
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int height = gridDim.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int indice = 3*indiceX + 3*indiceY * width;

	g_odata[indice] = g_idata[indice];
	g_odata[indice+1] = g_idata[indice+1];
	g_odata[indice+2] = g_idata[indice+2];

	__syncthreads();

	// If border pixel
	if(indiceX-1 < 0 || indiceX+1 >= width || indiceY+1 >= height || indiceY-1 < 0)
		return;

	//Detect L shape
	if(g_idata_v[indice] > 0){

		// VERTICAL TOP

		float w_left = g_idata_w[indiceX + indiceY*width-1];
		float w_right = g_idata_w[indiceX + indiceY*width+1];
		float w_current = g_idata_w[indiceX + indiceY*width];

		int dbas_v = 0;
		int dhaut_v = 0;

		int tmp_indice_v = indice;

		while(g_idata_v[tmp_indice_v] !=0) {
			++dbas_v;
			tmp_indice_v -= width;
		}

		tmp_indice_v = 0;

		while(g_idata_v[tmp_indice_v] !=0) {
			++dhaut_v;
			tmp_indice_v += width;
		}

		//  _
		// |		or 		|
		// |				|_
		if(g_idata_v[indice-3] == 0) {
			//Current pixel
			g_odata[indice] = w_left*g_idata[indice] + g_idata[indice-3]*(1-w_left);
			g_odata[indice+1] = w_left*g_idata[indice+1] + g_idata[indice-3+1]*(1-w_left);
			g_odata[indice+2] = w_left*g_idata[indice+2] + g_idata[indice-3+2]*(1-w_left);
			//Left pixel
			g_odata[indice-3] = w_current*g_idata[indice-3] + g_idata[indice]*(1-w_current);
			g_odata[indice-3+1] = w_current*g_idata[indice-3+1] + g_idata[indice+1]*(1-w_current);
			g_odata[indice-3+2] = w_current*g_idata[indice-3+2] + g_idata[indice+2]*(1-w_current);
		}

		//  _
		//   |		or 		 |
		//   |				_|
		if(g_idata_v[indice+3] == 0) {
			//Current pixel
			g_odata[indice] = w_right*g_idata[indice] + g_idata[indice+3]*(1-w_right);
			g_odata[indice+1] = w_right*g_idata[indice+1] + g_idata[indice+3+1]*(1-w_right);
			g_odata[indice+2] = w_right*g_idata[indice+2] + g_idata[indice+3+2]*(1-w_right);
			//Right pixel
			g_odata[indice+3] = w_current*g_idata[indice+3] + g_idata[indice]*(1-w_current);
			g_odata[indice+3+1] = w_current*g_idata[indice+3+1] + g_idata[indice+1]*(1-w_current);
			g_odata[indice+3+2] = w_current*g_idata[indice+3+2] + g_idata[indice+2]*(1-w_current);

		}

		//
		// |		or 		 |
		// |_				_|
		if(dbas_v == 0) {
			//Bottom pixel
			g_odata[indice+(3*width)] = w_current*g_idata[indice+(3*width)] + g_idata[indice]*(1-w_current);
			g_odata[indice+(3*width)+1] = w_current*g_idata[indice+(3*width)+1] + g_idata[indice+1]*(1-w_current);
			g_odata[indice+(3*width)+2] = w_current*g_idata[indice+(3*width)+2] + g_idata[indice+2]*(1-w_current);
		}

		//  _              _
		// |		or 		|
		// |				|
		if(dhaut_v == 0) {
			//Top pixel
			g_odata[indice-(3*width)] = w_current*g_idata[indice-(3*width)] + g_idata[indice]*(1-w_current);
			g_odata[indice-(3*width)+1] = w_current*g_idata[indice-(3*width)+1] + g_idata[indice+1]*(1-w_current);
			g_odata[indice-(3*width)+2] = w_current*g_idata[indice-(3*width)+2] + g_idata[indice+2]*(1-w_current);
		}
	}
}



__global__ void Blending(unsigned char* g_idata, unsigned char* g_idata_hl, unsigned char* g_idata_vt, float* g_idata_wl, float* g_idata_wt, unsigned char* g_odata) {
	int indiceX = threadIdx.x + blockIdx.x * blockDim.x;
	int indiceY = threadIdx.y + blockIdx.y * blockDim.y;
	int height = gridDim.y * blockDim.y;
	int width = gridDim.x * blockDim.x;
	int indice = 3*indiceX + 3*indiceY * width;

	g_odata[indice] = g_idata[indice];
	g_odata[indice+1] = g_idata[indice+1];
	g_odata[indice+2] = g_idata[indice+2];

	__syncthreads();

	// If border pixel
	if(indiceX-1 < 0 || indiceX+1 >= width || indiceY+1 >= height || indiceY-1 < 0)
		return;

	if(g_idata_vt[indice] != 0){
		g_odata[indice] = (g_idata[indice-3*width]*g_idata_wt[indiceX + indiceY*width - width] 
						+ g_idata[indice-3]*g_idata_wt[indiceX + indiceY*width - 1]
						+ g_idata[indice+3]*g_idata_wt[indiceX + indiceY*width + 1]
						+ g_idata[indice+3*width]*g_idata_wt[indiceX + indiceY*width + width]
						+ g_idata[indice]*g_idata_wt[indiceX + indiceY*width])/5;

		g_odata[indice+1] = (g_idata[indice-3*width+1]*g_idata_wt[indiceX + indiceY*width - width] 
						+ g_idata[indice-3+1]*g_idata_wt[indiceX + indiceY*width - 1]
						+ g_idata[indice+3+1]*g_idata_wt[indiceX + indiceY*width + 1]
						+ g_idata[indice+3*width+1]*g_idata_wt[indiceX + indiceY*width + width]
						+ g_idata[indice+1]*g_idata_wt[indiceX + indiceY*width])/5;

		g_odata[indice+2] = (g_idata[indice-3*width+2]*g_idata_wt[indiceX + indiceY*width - width] 
						+ g_idata[indice-3+2]*g_idata_wt[indiceX + indiceY*width - 1]
						+ g_idata[indice+3+2]*g_idata_wt[indiceX + indiceY*width + 1]
						+ g_idata[indice+3*width+2]*g_idata_wt[indiceX + indiceY*width + width]
						+ g_idata[indice+2]*g_idata_wt[indiceX + indiceY*width])/5;
	}

	__syncthreads();

	if(g_idata_hl[indice] != 0){
		g_odata[indice] = (g_idata[indice-3*width]*g_idata_wl[indiceX + indiceY*width - width] 
						+ g_idata[indice-3]*g_idata_wl[indiceX + indiceY*width - 1]
						+ g_idata[indice+3]*g_idata_wl[indiceX + indiceY*width + 1]
						+ g_idata[indice+3*width]*g_idata_wl[indiceX + indiceY*width + width]
						+ g_idata[indice]*g_idata_wl[indiceX + indiceY*width])/5;

		g_odata[indice+1] = (g_idata[indice-3*width+1]*g_idata_wl[indiceX + indiceY*width - width] 
						+ g_idata[indice-3+1]*g_idata_wl[indiceX + indiceY*width - 1]
						+ g_idata[indice+3+1]*g_idata_wl[indiceX + indiceY*width + 1]
						+ g_idata[indice+3*width+1]*g_idata_wl[indiceX + indiceY*width + width]
						+ g_idata[indice+1]*g_idata_wl[indiceX + indiceY*width])/5;

		g_odata[indice+2] = (g_idata[indice-3*width+2]*g_idata_wl[indiceX + indiceY*width - width] 
						+ g_idata[indice-3+2]*g_idata_wl[indiceX + indiceY*width - 1]
						+ g_idata[indice+3+2]*g_idata_wl[indiceX + indiceY*width + 1]
						+ g_idata[indice+3*width+2]*g_idata_wl[indiceX + indiceY*width + width]
						+ g_idata[indice+2]*g_idata_wl[indiceX + indiceY*width])/5;
	}
}

#endif // #ifndef _KERNEL_H_
