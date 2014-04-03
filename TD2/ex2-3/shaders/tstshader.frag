// A TOO SIMPLE FRAGMENT SHADER...
uniform sampler2D usampler;
uniform vec3 sepiaColor;
uniform float width;
uniform float height;

void makeColorSwitchGPU()
{
	vec4 v4_baseTexture = texture2D(usampler, gl_TexCoord[0].xy);
	v4_baseTexture.rgb = v4_baseTexture.bgr;
	gl_FragColor = v4_baseTexture;
}

void makeLuminanceGPU()
{
	// can do : mat3 * vec3(r, g, b) = vec3(lum, lum, lum) !
	vec4 v4_baseTexture = texture2D(usampler, gl_TexCoord[0].xy);
	float luminance = 0.3 * v4_baseTexture.r + 0.59 * v4_baseTexture.g + 0.11 * v4_baseTexture.b;
	v4_baseTexture.rgb = vec3(luminance, luminance, luminance);
	gl_FragColor = v4_baseTexture;
}

void makeSepiaGPU()
{
	vec4 v4_baseTexture = texture2D(usampler, gl_TexCoord[0].xy);
	float luminance = 0.3 * v4_baseTexture.r + 0.59 * v4_baseTexture.g + 0.11 * v4_baseTexture.b;
	v4_baseTexture.rgb = vec3((1. + 0.2)*luminance, (1. + 0.1)*luminance, (1. + 0.)*luminance);
	gl_FragColor = v4_baseTexture;
}

void makeBlurGPU()
{
	vec4 avgValue = vec4(0., 0., 0., 0.);

	for (float i = -1./width; i <= 1./width; i+=1./width) {
		for (float j = -1./height; j <= 1./height; j+=1./height) {
			avgValue += texture2D(usampler, gl_TexCoord[0].xy + vec2(i, j));
		}
	}
	avgValue /= 9.;

	gl_FragColor = avgValue;
}

void Sobel()
{
	mat3 mat3_luminanceTexture;
	int line = 0;
	int column = 0;
	for (float i = -1./width; i <= 1./width; i+=1./width) {
		for (float j = -1./height; j <= 1./height; j+=1./height) {
			vec4 v4_baseTexture = texture2D(usampler, gl_TexCoord[0].xy + vec2(i, j));
			float luminance = 0.3 * v4_baseTexture.r + 0.59 * v4_baseTexture.g + 0.11 * v4_baseTexture.b;
			mat3_luminanceTexture[line][column] = luminance;
			column += 1;
		}
		line += 1;
		column = 0;
	}

	// compute masks (horizontal and vertical)
	mat3 mat3_gradient_x = mat3(vec3(-1, -2, -1), 
				   vec3(0, 0, 0),
				   vec3(1, 2, 1));
	mat3 mat3_gradient_y = mat3(vec3(1, 0, -1),
				   vec3(2, 0, -2),
				   vec3(1, 0, -1));
	
	float gx = (mat3_gradient_x * mat3_luminanceTexture)[1][1];
	float gy = (mat3_gradient_y * mat3_luminanceTexture)[1][1];

	// norme of the gradient
	float g = sqrt(gx*gx + gy*gy);
	if(g > 1.){g = 1.;}
	
	gl_FragColor.rgb = vec3(g, g, g);
}

void main()
{
	// 0 - initialisation (rouge !)
	gl_FragColor.rgb = vec3(1.0,0.0,0.0);
	
	// 1 - Changement des couleurs ...
	//makeColorSwitchGPU();
		
	// 2.a - Traitement d'image local - luminance
	//makeLuminanceGPU();
	
	// 2.b - Sepia customise
	//makeSepiaGPU();
	
	// 3.a - Blur 3x3
	//makeBlurGPU();
	
	// 3.b - masque/filtre de Sobel
	Sobel();

}
