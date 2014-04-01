// A VERY SIMPLE VERTEX SHADER...

void main()
{
	// Transmission des coordonnees de texture
	gl_TexCoord[0] = gl_MultiTexCoord0;
	// Projection du point dans le repere camera
	gl_Position = ftransform();
}
