#include "engine.h"


int lastTime, frameCount = 0;

int imageW = 1000, imageH = 1000;
uchar4 *pixels;
GLuint gl_Tex, gl_Shader;

void draw(uchar4 *pixels, int width, int height, float3 *vertices, int3 *indices, int numVertices, int numTriangles);

void displayFunc();
void initOpenGLBuffers();

int main(int argc, char **argv) {
	// GL
	printf("GLUT Initialization... ");
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);
	glutDisplayFunc(displayFunc);
	printf("Done!\n");

	// Triangle Setup
	printf("Triangle Setup... ");
	int subdivisions = 5;
	float3 *vertices = makeSphere(make_float3(0.0, 0.0, 0.0), 1.0, subdivisions);
	int numVertices = 3 * pow(4, subdivisions + 1);

	int3 *indices = new int3[numVertices / 3];
	for (int i = 0; i < numVertices / 3; i++)
		indices[i] = make_int3(3 * i, 3 * i + 1, 3 * i + 2);
	int numIndices = numVertices / 3;
	printf("Done!\n");

	// Draw
	pixels = (uchar4 *)malloc(imageW * imageH * 4);
	draw(pixels, imageW, imageH, vertices, indices, numVertices, numIndices);

	// Render Loop
	lastTime = glutGet(GLUT_ELAPSED_TIME);
	initOpenGLBuffers();
	glutMainLoop();
}

void displayFunc() {
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glEnd();

	//glDisable(GL_FRAGMENT_PROGRAM_ARB);

	glutSwapBuffers();

	frameCount++;
	if (glutGet(GLUT_ELAPSED_TIME) - lastTime >= 1000) {
		char fps[256];
		sprintf(fps, "fps: %i", frameCount);
		glutSetWindowTitle(fps);
		frameCount = 0;
		lastTime = glutGet(GLUT_ELAPSED_TIME);
	}

	glutPostRedisplay();
}

void initOpenGLBuffers() {
	// Texture
	printf("Texture Setup... ");
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	printf("Done!\n");
}