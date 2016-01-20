#define _USE_MATH_DEFINES
#include <cmath>
#include "engine.h"

int lastTime, frameCount = 0;

int imageWidth = 1000, imageHeight = 1000;
float theta = 0;
uchar4 *image;
GLuint gl_Tex, gl_Shader;

int main(int argc, char **argv) {
	// GL
	printf("GLUT Initialization... ");
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageWidth, imageHeight);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);
	glutDisplayFunc(displayFunc);
	printf("Done!\n");

	// Triangle Setup
	printf("Triangle Setup... ");
	int subdivisions = 0;
	float3 *vertices = makeSphere(make_float3(0.0, 0.0, 0.0), 1.0, subdivisions);
	int numVertices = 3 * pow(4, subdivisions + 1);

	int3 *indices = new int3[numVertices / 3];
	for (int i = 0; i < numVertices / 3; i++)
		indices[i] = make_int3(3 * i, 3 * i + 1, 3 * i + 2);
	int numIndices = numVertices / 3;
	printf("Done!\n");

	// Matrices
	float3 eye = make_float3(1, 0, 1),
		at = make_float3(0, 0, 0),
		up = make_float3(0, 1, 0),
		*modelMat = lookAt(eye, at, up),
		*projMat = det3();

	// Texture
	initOpenGLBuffers();

	// Draw
	printf("Flex Setup... ");
	image = new uchar4[imageWidth * imageHeight];
	flex::init(true);
	flex::bufferImage(image, imageWidth, imageHeight);
	flex::bufferVertices(vertices, numVertices);
	flex::bufferIndices(indices, numIndices);
	flex::bufferModelViewMatrix(modelMat);
	flex::bufferProjectionMatrix(projMat);
	printf("Done!\n");

	// Render Loop
	printf("Starting Render Loop\n");
	lastTime = glutGet(GLUT_ELAPSED_TIME);
	glutMainLoop();
}

void displayFunc() {
	theta = fmod(theta + 0.01, 2 * M_PI);
	float3 eye = make_float3(sin(theta), 0, cos(theta)),
		at = make_float3(0, 0, 0),
		up = make_float3(0, 1, 0),
		*modelMat = lookAt(eye, at, up);
	flex::bufferModelViewMatrix(modelMat);

	flex::render();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);

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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
	printf("Done!\n");
}