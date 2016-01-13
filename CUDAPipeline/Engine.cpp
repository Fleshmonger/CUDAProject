// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

// CUDA runtime,

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <rendercheck_gl.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>

#include <thrust/transform.h> 
#include <thrust/functional.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h>

uchar4* rasterize(int w, int h);

int imageW = 1000, imageH = 1000;
uchar4 *h_Src = (uchar4 *)malloc(imageW * imageH * 4);
GLuint gl_Tex, gl_Shader;

void displayFunc(void);
void initOpenGLBuffers(int w, int h);

int main(int argc, char **argv) {
	printf("Initializing GLUT...\n");
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);
	h_Src = rasterize(imageW, imageH);
	/*
	for (int y = 0; y < imageH; y++)
	{
		for (int x = 0; x < imageW; x++)
		{
			h_Src[x + y * imageW].x = (int) ((1.0f * y / imageH) * 255);
			h_Src[x + y * imageW].y = (int)((1.0f * x / imageW) * 255);
			h_Src[x + y * imageW].z = 0;
			h_Src[x + y * imageW].w = 255;
		}
	}
	*/
	initOpenGLBuffers(imageW, imageH);
	displayFunc();
	glutMainLoop();
}

// OpenGL display function
void displayFunc(void) {
	// render the Mandelbrot image
	//renderImage(true, g_isJuliaSet, precisionMode);

	// load texture from PBO
	//  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
	//  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// fragment program is required to display floating point texture
	/*
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glDisable(GL_DEPTH_TEST);
	*/

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

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);

	//sdkStopTimer(&hTimer);
	glutSwapBuffers();
}

void initOpenGLBuffers(int w, int h) {
	// allocate new buffers
	printf("Creating GL texture...\n");
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
	printf("Texture created.\n");

	/*
	printf("Creating PBO...\n");
	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
	//While a PBO is registered to CUDA, it can't be used
	//as the destination for OpenGL drawing calls.
	//But in our particular case OpenGL is only used
	//to display the content of the PBO, specified by CUDA kernels,
	//so we need to register/unregister it only once.

	// DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
		cudaGraphicsMapFlagsWriteDiscard));
	printf("PBO created.\n");
	*/
}