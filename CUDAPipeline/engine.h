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

#include "meshes.h"

// CUDA
namespace flex {
	void init(bool cull);
	void bindImage(uchar4 image[], int windowWidth, int windowHeight);
	void bindVertices(float3 vertices[], int length);
	void bindIndices(int3 indices[], int length);
	void render();
}

int main(int argc, char **argv);
void displayFunc();
void initOpenGLBuffers();