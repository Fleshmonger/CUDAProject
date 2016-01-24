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

#include "common.h"

// CUDA
namespace flex {
	void init(bool cull, float3 light);
	void bufferImage(uchar4 image[], int windowWidth, int windowHeight);
	void bufferVertices(float4 vertices[], float4 normals[], int length);
	void bufferIndices(int3 indices[], int length);
	void bufferProjectionMatrix(float4 matrix[]);
	void bufferModelViewMatrix(float4 matrix[], float4 normalMatrix[]);
	void render();
}

int main(int argc, char **argv);
void displayFunc();
void initOpenGLBuffers();