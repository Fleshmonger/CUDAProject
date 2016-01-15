#include "engine.h"

void draw(uchar4 *pixels, int width, int height, float3 *vertices, int3 *indices, int numVertices, int numTriangles);

int imageW = 1000, imageH = 1000;
uchar4 *pixels;
GLuint gl_Tex, gl_Shader;

void displayFunc(void);
void initOpenGLBuffers(int w, int h);

int main(int argc, char **argv) {
	// GL
	printf("Initializing GLUT...\n");
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);

	// Triangle Setup

	// Sphere
	int subdivisions = 4;
	float3 *vertices = makeSphere(make_float3(0.0, 0.0, 0.0), 1.0, subdivisions);
	int numVertices = 3 * pow(4, subdivisions + 1);

	int3 *indices = new int3[numVertices / 3];
	for (int i = 0; i < numVertices / 3; i++)
		indices[i] = make_int3(3 * i, 3 * i + 1, 3 * i + 2);
	int numIndices = numVertices / 3;

	printf("numVertices = %d, numIndices = %d.\n", numVertices, numIndices);
	for (int i = 0; i < numVertices; i++)
		printf("Vertex %d = (%f, %f, %f).\n", i, vertices[i].x, vertices[i].y, vertices[i].z);

	// Tetrahedron
	/*
	float3 vertices[] = {
		make_float3(0.0, -1.0, 0.0),
		make_float3(0.942809, 0.333333, 0.0),
		make_float3(-0.471405, 0.333333, -0.816497),
		make_float3(-0.471405, 0.333333, 0.816497)
	};

	int3 indices[] = {
		make_int3(0, 1, 2),
		make_int3(3, 2, 1),
		make_int3(0, 3, 1),
		make_int3(0, 2, 3)
	};
	*/

	/*
	// Two triangles
	float3 vertices[] = {
		make_float3(0.0, 0.0, 0.0),
		make_float3(1.0, 0.0, 0.0),
		make_float3(1.0, 1.0, 0.0),
		make_float3(-1.0, 1.0, 0.0),
		make_float3(-1.0, -1.0, 0.0),
		make_float3(1.0, -1.0, 0.0)
	};

	int3 indices[] = {
		make_int3(0, 1, 2),
		make_int3(3, 4, 5)
	};


	int numVertices = sizeof(vertices) / sizeof(float3),
		numIndices = sizeof(indices) / sizeof(int3);
		*/
	// Draw
	pixels = (uchar4 *)malloc(imageW * imageH * 4);
	draw(pixels, imageW, imageH, vertices, indices, numVertices, numIndices);

	// Render
	initOpenGLBuffers(imageW, imageH);
	displayFunc();
	glutMainLoop();
}

// OpenGL display function
void displayFunc() {
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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
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