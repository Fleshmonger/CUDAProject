#include <vector> 
#include <iostream> 
#include <thrust/transform.h> 
#include <thrust/functional.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 

using namespace std;
using namespace thrust;

#define THREADS_PER_BLOCK 128
#define SQRT_TPB 8

struct fragment {
	bool *pixels;
	int i_x, i_y, i_width, i_height, index;

	__device__  fragment() {
		this->pixels = nullptr;
		this->i_x = 0;
		this->i_y = 0;
		this->i_width = 0;
		this->i_height = 0;
		this->index = 0;
	}

	__device__ fragment(bool *pixels, int i_x, int i_y, int i_width, int i_height, int index) {
		this->pixels = pixels;
		this->i_x = i_x;
		this->i_y = i_y;
		this->i_width = i_width;
		this->i_height = i_height;
		this->index = index;
	}
};

// Runs the vertex shader on a vertex.
__global__ void vertexShader(int *width, int *height, float3 vertices[], int *numVertices) {
	// Retrieve Vertex.
	int index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if (index >= *numVertices)
		return;
	float3 vertex = vertices[index];

	// Image Coordinates
	vertices[index] = make_float3((vertex.x / 2.0 + 0.5) * (*width), (vertex.y / 2.0 + 0.5) * (*height), vertex.z);
}


// Rasterizes a triangle.
__global__ void rasterizeTriangle(int *width, int *height, float3 vertices[], int3 indices[], fragment fragments[]) {
	// Retrieve Vertices
	int3 index = indices[blockIdx.x];
	float3 v1 = vertices[index.x],
		v2 = vertices[index.y],
		v3 = vertices[index.z];

	// Triangle Bounding Box
	float t_left = fmin(v1.x, fmin(v2.x, v3.x)),
		t_right = fmax(v1.x, fmax(v2.x, v3.x)),
		t_bottom = fmin(v1.y, fmin(v2.y, v3.y)),
		t_top = fmax(v1.y, fmax(v2.y, v3.y));

	// Fragment Dimensions
	int f_width = ceil((t_right - t_left) / SQRT_TPB),
		f_height = ceil((t_top - t_bottom) / SQRT_TPB),
		f_x = t_left + (threadIdx.x % SQRT_TPB) * f_width,
		f_y = t_bottom + (threadIdx.x / SQRT_TPB) * f_height;

	// Barycentric Init
	float denom = (v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y);

	// Init Pixels
	bool *f_pixels = (bool*)malloc(sizeof(bool) * f_width * f_height);
	bool f_empty = true;

	// Rasterize
	for (int x = 0; x < f_width; x++) {
		for (int y = 0; y < f_height; y++) {
			float i_x = f_x + x + 0.5, i_y = f_y + y + 0.5,
				alpha = ((v2.y - v3.y) * (i_x - v3.x) + (v3.x - v2.x) * (i_y - v3.y)) / denom,
				beta = ((v3.y - v1.y) * (i_x - v3.x) + (v1.x - v3.x) * (i_y - v3.y)) / denom,
				gamma = 1.0f - alpha - beta;
			if (0.0 <= alpha && 0.0 <= beta && 0.0 <= gamma) {
				f_pixels[x + y * f_width] = true;
				f_empty = false;
			}
			else
				f_pixels[x + y * f_width] = false;
		}
	}
	if (f_empty)
		free(f_pixels);
	else
		fragments[threadIdx.x + blockIdx.x * THREADS_PER_BLOCK] = fragment(f_pixels, f_x, f_y, f_width, f_height, blockIdx.x);
}

// Runs the fragment shader on a fragment.
__global__ void fragmentShader(uchar4 *d_pixels, int *width, float3 vertices[], int3 indices[], fragment fragments[]) {
	// Retrieve Fragment
	fragment frag = fragments[threadIdx.x + blockIdx.x * THREADS_PER_BLOCK];
	if (frag.pixels == nullptr)
		return;

	// Copy Fragment Pixels
	for (int x = 0; x < frag.i_width; x++) {
		for (int y = 0; y < frag.i_height; y++) {
			if (frag.pixels[x + y * frag.i_width])
				d_pixels[frag.i_x + x + (frag.i_y + y) * (*width)] = make_uchar4(255, 0, 0, 255);
		}
	}

	// Deallocate Fragment Pixels
	free(frag.pixels);
}

void pipeline(int numVertices, int numTriangles, uchar4 *d_pixels, int *d_width, int *d_height, float3 *d_vertices, int3 *d_indices, int *d_numVertices, fragment *d_fragments) {
	// Vertex Shader
	printf("Vertex shader...\n");
	vertexShader<<<ceil((float) numVertices / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_width, d_height, d_vertices, d_numVertices);
	cudaDeviceSynchronize();
	printf("Vertex shader complete.\n");

	// Rasterize
	printf("Rasterization...\n");
	rasterizeTriangle<<<numTriangles, THREADS_PER_BLOCK>>>(d_width, d_height, d_vertices, d_indices, d_fragments);
	cudaDeviceSynchronize();
	printf("Rasterization complete.\n");

	// Fragment Shader
	printf("Fragment shader...\n");
	fragmentShader<<<numTriangles, THREADS_PER_BLOCK>>>(d_pixels, d_width, d_vertices, d_indices, d_fragments);
	cudaDeviceSynchronize();
	printf("Fragment shader complete.\n");

	// Geometry Shader
}

void draw(uchar4 *pixels, int width, int height, float3 *vertices, int3 *indices, int numVertices, int numTriangles) {
	uchar4 *d_pixels;
	int *d_width, *d_height, *d_numVertices;
	float3 *d_vertices;
	int3 *d_indices;
	fragment *d_fragments;

	cudaMalloc((void **)&d_pixels, sizeof(uchar4) * width * height);
	cudaMalloc((void **)&d_width, sizeof(int));
	cudaMalloc((void **)&d_height, sizeof(int));
	cudaMalloc((void **)&d_numVertices, sizeof(int));
	cudaMalloc((void **)&d_vertices, sizeof(float3) * numVertices);
	cudaMalloc((void **)&d_indices, sizeof(int3) * numTriangles);
	cudaMalloc((void **)&d_fragments, sizeof(fragment) * numTriangles * THREADS_PER_BLOCK);

	cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_numVertices, &numVertices, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertices, vertices, sizeof(float3) * numVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, indices, sizeof(int3) * numTriangles, cudaMemcpyHostToDevice);

	pipeline(numVertices, numTriangles, d_pixels, d_width, d_height, d_vertices, d_indices, d_numVertices, d_fragments);

	cudaMemcpy(pixels, d_pixels, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

	cudaFree(d_pixels);
	cudaFree(d_width);
	cudaFree(d_height);
	cudaFree(d_vertices);
	cudaFree(d_indices);
	cudaFree(d_fragments);
}