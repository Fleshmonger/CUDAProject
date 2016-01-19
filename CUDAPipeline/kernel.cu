#include <vector> 
#include <iostream> 
#include <thrust/transform.h> 
#include <thrust/functional.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 

using namespace std;
using namespace thrust;

#define THREADS_PER_BLOCK 256
#define SQRT_TPB 16

struct fragment {
	int i_x, i_y, i_width, i_height, index;

	__device__  fragment() {
		this->i_x = 0;
		this->i_y = 0;
		this->i_width = 0;
		this->i_height = 0;
		this->index = 0;
	}

	__device__ fragment(int i_x, int i_y, int i_width, int i_height, int index) {
		this->i_x = i_x;
		this->i_y = i_y;
		this->i_width = i_width;
		this->i_height = i_height;
		this->index = index;
	}
};

int width, height, numVertices, numTriangles, *d_width, *d_height, *d_numVertices;
int3 *d_indices;
float3 *d_vertices;
uchar4 *pixels, *d_pixels;

// Runs the vertex shader on a vertex.
__global__ void vertexShader(int *width, int *height, float3 vertices[], int *numVertices) {
	// Retrieve Vertex.
	int index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if (index >= *numVertices)
		return;
	//float3 vertex = vertices[index];
}

__device__ float d_min(float a, float b) {
	if (a < b)
		return a;
	else
		return b;
}

__device__ float d_min(float a, float b, float c) {
	return d_min(a, d_min(b, c));
}
__device__ float d_max(float a, float b) {
	if (a > b)
		return a;
	else
		return b;
}

__device__ float d_max(float a, float b, float c) {
	return d_max(a, d_max(b, c));
}

__device__ float d_ceil(float a) {
	return a - (int)a > 0.0 ? (int)a + 1.0 : a;
}

__device__ float3 d_cross(float3 u, float3 v) {
	return make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

__device__ float3 d_subtract(float3 u, float3 v) {
	return make_float3(u.x - v.x, u.y - v.y, u.z - v.z);
}

/*
__device__ float d_dot(float3 u, float3 v) {
	return u.x * v.x + u.y * v.y + u.z * v.z;
}
*/

// Rasterizes a triangle.
__global__ void rasterizeTriangle(int *width, int *height, float3 vertices[], int3 indices[], fragment fragments[], float3 interpolation[], int triangles[]) {
	// Retrieve Vertices
	int3 index = indices[blockIdx.x];
	float3 v1 = vertices[index.x],
		v2 = vertices[index.y],
		v3 = vertices[index.z];

	// Face Culling
	if (d_cross(d_subtract(v2, v1), d_subtract(v3, v1)).z < 0.0)
		return;

	// Map Image Coords
	float2 i_v1 = make_float2((v1.x / 2.0 + 0.5) * (*width), (v1.y / 2.0 + 0.5) * (*height)),
		i_v2 = make_float2((v2.x / 2.0 + 0.5) * (*width), (v2.y / 2.0 + 0.5) * (*height)),
		i_v3 = make_float2((v3.x / 2.0 + 0.5) * (*width), (v3.y / 2.0 + 0.5) * (*height));

	// Triangle Bounding Box
	int t_left = d_max(0.0, d_min(i_v1.x, i_v2.x, i_v3.x)),
		t_bottom = d_max(0.0, d_min(i_v1.y, i_v2.y, i_v3.y)),
		t_right = d_min(*width - 1, d_ceil(d_max(i_v1.x, i_v2.x, i_v3.x))),
		t_top = d_min(*height - 1, d_ceil(d_max(i_v1.y, i_v2.y, i_v3.y)));

	// Fragment Dimensions
	int f_width = d_ceil(((float) (t_right - t_left)) / SQRT_TPB),
		f_height = d_ceil(((float) (t_top - t_bottom)) / SQRT_TPB),
		f_x = t_left + (threadIdx.x % SQRT_TPB) * f_width,
		f_y = t_bottom + (threadIdx.x / SQRT_TPB) * f_height;

	// Barycentric Init
	float denom = (i_v2.y - i_v3.y) * (i_v1.x - i_v3.x) + (i_v3.x - i_v2.x) * (i_v1.y - i_v3.y);

	// Rasterize
	for (int x = f_x; x < d_min(f_x + f_width, *width); x++) {
		for (int y = f_y; y < d_min(f_y + f_height, *height); y++) {
			float i_x = x + 0.5, i_y = y + 0.5,
				alpha = ((i_v2.y - i_v3.y) * (i_x - i_v3.x) + (i_v3.x - i_v2.x) * (i_y - i_v3.y)) / denom,
				beta = ((i_v3.y - i_v1.y) * (i_x - i_v3.x) + (i_v1.x - i_v3.x) * (i_y - i_v3.y)) / denom,
				gamma = 1.0f - alpha - beta;
			if (0.0 <= alpha && 0.0 <= beta && 0.0 <= gamma) {
				interpolation[x + y * (*width)] = make_float3(alpha, beta, gamma);
				triangles[x + y * (*width)] = blockIdx.x;
			}
		}
	}
	fragments[threadIdx.x + blockIdx.x * THREADS_PER_BLOCK] = fragment(f_x, f_y, f_width, f_height, blockIdx.x);
}

// Runs the fragment shader on a fragment.
__global__ void fragmentShader(uchar4 *d_pixels, int *width, int *height, float3 vertices[], int3 indices[], fragment fragments[], float3 interpolation[], int triangles[]) {
	// Retrieve Fragment
	fragment frag = fragments[threadIdx.x + blockIdx.x * THREADS_PER_BLOCK];

	// Copy Fragment Pixels
	for (int x = frag.i_x; x < d_min(frag.i_x + frag.i_width, *width); x++) {
		for (int y = frag.i_y; y < d_min(frag.i_y + frag.i_height, *height); y++) {
			int index = x + y * (*width);
			if (triangles[index] == frag.index) {
				int3 vIndex = indices[triangles[index]];
				float3 v1 = vertices[vIndex.x],
					v2 = vertices[vIndex.y],
					v3 = vertices[vIndex.z];
				float red = ((v1.x + 1.0) / 2) * interpolation[index].x + ((v2.x + 1.0) / 2) * interpolation[index].y + ((v3.x + 1.0) / 2) * interpolation[index].z,
					green = ((v1.y + 1.0) / 2) * interpolation[index].x + ((v2.y + 1.0) / 2) * interpolation[index].y + ((v3.y + 1.0) / 2) * interpolation[index].z,
					blue = ((v1.z + 1.0) / 2) * interpolation[index].x + ((v2.z + 1.0) / 2) * interpolation[index].y + ((v3.z + 1.0) / 2) * interpolation[index].z;
				d_pixels[index] = make_uchar4(red * 255, green * 255, blue * 255, 255);
			}
		}
	}
}

void pipeline(fragment *d_fragments, float3 *d_interpolation, int *d_triangles) {
	// Vertex Shader
	printf("Vertex shader...\n");
	vertexShader<<<ceil(((float) numVertices) / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_width, d_height, d_vertices, d_numVertices);
	cudaDeviceSynchronize();
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	printf("Vertex shader complete.\n");

	// Rasterize
	printf("Rasterization...\n");
	rasterizeTriangle<<<numTriangles, THREADS_PER_BLOCK>>>(d_width, d_height, d_vertices, d_indices, d_fragments, d_interpolation, d_triangles);
	//cudaDeviceSynchronize();
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	printf("Rasterization complete.\n");

	// Fragment Shader
	printf("Fragment shader...\n");
	fragmentShader<<<numTriangles, THREADS_PER_BLOCK>>>(d_pixels, d_width, d_height, d_vertices, d_indices, d_fragments, d_interpolation, d_triangles);
	cudaDeviceSynchronize();
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	printf("Fragment shader complete.\n");

	// Geometry Shader
}

void bindImage(uchar4 image[], int imageWidth, int imageHeight) {
	pixels = image;
	width = imageWidth;
	height = imageHeight;

	cudaFree(d_pixels);
	cudaFree(d_width);
	cudaFree(d_height);

	cudaMalloc((void **)&d_pixels, sizeof(uchar4) * imageWidth * imageHeight);
	cudaMalloc((void **)&d_width, sizeof(int));
	cudaMalloc((void **)&d_height, sizeof(int));

	cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
}

void bindVertices(float3 vertices[], int length) {
	numVertices = length;

	cudaFree(d_vertices);
	cudaFree(d_numVertices);

	cudaMalloc((void **)&d_numVertices, sizeof(int));
	cudaMalloc((void **)&d_vertices, sizeof(float3) * length);

	cudaMemcpy(d_numVertices, &numVertices, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertices, vertices, sizeof(float3) * length, cudaMemcpyHostToDevice);
}

void bindIndices(int3 indices[], int length) {
	cudaFree(d_indices);
	numTriangles = length;
	cudaFree(d_indices);
	cudaMalloc((void **)&d_indices, sizeof(int3) * length);
	cudaMemcpy(d_indices, indices, sizeof(int3) * length, cudaMemcpyHostToDevice);
}

void draw() {
	float3 *interpolation = new float3[width * height];
	int *triangles = new int[width * height];
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			interpolation[x + y * width] = make_float3(-1.0, -1.0, -1.0);
			triangles[x + y * width] = -1;
		}
	}

	fragment *d_fragments;
	float3 *d_interpolation;
	int *d_triangles;

	cudaMalloc((void **)&d_fragments, sizeof(fragment) * numTriangles * THREADS_PER_BLOCK);
	cudaMalloc((void **)&d_interpolation, sizeof(float3) * width * height);
	cudaMalloc((void **)&d_triangles, sizeof(int) * width * height);

	cudaMemcpy(d_interpolation, interpolation, sizeof(float3) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_triangles, triangles, sizeof(int) * width * height, cudaMemcpyHostToDevice);

	delete[] interpolation;
	delete[] triangles;

	pipeline(d_fragments, d_interpolation, d_triangles);

	cudaMemcpy(pixels, d_pixels, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

	cudaFree(d_fragments);
	cudaFree(d_interpolation);
	cudaFree(d_triangles);
}