#include <vector> 
#include <iostream> 
#include <thrust/transform.h> 
#include <thrust/functional.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 

using namespace std;
using namespace thrust;

#define THREADS_PER_BLOCK 64
#define SQRT_TPB 8

struct fragment {
	int i_x, i_y, i_width, i_height, index;

	fragment(int i_x, int i_y, int i_width, int height, int index) {
		this->i_x = i_x;
		this->i_y = i_y;
		this->index = index;
	}
};

thrust::device_vector<fragment> fragments;

/*
// Rasterizes a fragment.
__global__ void rasterizeFragment(int i_x, int i_y, int i_width, int i_height, int width, int height, float3 *vertices, int3 *indices) {
	// Retrieve Vertices
	int3 index = indices[blockIdx.x];
	float3 v1 = vertices[index.x],
		v2 = vertices[index.y],
		v3 = vertices[index.z];

	// Image Coordinates
	float i_v1x = v1.x * width,
		i_v1y = v1.y * height,
		i_v2x = v2.x * width,
		i_v2y = v2.y * height,
		i_v3x = v3.x * width,
		i_v3y = v3.y * height;

	// Triangle Bounding Box
	float left = fmin(v1.x, fmin(v2.x, v3.x)) * width,
		right = fmax(v1.x, fmax(v2.x, v3.x)) * width,
		bottom = fmin(v1.y, fmin(v2.y, v3.y)) * height,
		top = fmax(v1.y, fmax(v2.y, v3.y)) * height;

	// Barycentric Init
	float alpha_denom = (i_v2y - i_v3y) * (i_v1x - i_v3x) + (i_v3x - i_v2x) * (i_v1y - i_v3y),
		beta_denom = (i_v2y - i_v3y) * (i_v1x - i_v3x) + (i_v3x - i_v2x) * (i_v1y - i_v3y);

	for (int x = round(left); x < right; x++) {
		for (int y = round(bottom); y < top; y++) {
			float i_x = x + 0.5, i_y = y + 0.5,
				alpha = ((i_v2y - i_v3y) * (i_x - i_v3x) + (i_v3x - i_v2x) * (i_y - i_v3y)) / alpha_denom,
				beta = ((i_v3y - i_v1y) * (i_x - i_v3x) + (i_v1x - i_v3x) * (i_y - i_v3y)) / beta_denom,
				gamma = 1.0f - alpha - beta;
			if (0.0 < alpha && 0.0 < beta && 0.0 < gamma)
				pixels[x + y * (*width)] = make_uchar4(255, 0, 0, 255);
		}
	}
}
*/

// Rasterizes a triangle.
__global__ void rasterizeTriangle(uchar4 *pixels, int *width, int *height, float3 *vertices, int3 *indices) {
	// Retrieve Vertices
	int3 index = indices[blockIdx.x];
	float3 v1 = vertices[index.x],
		v2 = vertices[index.y],
		v3 = vertices[index.z];

	// Image Coordinates
	float i_v1x = v1.x * (*width),
		i_v1y = v1.y * (*height),
		i_v2x = v2.x * (*width),
		i_v2y = v2.y * (*height),
		i_v3x = v3.x * (*width),
		i_v3y = v3.y * (*height);

	// Triangle Bounding Box
	float t_left = fmin(v1.x, fmin(v2.x, v3.x)) * (*width),
		t_right = fmax(v1.x, fmax(v2.x, v3.x)) * (*width),
		t_bottom = fmin(v1.y, fmin(v2.y, v3.y)) * (*height),
		t_top = fmax(v1.y, fmax(v2.y, v3.y)) * (*height);

	int f_x = threadIdx.x % SQRT_TPB,
		f_y = threadIdx.x / SQRT_TPB,
		f_width = ceil((t_right - t_left) / SQRT_TPB),
		f_height = ceil((t_top - t_bottom) / SQRT_TPB);

	printf("%d, %d, %d, %d\n", f_x, f_y, f_width, f_height);

	// Barycentric Init
	float alpha_denom = (i_v2y - i_v3y) * (i_v1x - i_v3x) + (i_v3x - i_v2x) * (i_v1y - i_v3y),
		beta_denom = (i_v2y - i_v3y) * (i_v1x - i_v3x) + (i_v3x - i_v2x) * (i_v1y - i_v3y);

	// Rasterize
	for (int x = t_left + f_x * f_width; x < t_left + f_x * f_width + f_width; x++) {
		for (int y = t_bottom + f_y * f_height; y < t_bottom + f_y * f_height + f_height; y++) {
			float i_x = x + 0.5, i_y = y + 0.5,
				alpha = ((i_v2y - i_v3y) * (i_x - i_v3x) + (i_v3x - i_v2x) * (i_y - i_v3y)) / alpha_denom,
				beta = ((i_v3y - i_v1y) * (i_x - i_v3x) + (i_v1x - i_v3x) * (i_y - i_v3y)) / beta_denom,
				gamma = 1.0f - alpha - beta;
			if (0.0 < alpha && 0.0 < beta && 0.0 < gamma) {
				pixels[x + y * (*width)] = make_uchar4(255, 0, 0, 255);
			}
		}
	}
}

void rasterize(uchar4 *pixels, int width, int height, float3 *vertices, int3 *indices, int numVertices, int numTriangles) {
	fragments = thrust::device_vector<fragment>();

	uchar4 *d_pixels;
	int *d_width, *d_height;
	float3 *d_vertices;
	int3 *d_indices;

	cudaMalloc((void **)&d_pixels, sizeof(uchar4) * width * height);
	cudaMalloc((void **)&d_width, sizeof(int));
	cudaMalloc((void **)&d_height, sizeof(int));
	cudaMalloc((void **)&d_vertices, sizeof(float3) * numVertices);
	cudaMalloc((void **)&d_indices, sizeof(int3) * numTriangles);

	cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertices, vertices, sizeof(float3) * numVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, indices, sizeof(int3) * numTriangles, cudaMemcpyHostToDevice);

	rasterizeTriangle<<<numTriangles, THREADS_PER_BLOCK>>>(d_pixels, d_width, d_height, d_vertices, d_indices);

	cudaMemcpy(pixels, d_pixels, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

	cudaFree(d_pixels);
	cudaFree(d_width);
	cudaFree(d_height);
	cudaFree(d_vertices);
	cudaFree(d_indices);
}