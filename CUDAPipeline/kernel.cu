#include <vector> 
#include <iostream> 
#include <thrust/transform.h> 
#include <thrust/functional.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 

using namespace std;
using namespace thrust;

/*

*/

__global__ void rasterizeTriangle(uchar4 *pixels, int *width, int *height, float3 *vertices, int3 *indices) {
	int3 index = indices[blockIdx.x];
	float3 v1 = vertices[index.x],
		v2 = vertices[index.y],
		v3 = vertices[index.z];
	float i_v1x = v1.x * (*width),
		i_v1y = v1.y * (*height),
		i_v2x = v2.x * (*width),
		i_v2y = v2.y * (*height),
		i_v3x = v3.x * (*width),
		i_v3y = v3.y * (*height);
	int left = fmin(v1.x, fmin(v2.x, v3.x)) * (*width),
		right = fmax(v1.x, fmax(v2.x, v3.x)) * (*width),
		bottom = fmin(v1.y, fmin(v2.y, v3.y)) * (*height),
		top = fmax(v1.y, fmax(v2.y, v3.y)) * (*height);
	for (int x = left; x < right; x++) {
		for (int y = bottom; y < top; y++) {
			float alpha = ((i_v2y - i_v3y)*(x - i_v3x) + (i_v3x - i_v2x)*(y - i_v3y)) /
				((i_v2y - i_v3y)*(i_v1x - i_v3x) + (i_v3x - i_v2x)*(i_v1y - i_v3y)),
				beta = ((i_v3y - i_v1y)*(x - i_v3x) + (i_v1x - i_v3x)*(y - i_v3y)) /
				((i_v2y - i_v3y)*(i_v1x - i_v3x) + (i_v3x - i_v2x)*(i_v1y - i_v3y)),
				gamma = 1.0f - alpha - beta;
			if (0 < alpha && 0 < beta && 0 < gamma)
				pixels[x + y * (*width)] = make_uchar4(255, 0, 0, 255);
		}
	}
}

void rasterize(uchar4 *pixels, int width, int height, float3 *vertices, int3 *indices, int vLength, int iLength) {
	uchar4 *d_pixels;
	int *d_width, *d_height;
	float3 *d_vertices;
	int3 *d_indices;

	cudaMalloc((void **)&d_pixels, sizeof(uchar4) * width * height);
	cudaMalloc((void **)&d_width, sizeof(int));
	cudaMalloc((void **)&d_height, sizeof(int));
	cudaMalloc((void **)&d_vertices, sizeof(float3) * vLength);
	cudaMalloc((void **)&d_indices, sizeof(int3) * iLength);

	cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertices, vertices, sizeof(float3) * vLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, indices, sizeof(int3) * iLength, cudaMemcpyHostToDevice);

	rasterizeTriangle<<<1, 1 >>>(d_pixels, d_width, d_height, d_vertices, d_indices);

	cudaMemcpy(pixels, d_pixels, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

	cudaFree(d_pixels);
	cudaFree(d_width);
	cudaFree(d_height);
	cudaFree(d_vertices);
	cudaFree(d_indices);
}