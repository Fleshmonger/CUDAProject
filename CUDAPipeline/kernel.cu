#include <vector> 
#include <iostream> 
#include <thrust/transform.h> 
#include <thrust/functional.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 

using namespace std;
using namespace thrust;

__global__ void rasterizeBlock(uchar4 *frags, int *w, int *h) {
	for (int y = 0; y < *h; y++)
		for (int x = 0; x < *w; x++)
			frags[x + y * (*w)] = make_uchar4(255, 0, 0, 255);
}

void rasterize(uchar4 *pixels, int width, int height, float3 *vertices, int3 *indices) {
	int3 index = indices[0];
	float3 v1 = *(vertices + index.x),
		v2 = *(vertices + index.y),
		v3 = *(vertices + index.z);

	float x1 = fmin(v1.x, fmin(v2.x, v3.x)) * width,
		x2 = fmax(v1.x, fmax(v2.x, v3.x)) * width,
		y1 = fmin(v1.y, fmin(v2.y, v3.y)) * height,
		y2 = fmax(v1.y, fmax(v2.y, v3.y)) * height;

	printf("%f, %f, %f, %f", x1, x2, y1, y2);

	int bX = x1, bY = y1, bW = x2 - x1, bH = y2 - y1;
	int *d_bW, *d_bH;
	uchar4* block = (uchar4 *)malloc(sizeof(uchar4) * bW * bH);
	uchar4* d_block;

	cudaMalloc((void **) &d_bW, sizeof(int));
	cudaMalloc((void **) &d_bH, sizeof(int));
	cudaMalloc((void **) &d_block, sizeof(uchar4) * bW * bH);

	cudaMemcpy(d_bW, &bW, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bH, &bH, sizeof(int), cudaMemcpyHostToDevice);

	rasterizeBlock<<<1,1>>>(d_block, d_bW, d_bH);
	
	cudaMemcpy(block, d_block, sizeof(uchar4) * bW * bH, cudaMemcpyDeviceToHost);

	for (int y = 0; y < bH; y++)
		for (int x = 0; x < bW; x++)
			pixels[x + bX + width * (y + bY)] = block[x + y * bW];

	free(block);
	cudaFree(d_bW);
	cudaFree(d_bH);
	cudaFree(d_block);
}