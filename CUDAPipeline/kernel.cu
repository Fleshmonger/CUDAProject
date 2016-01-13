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

uchar4* rasterize(int w, int h, float3 *vertices, int *indices) {
	uchar4* frags = (uchar4 *)malloc(sizeof(uchar4) * w * h);;

	float3 v1 = *(vertices + indices[0]),
		v2 = *(vertices + indices[1]),
		v3 = *(vertices + indices[2]);

	float x1 = fmin(v1.x, fmin(v2.x, v3.x)) * w,
		x2 = fmax(v1.x, fmax(v2.x, v3.x)) * w,
		y1 = fmin(v1.y, fmin(v2.y, v3.y)) * h,
		y2 = fmax(v1.y, fmax(v2.y, v3.y)) * h;

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
			frags[x + bX + w * (y + bY)] = block[x + y * bW];

	free(block);
	cudaFree(d_bW);
	cudaFree(d_bH);
	cudaFree(d_block);

	return frags;
}

extern "C" int test(void) {
	vector<float>  vh = { 0, 1, 2, 3, 4, 5, 6, 7 };
	device_vector<float> v = vh;
	device_vector<float> v_out(v.size());
	thrust::transform(v.begin(), v.end(), v_out.begin(),
		[=] __device__(float x) {
		return x*42 + 7;
	});
	for (size_t i = 0; i < v_out.size(); i++)
		std::cout << v_out[i] << std::endl;
	return 0;
}