#include <vector> 
#include <iostream> 
#include <thrust/transform.h> 
#include <thrust/functional.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 

using namespace std;
using namespace thrust;

namespace flex {
	#define THREADS_PER_BLOCK 256
	#define SQRT_TPB 16

	struct fragment {
		int i_x, i_y, i_width, i_height, index;

		__host__ __device__ fragment() {
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

	bool *d_cull;
	int width = 0, height = 0, numVertices = 0, numTriangles = 0, *triangles, *d_width, *d_height, *d_numVertices, *d_triangles;
	int3 *d_indices;
	float3 *interpolation, *d_interpolation, *d_lightPos, *d_N, *d_E, *d_L;
	float4 *d_projMat, *d_modelMat, *d_vertices, *d_verticesTransformed, *d_normals, *d_normalMatrix;
	uchar4 *background, *pixels, *d_pixels;
	fragment *fragments, *d_fragments;

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

	__device__ float4 d_negate(float4 v) {
		return make_float4(-v.x, -v.y, -v.z, 1);
	}

	__device__ float d_max(float a, float b, float c) {
		return d_max(a, d_max(b, c));
	}

	__device__ float d_length(float3 v) {
		return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	}

	__device__ float d_length(float4 v) {
		return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	}

	__device__ float d_dot(float3 v, float3 u){
		return v.x * u.x + v.y * u.y + v.z * u.z;
	}

	__device__ float3 d_normalize(float3 v) {
		float l = d_length(v);
		return make_float3(v.x / l, v.y / l, v.z / l);
	}

	__device__ float4 d_normalize(float4 v) {
		float l = d_length(v);
		return make_float4(v.x / l, v.y / l, v.z / l, 1);
	}
	__device__ float d_ceil(float a) {
		return a - (int)a > 0.0 ? (int)a + 1.0 : a;
	}

	__device__ float4 d_cross(float4 u, float4 v) {
		return make_float4(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x, 1);
	}

	__device__ float3 d_subtract(float3 v, float3 u) {
		return make_float3(v.x - u.x, v.y - u.y, v.z - u.z);
	}

	__device__ float4 d_subtract(float4 v, float4 u) {
		return make_float4(v.x - u.x, v.y - u.y, v.z - u.z, 1);
	}

	__device__ float3 d_add(float3 v, float3 u) {
		return make_float3(v.x + u.x, v.y + u.y, v.z + u.z);
	}

	__device__ float d_pow(float n, int p){
		float num = 1;
		for (int i = 0; i < p; i++){
			num *= n;
		}
		return num;
	}

	__device__ float3 d_mult(float3 mat, float v) {
		return make_float3(mat.x * v, mat.y * v, mat.z * v);
	}

	__device__ float3 d_mult(float4 mat[], float3 v) {
		return make_float3(
			mat[0].x * v.x + mat[1].x * v.y + mat[2].x * v.z,
			mat[0].y * v.x + mat[1].y * v.y + mat[2].y * v.z,
			mat[0].z * v.x + mat[1].z * v.y + mat[2].z * v.z
			);
	}

	__device__ float4 d_mult(float4 mat[], float4 v) {
		return make_float4(
			mat[0].x * v.x + mat[1].x * v.y + mat[2].x * v.z + mat[3].x * v.w,
			mat[0].y * v.x + mat[1].y * v.y + mat[2].y * v.z + mat[3].y * v.w,
			mat[0].z * v.x + mat[1].z * v.y + mat[2].z * v.z + mat[3].z * v.w,
			mat[0].w * v.x + mat[1].w * v.y + mat[2].w * v.z + mat[3].w * v.w
			);
	}

	/*
	__device__ float4* d_mult(float4 mata[], float4 matb[]) {
		float4 mat[4] = {
			mult(mata, matb[0]),
			mult(mata, matb[1]),
			mult(mata, matb[2]),
			mult(mata, matb[3])
		};
		return mat;
	}*/

	// Runs the vertex shader on a vertex.
	__global__ void vertexShader(int *width, int *height, float4 vertices[], float4 verticesTransformed[], int *numVertices, float4 projMat[], float4 modelMat[], float4 normals[], float4 normalMatrix[], float3 *lightPos, float3 N[], float3 E[], float3 L[]) {
		// Retrieve Vertex.
		int index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
		if (index >= *numVertices)
			return;
		//verticesTransformed[index] = mult(mult(projMat, modelMat), vertices[index]);
		verticesTransformed[index] = d_mult(projMat, d_mult(modelMat, vertices[index]));
		//verticesTransformed[index] = vertices[index];

		float4 pos = d_mult(modelMat, vertices[index]),
			varN = d_normalize(d_mult(normalMatrix, normals[index])),
			varE = d_negate(d_normalize(pos));

		N[index] = make_float3(varN.x, varN.y, varN.z);
		E[index] = make_float3(varE.x, varE.y, varE.z);
		L[index] = d_normalize(d_subtract(*lightPos, make_float3(pos.x, pos.y, pos.z)));

	}

	// Rasterizes a triangle.
	__global__ void rasterizeTriangle(int *width, int *height, float4 vertices[], int3 indices[], fragment fragments[], float3 interpolation[], int triangles[], bool *cull) {
		// Retrieve Vertices
		int3 index = indices[blockIdx.x];
		float4 v1 = vertices[index.x],
			v2 = vertices[index.y],
			v3 = vertices[index.z];

		// Face Culling
		if (*cull && d_cross(d_subtract(v2, v1), d_subtract(v3, v1)).z < 0.0)
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
		int f_width = d_ceil(((float)(t_right - t_left)) / SQRT_TPB),
			f_height = d_ceil(((float)(t_top - t_bottom)) / SQRT_TPB),
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

	__device__ float3 interpolateVector(float3 v1, float3 v2, float3 v3, float alpha, float beta, float gamma) {
		return make_float3(
			v1.x * alpha + v2.x * beta + v3.x * gamma,
			v1.y * alpha + v2.y * beta + v3.y * gamma,
			v1.z * alpha + v2.z * beta + v3.z * gamma
			);
	}

	__device__ float3 interpolateVector(float4 v1, float4 v2, float4 v3, float alpha, float beta, float gamma) {
		return make_float3(
			v1.x * alpha + v2.x * beta + v3.x * gamma,
			v1.y * alpha + v2.y * beta + v3.y * gamma,
			v1.z * alpha + v2.z * beta + v3.z * gamma
			);
	}

	// Runs the fragment shader on a fragment.
	__global__ void fragmentShader(uchar4 *pixels, int *width, int *height, float4 vertices[], int3 indices[], fragment fragments[], float3 interpolation[], int triangles[], float3 N[], float3 E[], float3 L[]) {
		// Retrieve Fragment
		fragment frag = fragments[threadIdx.x + blockIdx.x * THREADS_PER_BLOCK];

		float3 ambientProduct = make_float3(0.1, 0.1, 0.1), diffuseProduct = make_float3(0.5, 0.5, 0.5), specularProduct = make_float3(0.2, 0.2, 0.2);
		float shininess = 5;

		// Copy Fragment Pixels
		for (int x = frag.i_x; x < d_min(frag.i_x + frag.i_width, *width); x++) {
			for (int y = frag.i_y; y < d_min(frag.i_y + frag.i_height, *height); y++) {
				int index = x + y * (*width);
				if (triangles[index] == frag.index) {
					int3 vIndex = indices[triangles[index]];

					// Phong Shading
					float alpha = interpolation[index].x,
						beta = interpolation[index].y,
						gamma = interpolation[index].z;
					float3 varE = interpolateVector(E[vIndex.x], E[vIndex.y], E[vIndex.z], alpha, beta, gamma),
						varN = interpolateVector(N[vIndex.x], N[vIndex.y], N[vIndex.z], alpha, beta, gamma),
						varL = interpolateVector(L[vIndex.x], L[vIndex.y], L[vIndex.z], alpha, beta, gamma);

					float3 H = d_normalize(d_add(varL, varE));

					float3 ambient = ambientProduct;

					float Kd = d_max(d_dot(varN, varL), 0.0);
					float3 diffuse = d_mult(diffuseProduct, Kd);

					float Ks = d_pow(d_max(d_dot(varN, H), 0.0), shininess);
					float3 specular = d_mult(specularProduct, Ks);

					if (d_dot(varL, varN) < 0)
						specular = make_float3(0, 0, 0);

					float r = d_min((ambient.x + specular.x + diffuse.x) * 255, 255),
						g = d_min((ambient.y + specular.y + diffuse.y) * 255, 255),
						b = d_min((ambient.z + specular.z + diffuse.z) * 255, 255);
					pixels[index] = make_uchar4(r, g, b, 255);

					/*
					// Interpolation Shading
					float4 v1 = vertices[vIndex.x],
					v2 = vertices[vIndex.y],
					v3 = vertices[vIndex.z];

					float red = ((v1.x + 1.0) / 2) * interpolation[index].x + ((v2.x + 1.0) / 2) * interpolation[index].y + ((v3.x + 1.0) / 2) * interpolation[index].z,
						green = ((v1.y + 1.0) / 2) * interpolation[index].x + ((v2.y + 1.0) / 2) * interpolation[index].y + ((v3.y + 1.0) / 2) * interpolation[index].z,
						blue = ((v1.z + 1.0) / 2) * interpolation[index].x + ((v2.z + 1.0) / 2) * interpolation[index].y + ((v3.z + 1.0) / 2) * interpolation[index].z;
					pixels[index] = make_uchar4(red * 255, green * 255, blue * 255, 255);
					*/
					
					/*
					// Wireframe
					float k = 0.1;
					if (interpolation[index].x < k || interpolation[index].y < k || interpolation[index].z < k)
						pixels[index] = make_uchar4(255, 255, 255, 255);
					*/
				}
			}
		}
	}

	void pipeline() {
		// Vertex Shader
		//printf("Vertex shader...\n");
		vertexShader << <ceil(((float)numVertices) / THREADS_PER_BLOCK), THREADS_PER_BLOCK >> >(d_width, d_height, d_vertices, d_verticesTransformed, d_numVertices, d_projMat, d_modelMat, d_normals, d_normalMatrix, d_lightPos, d_N, d_E, d_L);
		cudaDeviceSynchronize();
		//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		//printf("Vertex shader complete.\n");

		// Rasterize
		//printf("Rasterization...\n");
		rasterizeTriangle<<<numTriangles, THREADS_PER_BLOCK>>>(d_width, d_height, d_verticesTransformed, d_indices, d_fragments, d_interpolation, d_triangles, d_cull);
		//cudaDeviceSynchronize();
		//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		//printf("Rasterization complete.\n");

		// Fragment Shader
		//printf("Fragment shader...\n");
		fragmentShader<<<numTriangles, THREADS_PER_BLOCK>>>(d_pixels, d_width, d_height, d_vertices, d_indices, d_fragments, d_interpolation, d_triangles, d_N, d_E, d_L);
		//cudaDeviceSynchronize();
		//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		//printf("Fragment shader complete.\n");

		// Geometry Shader
	}

	void init(bool cull, float3 lightPos) {
		cudaFree(d_cull);
		cudaFree(d_lightPos);

		cudaMalloc((void **)&d_cull, sizeof(bool));
		cudaMalloc((void **)&d_lightPos, sizeof(float3));

		cudaMemcpy(d_cull, &cull, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(d_lightPos, &lightPos, sizeof(float3), cudaMemcpyHostToDevice);
	}

	void bufferImage(uchar4 image[], int imageWidth, int imageHeight) {
		delete[] background;
		cudaFree(d_pixels);
		cudaFree(d_width);
		cudaFree(d_height);

		background = new uchar4[imageWidth * imageHeight];
		pixels = image;
		width = imageWidth;
		height = imageHeight;

		cudaMalloc((void **)&d_pixels, sizeof(uchar4) * width * height);
		cudaMalloc((void **)&d_width, sizeof(int));
		cudaMalloc((void **)&d_height, sizeof(int));

		cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);

		cudaFree(d_interpolation);
		cudaFree(d_triangles);
		delete[] interpolation;
		delete[] triangles;
		interpolation = new float3[width * height];
		triangles = new int[width * height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				interpolation[x + y * width] = make_float3(-1.0, -1.0, -1.0);
				triangles[x + y * width] = -1;
			}
		}

		cudaMalloc((void **)&d_interpolation, sizeof(float3) * width * height);
		cudaMalloc((void **)&d_triangles, sizeof(int) * width * height);
	}

	void bufferVertices(float4 vertices[], float4 normals[], int length) {
		numVertices = length;

		cudaFree(d_vertices);
		cudaFree(d_verticesTransformed);
		cudaFree(d_normals);
		cudaFree(d_numVertices);
		cudaFree(d_N);
		cudaFree(d_E);
		cudaFree(d_L);

		cudaMalloc((void **)&d_vertices, sizeof(float4) * numVertices);
		cudaMalloc((void **)&d_verticesTransformed, sizeof(float4) * numVertices);
		cudaMalloc((void **)&d_normals, sizeof(float4) * numVertices);
		cudaMalloc((void **)&d_numVertices, sizeof(int));
		cudaMalloc((void **)&d_N, sizeof(float3) * numVertices);
		cudaMalloc((void **)&d_E, sizeof(float3) * numVertices);
		cudaMalloc((void **)&d_L, sizeof(float3) * numVertices);

		cudaMemcpy(d_numVertices, &numVertices, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vertices, vertices, sizeof(float4) * numVertices, cudaMemcpyHostToDevice);
		cudaMemcpy(d_normals, normals, sizeof(float4)* numVertices, cudaMemcpyHostToDevice);
	}

	void bufferIndices(int3 indices[], int length) {
		numTriangles = length;
		cudaFree(d_indices);
		cudaMalloc((void **)&d_indices, sizeof(int3) * numTriangles);
		cudaMemcpy(d_indices, indices, sizeof(int3) * numTriangles, cudaMemcpyHostToDevice);

		cudaFree(d_fragments);
		delete[] fragments;
		fragments = new fragment[numTriangles * THREADS_PER_BLOCK];
		cudaMalloc((void **)&d_fragments, sizeof(fragment) * numTriangles * THREADS_PER_BLOCK);
	}

	void bufferProjectionMatrix(float4 matrix[]) {
		cudaFree(d_projMat);
		
		cudaMalloc((void **)&d_projMat, sizeof(float4) * 4);

		cudaMemcpy(d_projMat, matrix, sizeof(float4) * 4, cudaMemcpyHostToDevice);
	}

	void bufferModelViewMatrix(float4 matrix[], float4 normalMatrix[]) {
		cudaFree(d_modelMat);
		cudaFree(d_normalMatrix);

		cudaMalloc((void **)&d_modelMat, sizeof(float4) * 4);
		cudaMalloc((void **)&d_normalMatrix, sizeof(float4) * 4);

		cudaMemcpy(d_modelMat, matrix, sizeof(float4) * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(d_normalMatrix, normalMatrix, sizeof(float4) * 4, cudaMemcpyHostToDevice);
	}

	void render() {
		cudaMemcpy(d_fragments, fragments, sizeof(fragment) * numTriangles * THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
		cudaMemcpy(d_interpolation, interpolation, sizeof(float3) * width * height, cudaMemcpyHostToDevice);
		cudaMemcpy(d_triangles, triangles, sizeof(int) * width * height, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pixels, background, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice);

		pipeline();

		cudaMemcpy(pixels, d_pixels, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);
	}
}