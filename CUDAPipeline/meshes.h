#include <cuda_runtime.h>
#include <helper_functions.h>

float3 mix(float3 v1, float3 v2, float s);
float3 normalize(float3 v);
float3 mult(float f, float3 v);
float3* divideTriangle(float3 vertices[], float radius, float3 v1, float3 v2, float3 v3, int count);
void tetrahedron(float3 vertices[], float radius, float3 v1, float3 v2, float3 v3, float3 v4, int count);
float3* makeSphere(float3 center, float radius, int subdivisions);