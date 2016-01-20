#include <cuda_runtime.h>
#include <helper_functions.h>

bool equal(float3 v, float3 u);
float length(float3 v);
float3 negate(float3 v);
float3 subtract(float3 v, float3 u);
float3 normalize(float3 v);
float3 cross(float3 u, float3 v);
float3 mix(float3 v1, float3 v2, float s);
float3 mult(float f, float3 v);

float3* divideTriangle(float3 vertices[], float radius, float3 v1, float3 v2, float3 v3, int count);
void tetrahedron(float3 vertices[], float radius, float3 v1, float3 v2, float3 v3, float3 v4, int count);
float3* makeSphere(float3 center, float radius, int subdivisions);

float3* det3();
float3* lookAt(float3 eye, float3 at, float3 up);