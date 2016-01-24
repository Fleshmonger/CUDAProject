#include <cuda_runtime.h>
#include <helper_functions.h>

bool equal(float3 v, float3 u);
float length(float4 v);
float length(float3 v);
float3 negate(float3 v);
float4 subtract(float4 v, float4 u);
float3 normalize(float3 v);
float4 normalize(float4 v);
float3 cross(float3 u, float3 v);
float4 mix(float4 v1, float4 v2, float s);
float4 mult(float f, float4 v);

void tetrahedron(float4 vertices[], float radius, float4 v1, float4 v2, float4 v3, float4 v4, int count);
float4* makeSphere(float3 center, float radius, int subdivisions);
float4* divideTriangle(float4 vertices[], float radius, float4 v1, float4 v2, float4 v3, int count);
float4* computeFOVProjection(float fovy, float aspect, float nea, float fa);

float4* det4();
float4* lookAt(float3 eye, float3 at, float3 up);