#define _USE_MATH_DEFINES
#include "common.h"

// Vectors

bool equal(float3 v, float3 u) {
	return v.x == u.x && v.y == u.y && v.z == u.z;
}

float length(float4 v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float length(float3 v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float3 negate(float3 v) {
	return make_float3(-v.x, -v.y, -v.z);
}

float3 subtract(float3 v, float3 u) {
	return make_float3(v.x - u.x, v.y - u.y, v.z - u.z);
}

float4 normalize(float4 v) {
	float l = length(v);
	return make_float4(v.x / l, v.y / l, v.z / l, 1);
}

float3 normalize(float3 v) {
	float l = length(v);
	return make_float3(v.x / l, v.y / l, v.z / l);
}

float3 cross(float3 u, float3 v) {
	return make_float3(u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x
		);
}

float4 mix(float4 v1, float4 v2, float s) {
	return make_float4(
		(1.0 - s) * v1.x + s * v2.x,
		(1.0 - s) * v1.y + s * v2.y,
		(1.0 - s) * v1.z + s * v2.z,
		1
		);
}

float dot(float3 v, float3 u){
	return v.x * u.x + v.y * u.y + v.z * u.z;
}

float4 mult(float f, float4 v) {
	return make_float4(f * v.x, f * v.y, f * v.z, 1);
}

// Meshes

float4* divideTriangle(float4 vertices[], float radius, float4 v1, float4 v2, float4 v3, int count) {
	if (count > 0) {
		float4 v1_2 = mult(radius, normalize(mix(v1, v2, 0.5))),
			v1_3 = mult(radius, normalize(mix(v1, v3, 0.5))),
			v2_3 = mult(radius, normalize(mix(v2, v3, 0.5)));
		vertices = divideTriangle(vertices, radius, v1, v1_2, v1_3, count - 1);
		vertices = divideTriangle(vertices, radius, v1_2, v2, v2_3, count - 1);
		vertices = divideTriangle(vertices, radius, v2_3, v3, v1_3, count - 1);
		return divideTriangle(vertices, radius, v1_2, v2_3, v1_3, count - 1);
	} else {
		vertices[0] = v1;
		vertices[1] = v3;
		vertices[2] = v2;
		return vertices + 3;
	}
}

void tetrahedron(float4 vertices[], float radius, float4 v1, float4 v2, float4 v3, float4 v4, int count) {
	vertices = divideTriangle(vertices, radius, v1, v2, v3, count);
	vertices = divideTriangle(vertices, radius, v4, v3, v2, count);
	vertices = divideTriangle(vertices, radius, v1, v4, v2, count);
	vertices = divideTriangle(vertices, radius, v1, v3, v4, count);
}

float4* makeSphere(float3 center, float radius, int subdivisions) {
	float4 *vertices = new float4[(int)(3 * pow(4.0, subdivisions + 1))];
	float4 v1 = make_float4(center.x, center.y, center.z - radius, 1),
		v2 = make_float4(center.x, center.y + 0.942809 * radius, center.z + 0.333333 * radius, 1),
		v3 = make_float4(center.x - 0.816497 * radius, center.y - 0.471405 * radius, center.z + 0.333333 * radius, 1),
		v4 = make_float4(center.x + 0.816497 * radius, center.y - 0.471405 * radius, center.z + 0.333333 * radius, 1);
	tetrahedron(vertices, radius, v1, v2, v3, v4, subdivisions);
	return vertices;
}

// Matrices

float4* det4() {
	float4 *mat = new float4[4];
	for (int i = 0; i < 4; i++)
		mat[i] = make_float4(0, 0, 0, 0);
	mat[0].x = 1;
	mat[1].y = 1;
	mat[2].z = 1;
	mat[3].w = 1;
	return mat;
}

float4* lookAt(float3 eye, float3 at, float3 up) {
	if (equal(eye, at))
		return det4();

	float3 v = normalize(subtract(at, eye)),
		n = normalize(cross(v, up)),
		u = normalize(cross(n, v));

	v = negate(v);

	float4 mat[4] = {
		/*
		make_float4(n.x, n.y, n.z, -dot(n, eye)),
		make_float4(u.x, u.y, u.z, -dot(u, eye)),
		make_float4(v.x, v.y, v.z, -dot(v, eye)),
		make_float4(0, 0, 0, 1) }; */

		make_float4(n.x, u.x, v.x, 0),
		make_float4(n.y, u.y, v.y, 0),
		make_float4(n.z, u.z, v.z, 0),
		make_float4(-dot(n, eye), -dot(u, eye), -dot(v, eye), 0)};
	return mat;
}

float4* computeFOVProjection(float fovy, float aspect, float nearPlane, float farPlane){
	/*float f = 1.0 / tan((fovy / 180 * M_PI) / 2);
	//float xScale = yScale / aspect;
	float d = farPlane - nearPlane;

	float4 v
	/*= make_float4(f / aspect, 0, 1, 0),
	n = make_float4(0, f, 0, 0),
	u = make_float4(0, 0, -(nearPlane + farPlane) / d, -2 * nearPlane * farPlane / d),
	w = make_float4(0, 0, -1, 0);
	*/
	/*
		= make_float4(f / aspect, 0, 0, 0),
		n = make_float4(0, f, 0, 0),
		u = make_float4(0, 0, -(nearPlane + farPlane) / d, - 1),
		w = make_float4(0, 0, -2 * nearPlane * farPlane / d, 0);*/

	float top = nearPlane * tan(M_PI / 180 * fovy / 2);
	float bottom = -top;
	float right = top * aspect;
	float left = -right;

	float4 v = make_float4((2*nearPlane)/(right-left), 0, (right+left)/(right-left), 0),
		n = make_float4(0, (2*nearPlane)/(top-bottom), (top+bottom)/(top-bottom), 0),
		u = make_float4(0, 0, -(farPlane + nearPlane) / (farPlane - nearPlane), -(2 * farPlane*nearPlane) / (farPlane - nearPlane)),
		w = make_float4(0, 0, -1, 0);

	float4 proj[4] = {
		v,
		n,
		u,
		w
	};
	return proj;
}

/*
function lookAt(eye, at, up)
{

	if (equal(eye, at)) {
		return mat4();
	}

	var v = normalize(subtract(at, eye));  // view direction vector
	var n = normalize(cross(v, up));       // perpendicular vector
	var u = normalize(cross(n, v));        // "new" up vector

	v = negate(v);

	var result = mat4(
		vec4(n, -dot(n, eye)),
		vec4(u, -dot(u, eye)),
		vec4(v, -dot(v, eye)),
		vec4()
		);

	return result;
}
*/