#include "common.h"

// Vectors

bool equal(float3 v, float3 u) {
	return v.x == u.x && v.y == u.y && v.z == u.z;
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

float3 mix(float3 v1, float3 v2, float s) {
	return make_float3(
		(1.0 - s) * v1.x + s * v2.x,
		(1.0 - s) * v1.y + s * v2.y,
		(1.0 - s) * v1.z + s * v2.z
		);
}

float3 mult(float f, float3 v) {
	return make_float3(f * v.x, f * v.y, f * v.z);
}

// Meshes

float3* divideTriangle(float3 vertices[], float radius, float3 v1, float3 v2, float3 v3, int count) {
	if (count > 0) {
		float3 v1_2 = mult(radius, normalize(mix(v1, v2, 0.5))),
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

void tetrahedron(float3 vertices[], float radius, float3 v1, float3 v2, float3 v3, float3 v4, int count) {
	vertices = divideTriangle(vertices, radius, v1, v2, v3, count);
	vertices = divideTriangle(vertices, radius, v4, v3, v2, count);
	vertices = divideTriangle(vertices, radius, v1, v4, v2, count);
	vertices = divideTriangle(vertices, radius, v1, v3, v4, count);
}

float3* makeSphere(float3 center, float radius, int subdivisions) {
	float3 *vertices = new float3[(int) (3 * pow(4.0, subdivisions + 1))];
	float3 v1 = make_float3(center.x, center.y, center.z - radius),
		v2 = make_float3(center.x, center.y + 0.942809 * radius, center.z + 0.333333 * radius),
		v3 = make_float3(center.x - 0.816497 * radius, center.y - 0.471405 * radius, center.z + 0.333333 * radius),
		v4 = make_float3(center.x + 0.816497 * radius, center.y - 0.471405 * radius, center.z + 0.333333 * radius);
	tetrahedron(vertices, radius, v1, v2, v3, v4, subdivisions);
	return vertices;
}

// Matrices

float3* det3() {
	float3 *mat = new float3[3];
	for (int i = 0; i < 3; i++)
		mat[i] = make_float3(0, 0, 0);
	mat[0].x = 1;
	mat[1].y = 1;
	mat[2].z = 1;
	return mat;
}

float3* lookAt(float3 eye, float3 at, float3 up) {
	if (equal(eye, at))
		return det3();

	float3 v = normalize(subtract(at, eye)),
		n = normalize(cross(v, up)),
		u = normalize(cross(n, v));

	v = negate(v);

	float3 mat[3] = {
		n,
		u,
		v
	};
	return mat;
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