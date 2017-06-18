#pragma once

#ifndef _MESH_H_
#define _MESH_H_

#include "Object.h"
#include "CudaVector.h"

#define MESH_TYPE 4

struct MeshTriangle {
	// mesh
	int index[3];
	KVector3 center;
	// precompute
	float d[3];
	KVector3 e[3];
};

class Mesh : public Object
{
public:
	// Constructors
	__host__ __device__ Mesh(__in const KPoint3& points, __in const Color& diffuse,
		__in const Color& specular, __in const float shininess,
		__in const float reflectance, __in const float transmittance,
		__in const float density);
	__host__ __device__ Mesh(__in const Mesh& cpy);
	// Destructors
	__host__ __device__ virtual ~Mesh();

private:
	Cuda::Vector<KVector3> vertices;
	Cuda::Vector<KVector3> normals;
	Cuda::Vector<MeshTriangle> triangles;
	KPoint3 mesh_center;

public:
	__host__ __device__ Mesh& operator =(const Mesh& other);
	__host__ __device__ virtual Object* GetHeapCopy() const;
	__host__ __device__ virtual void GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const;
	__host__ __device__ virtual void GetNormal(__in const KPoint3& point, __out KVector3& normal) const;
	__host__ __device__ virtual int GetType() const;
	__host__ __device__ virtual KPoint3 GetPosition() const;
};

#endif