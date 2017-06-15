#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "Object.h"

#define TRIANGLE_TYPE 3

class Triangle : public Object
{
public:
	// Constructors
	__host__ __device__ Triangle(__in const KPoint3 points[3], __in const Color& diffuse,
			__in const Color& specular, __in const float shininess,
			__in const float reflectance, __in const float transmittance, __in const float density);
	__host__ __device__ Triangle(__in const Triangle& cpy);
	// Destructors
	__host__ __device__ virtual ~Triangle();

private:
	// Members
	KPoint3 points[3];

public:
	// Methods
	__host__ __device__ Triangle& operator =(const Triangle& other);
	__host__ __device__ virtual Object* GetHeapCopy() const;
	__host__ __device__ virtual void GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const;
	__host__ __device__ virtual void GetNormal(__in const KPoint3& point, __out KVector3& normal) const;
	__host__ __device__ virtual void GetNormal(__out KVector3& normal) const;	// overload
	__host__ __device__ virtual int GetType() const;
	__host__ __device__ virtual KPoint3 GetPosition() const;
};

Triangle::Triangle(__in const KPoint3 points[3], __in const Color& diffuse,
	__in const Color& specular, __in const float shininess,
	__in const float reflectance, __in const float transmittance, __in const float density)
	: Object(diffuse, specular, shininess, reflectance, transmittance, density),
	points{ points[0], points[1], points[2] }
{}

Triangle::Triangle(__in const Triangle& cpy)
	: Object(cpy), points{ cpy.points[0], cpy.points[1], cpy.points[2] }
{}

Triangle::~Triangle()
{}

Triangle& Triangle::operator =(const Triangle& other)
{
	this->ambient = other.ambient;
	this->diffuse = other.diffuse;
	this->specular = other.specular;
	this->shininess = other.shininess;
	this->reflectance = other.reflectance;
	this->transmittance = other.transmittance;
	this->density = other.density;
	this->id = other.id;
	for (int i = 0; i < 3; i++)
		this->points[i] = other.points[i];

	return *this;
}

Object* Triangle::GetHeapCopy() const
{
	return new Triangle(*this);
}

void Triangle::GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const
{
	// get plane normal
	KVector3 n;
	GetNormal(n);

	// p(t) = eye + d*t;
	// solve t
	KVector3 eye = cast_vec3(ray.GetPoint());
	KVector3 dir = ray.GetDirection();
	dir = dir.Normalize();
	float t = -(n * eye - (n * cast_vec3(this->points[0]))) / (n * dir);

	// check whether plane is behind camera
	if (t <= 0.0f)
	{
		is_intersect = false;
		return;
	}

	KPoint3 plane_point = KPoint3(ray.GetPoint() + dir * t);

	// check whether point is inside triangle
	// Compute vectors        
	KVector3 v0 = this->points[2] - this->points[0];
	KVector3 v1 = this->points[1] - this->points[1];
	KVector3 v2 = plane_point - this->points[0];

	// Compute dot products
	float dot00 = v0 * v0;
	float dot01 = v0 * v1;
	float dot02 = v0 * v2;
	float dot11 = v1 * v1;
	float dot12 = v1 * v2;

	// Compute barycentric coordinates
	float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is in triangle
	if ((u >= 0.0f) && (v >= 0.0f) && (u + v < 1.0f))
	{
		intersect_point = plane_point;
		is_intersect = true;
	}
	else
		is_intersect = false;
}

void Triangle::GetNormal(__in const KPoint3& point, __out KVector3& normal) const
{
	// same normals regardless of point
	KVector3 tempV1 = (this->points[0] - this->points[1]).Normalize();
	KVector3 tempV2 = (this->points[2] - this->points[1]).Normalize();

	normal = (tempV2^tempV1).Normalize();
}

void Triangle::GetNormal(__out KVector3& normal) const
{
	// same normals regardless of point
	KVector3 tempV1 = (this->points[0] - this->points[1]).Normalize();
	KVector3 tempV2 = (this->points[2] - this->points[1]).Normalize();

	normal = (tempV2^tempV1).Normalize();
}

int Triangle::GetType() const
{
	return TRIANGLE_TYPE;
}

KPoint3 Triangle::GetPosition() const
{
	return KPoint3((this->points[0][0] + this->points[1][0] + this->points[2][0]) / 3,
					(this->points[0][1] + this->points[1][1] + this->points[2][1]) / 3,
					(this->points[0][2] + this->points[1][2] + this->points[2][2]) / 3);
}

#endif