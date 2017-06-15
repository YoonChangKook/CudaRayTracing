#ifndef _PLANE_H_
#define _PLANE_H_

#include "Object.h"

#define PLANE_TYPE 2

class Plane : public Object
{
public:
	// Constructors
	__host__ __device__ Plane(__in const KPoint3 points[3], __in const Color& diffuse, __in const Color& specular,
		__in const float shininess,	__in const float reflectance, __in const float transmittance, 
		__in const float density);
	__host__ __device__ Plane(__in const KVector3& normal, __in const KPoint3& point, __in const Color& diffuse,
		__in const Color& specular, __in const float shininess, __in const float reflectance, 
		__in const float transmittance, __in const float density);
	__host__ __device__ Plane(__in const Plane& cpy);

	// Destructors
	__host__ __device__ virtual ~Plane();

private:
	// Members
	KVector3 normal;
	KPoint3 point;
	float d;

public:
	// Methods
	__host__ __device__ Plane& operator =(const Plane& other);
	__host__ __device__ virtual Object* GetHeapCopy() const;
	__host__ __device__ virtual void GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, 
														__out bool& is_intersect) const;
	__host__ __device__ virtual void GetNormal(__in const KPoint3& point, __out KVector3& normal) const;
	__host__ __device__ virtual int GetType() const;
	__host__ __device__ virtual KPoint3 GetPosition() const;
};

Plane::Plane(__in const KPoint3 points[3], __in const Color& diffuse,
	__in const Color& specular, __in const float shininess, __in const float reflectance,
	__in const float transmittance, __in const float density)
	: Object(diffuse, specular, shininess, reflectance, transmittance, density), point(points[0])
{
	// calculate 
	KVector3 tempV1 = (points[0] - points[1]).Normalize();
	KVector3 tempV2 = (points[2] - points[1]).Normalize();
	this->normal = (tempV2^tempV1).Normalize();

	// get D
	this->d = -(this->normal * cast_vec3(points[0]));
}

Plane::Plane(__in const KVector3& normal, __in const KPoint3& point, __in const Color& diffuse,
	__in const Color& specular, __in const float shininess, __in const float reflectance,
	__in const float transmittance, __in const float density)
	: Object(diffuse, specular, shininess, reflectance, transmittance, density),
	normal(normal), point(point)
{
	// get D
	this->d = -(normal * cast_vec3(point));
}

Plane::Plane(__in const Plane& cpy)
	: Object(cpy), normal(cpy.normal), d(cpy.d)
{}

Plane::~Plane()
{}

Plane& Plane::operator =(const Plane& other)
{
	this->ambient = other.ambient;
	this->diffuse = other.diffuse;
	this->specular = other.specular;
	this->shininess = other.shininess;
	this->reflectance = other.reflectance;
	this->transmittance = other.transmittance;
	this->density = other.density;
	this->id = other.id;
	this->normal = other.normal;
	this->point = other.point;
	this->d = other.d;

	return *this;
}

Object* Plane::GetHeapCopy() const
{
	return new Plane(*this);
}

void Plane::GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const
{
	// p(t) = eye + d*t;
	// solve t
	KVector3 eye = cast_vec3(ray.GetPoint());
	KVector3 dir = ray.GetDirection();
	dir = dir.Normalize();
	float t = -(this->normal * eye + this->d) / (this->normal * dir);

	// check whether plane is behind camera
	if (t <= 0.0f)
	{
		is_intersect = false;
		return;
	}
	else
	{
		intersect_point = ray.GetPoint() + dir * t;
		is_intersect = true;
	}
}

void Plane::GetNormal(__in const KPoint3& point, __out KVector3& normal) const
{
	normal = this->normal;
}

int Plane::GetType() const
{
	return PLANE_TYPE;
}

KPoint3 Plane::GetPosition() const
{
	return this->point;
}

#endif