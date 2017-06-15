#ifndef _SPHERE_H_
#define _SPHERE_H_

#include "Object.h"

#define SPHERE_TYPE 1

class Sphere : public Object
{
public:
	// Constructors
	__host__ __device__ Sphere(__in const KPoint3& pos, float r, __in const Color& diffuse,
			__in const Color& specular, __in const float shininess,
			__in const float reflectance, __in const float transmittance, __in const float density);
	__host__ __device__ Sphere(__in const Sphere& cpy);
	// Destructors
	__host__ __device__ virtual ~Sphere();

private:
	// Members
	KPoint3 position;
	float r;

public:
	// Methods
	__host__ __device__ float GetR() const;
	__host__ __device__ Sphere& operator =(const Sphere& other);
	__host__ __device__ virtual Object* GetHeapCopy() const;
	__host__ __device__ virtual void GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const;
	__host__ __device__ virtual void GetNormal(__in const KPoint3& point, __out KVector3& normal) const;
	__host__ __device__ virtual int GetType() const;
	__host__ __device__ virtual KPoint3 GetPosition() const;
};

// Constructors
Sphere::Sphere(__in const KPoint3& pos, float r, __in const Color& diffuse,
	__in const Color& specular, __in const float shininess,
	__in const float reflectance, __in const float transmittance, __in const float density)
	: Object(diffuse, specular, shininess, reflectance, transmittance, density),
	position(pos), r(r)
{}

Sphere::Sphere(__in const Sphere& cpy)
	: Object(cpy), position(cpy.position), r(cpy.r)
{}

// Destructors
Sphere::~Sphere()
{}

// Methods
float Sphere::GetR() const
{
	return this->r;
}

Sphere& Sphere::operator =(const Sphere& other)
{
	this->ambient = other.ambient;
	this->diffuse = other.diffuse;
	this->specular = other.specular;
	this->shininess = other.shininess;
	this->reflectance = other.reflectance;
	this->transmittance = other.transmittance;
	this->density = other.density;
	this->id = other.id;
	this->position = other.position;
	this->r = other.r;

	return *this;
}

Object* Sphere::GetHeapCopy() const
{
	return new Sphere(*this);
}

void Sphere::GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const
{
	KVector3 c = position - ray.GetPoint();
	float tc = c * ray.GetDirection();

	// Check whether the sphere is behind the camera
	if (tc <= 0)
	{
		is_intersect = false;
		return;
	}

	float bSqr = MAX(0.0f, c * c - tc * tc);
	float b = sqrtf(bSqr);

	// Check whether the ray doesn't crash with the sphere
	if (b > r)
	{
		is_intersect = false;
		return;
	}

	float tuSqr = MAX(0.0f, r * r - b * b);
	float tu = sqrtf(tuSqr);
	float t;

	if (dist(ray.GetPoint(), position) > r)
		t = tc - tu;
	else
		t = tc + tu;

	// Set output
	is_intersect = true;
	intersect_point = ray.GetPoint() + t * ray.GetDirection();
}

void Sphere::GetNormal(__in const KPoint3& point, __out KVector3& normal) const
{
	normal = (point - this->position).Normalize();
}

int Sphere::GetType() const
{
	return SPHERE_TYPE;
}

KPoint3 Sphere::GetPosition() const
{
	return this->position;
}

#endif