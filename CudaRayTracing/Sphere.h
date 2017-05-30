#ifndef _SPHERE_H_
#define _SPHERE_H_

#include "Object.h"

class Sphere : public Object
{
public:
	// Constructors
	Sphere(__in const KPoint3& pos, float r, __in const Color& diffuse,
			__in const Color& specular, __in const float shininess,
			__in const float reflectance, __in const float transmittance, __in const float density);
	Sphere(__in const Sphere& cpy);
	// Destructors
	virtual ~Sphere();

private:
	// Members
	KPoint3 position;
	float r;

public:
	// Methods
	virtual Object* GetHeapCopy() const;
	virtual void GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const;
	virtual void GetNormal(__in const KPoint3& point, __out KVector3& normal) const;
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
Object* Sphere::GetHeapCopy() const
{
	return new Sphere(*this);
}

void Sphere::GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const
{
	KVector3 c = position - ray.GetPoint();
	double tc = c * ray.GetDirection();

	// Check whether the sphere is behind the camera
	if (tc <= 0)
	{
		is_intersect = false;
		return;
	}

	double b = sqrt(c * c - tc * tc);

	// Check whether the ray doesn't crash with the sphere
	if (b > r)
	{
		is_intersect = false;
		return;
	}

	double tu = sqrt(r * r - b * b);
	double t;

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

#endif