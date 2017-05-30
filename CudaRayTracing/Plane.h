#ifndef _PLANE_H_
#define _PLANE_H_

#include "Object.h"

class Plane : public Object
{
public:
	// Constructors
	Plane(__in const KPoint3 points[3], __in const Color& diffuse, __in const Color& specular, 
		__in const float shininess,	__in const float reflectance, __in const float transmittance, 
		__in const float density);
	Plane(__in const KVector3& normal, __in const KPoint3& point, __in const Color& diffuse,
		__in const Color& specular, __in const float shininess, __in const float reflectance, 
		__in const float transmittance, __in const float density);
	Plane(__in const Plane& cpy);

	// Destructors
	virtual ~Plane();

private:
	// Members
	KVector3 normal;
	double d;

public:
	// Methods
	virtual Object* GetHeapCopy() const;
	virtual void GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, 
									__out bool& is_intersect) const;
	virtual void GetNormal(__in const KPoint3& point, __out KVector3& normal) const;
};

Plane::Plane(__in const KPoint3 points[3], __in const Color& diffuse,
	__in const Color& specular, __in const float shininess, __in const float reflectance,
	__in const float transmittance, __in const float density)
	: Object(diffuse, specular, shininess, reflectance, transmittance, density)
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
	normal(normal)
{
	// get D
	this->d = -(normal * cast_vec3(point));
}

Plane::Plane(__in const Plane& cpy)
	: Object(cpy), normal(cpy.normal), d(cpy.d)
{}

Plane::~Plane()
{}

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
	double t = -(this->normal * eye + this->d) / (this->normal * dir);

	// check whether plane is behind camera
	if (t <= 0)
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

#endif