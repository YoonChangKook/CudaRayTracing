#pragma once

#ifndef _PLANE_H_
#define _PLANE_H_

#include "Object.h"

class Plane : public Object
{
public:
	// Constructors
	Plane(__in const GPoint3 points[3], __in const Color& diffuse, __in const Color& specular, 
		__in const float shininess,	__in const float reflectance, __in const float transmittance, 
		__in const float density);
	Plane(__in const GVector3& normal, __in const GPoint3& point, __in const Color& diffuse,
		__in const Color& specular, __in const float shininess, __in const float reflectance, 
		__in const float transmittance, __in const float density);
	Plane(__in const Plane& cpy);

	// Destructors
	virtual ~Plane();

private:
	// Members
	GVector3 normal;
	double d;

public:
	// Methods
	virtual Object* GetHeapCopy() const;
	virtual void GetIntersectionPoint(__in const Ray& ray, __out GPoint3& intersect_point, 
									__out bool& is_intersect) const;
	virtual void GetNormal(__in const GPoint3& point, __out GVector3& normal) const;
};

#endif