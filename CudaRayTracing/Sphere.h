#pragma once

#ifndef _SPHERE_H_
#define _SPHERE_H_

#include "Object.h"

class Sphere : public Object
{
public:
	// Constructors
	Sphere(__in const GPoint3& pos, float r, __in const Color& diffuse,
			__in const Color& specular, __in const float shininess,
			__in const float reflectance, __in const float transmittance, __in const float density);
	Sphere(__in const Sphere& cpy);
	// Destructors
	virtual ~Sphere();

private:
	// Members
	GPoint3 position;
	float r;

public:
	// Methods
	virtual Object* GetHeapCopy() const;
	virtual void GetIntersectionPoint(__in const Ray& ray, __out GPoint3& intersect_point, __out bool& is_intersect) const;
	virtual void GetNormal(__in const GPoint3& point, __out GVector3& normal) const;
};

#endif