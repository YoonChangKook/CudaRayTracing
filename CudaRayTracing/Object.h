#pragma once

#ifndef _OBJECT_H_
#define _OBJECT_H_

#include "gmath.h"
#include "Ray.h"
#include "Color.h"
#include "PointLight.h"

class Object
{
public:
	// Constructors
	Object(__in const Color& diffuse, __in const Color& specular,
			__in const float shininess, __in const float reflectance,
			__in const float transmittance, __in const float density);
	Object(__in const Object& cpy);
	// Destructors
	virtual ~Object();

private:
	// Static Members
	static const float ambient_ratio;
	// Members
	Color ambient;
	Color diffuse;
	Color specular;
	float shininess;
	float reflectance;
	float transmittance;
	float density;

public:
	// Methods
	float GetReflectance() const;
	float GetTransmittance() const;
	float GetDensity() const;
	virtual Object* GetHeapCopy() const = 0;
	// Abstract Method
	virtual void GetIntersectionPoint(__in const Ray& ray, __out GPoint3& intersect_point, __out bool& is_intersect) const = 0;
	virtual void GetNormal(__in const GPoint3& point, __out GVector3& normal) const = 0;
	// Virtual Method
	virtual void Local_Illumination(__in const GPoint3& point, __in const GVector3& normal,
									__in const Ray& ray, __in const PointLight& light, 
									__out Color& color) const;
};

#endif