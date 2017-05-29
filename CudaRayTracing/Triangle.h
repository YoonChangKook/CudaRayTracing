#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "Object.h"

class Triangle : public Object
{
public:
	// Constructors
	Triangle(__in const GPoint3 points[3], __in const Color& diffuse,
			__in const Color& specular, __in const float shininess,
			__in const float reflectance, __in const float transmittance, __in const float density);
	Triangle(__in const Triangle& cpy);
	// Destructors
	virtual ~Triangle();

private:
	// Members
	GPoint3 points[3];

public:
	// Methods
	virtual Object* GetHeapCopy() const;
	virtual void GetIntersectionPoint(__in const Ray& ray, __out GPoint3& intersect_point, __out bool& is_intersect) const;
	virtual void GetNormal(__in const GPoint3& point, __out GVector3& normal) const;
	virtual void GetNormal(__out GVector3& normal) const;	// overload
};

#endif