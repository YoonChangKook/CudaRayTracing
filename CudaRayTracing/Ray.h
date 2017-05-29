#ifndef _RAY_H_
#define _RAY_H_

#include "gmath.h"

class Ray
{
public:
	// Constructors
	Ray(__in const GPoint3& point, __in const GVector3& direction);
	Ray(__in const Ray& cpy);
	// Destructors
	virtual ~Ray();

private:
	// Members
	GPoint3 point;
	GVector3 direction;
	
public:
	// Methods
	const GPoint3& GetPoint() const;
	const GVector3& GetDirection() const;
};

#endif _RAY_H_