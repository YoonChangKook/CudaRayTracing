#ifndef _RAY_H_
#define _RAY_H_

#include "KMath.h"

class Ray
{
public:
	// Constructors
	__host__ __device__ Ray(__in const KPoint3& point, __in const KVector3& direction);
	__host__ __device__ Ray(__in const Ray& cpy);
	// Destructors
	__host__ __device__ virtual ~Ray();

private:
	// Members
	KPoint3 point;
	KVector3 direction;
	
public:
	// Methods
	__host__ __device__ const KPoint3& GetPoint() const;
	__host__ __device__ const KVector3& GetDirection() const;
};

Ray::Ray(const KPoint3& point, const KVector3& direction)
	: point(point), direction(direction)
{
	this->direction.Normalize();
}

Ray::Ray(__in const Ray& cpy)
	: point(cpy.point), direction(cpy.direction)
{}

Ray::~Ray()
{}

const KPoint3& Ray::GetPoint() const
{
	return point;
}

const KVector3& Ray::GetDirection() const
{
	return direction;
}

#endif _RAY_H_