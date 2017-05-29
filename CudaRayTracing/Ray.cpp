#include "Ray.h"

Ray::Ray(const GPoint3& point, const GVector3& direction)
	: point(point), direction(direction)
{
	this->direction.Normalize();
}

Ray::Ray(__in const Ray& cpy)
	: point(cpy.point), direction(cpy.direction)
{}

Ray::~Ray()
{}

const GPoint3& Ray::GetPoint() const
{
	return point;
}

const GVector3& Ray::GetDirection() const
{
	return direction;
}