#include "Ray.h"
#include "math_functions.h"

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