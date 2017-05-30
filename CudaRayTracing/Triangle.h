#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "Object.h"

class Triangle : public Object
{
public:
	// Constructors
	Triangle(__in const KPoint3 points[3], __in const Color& diffuse,
			__in const Color& specular, __in const float shininess,
			__in const float reflectance, __in const float transmittance, __in const float density);
	Triangle(__in const Triangle& cpy);
	// Destructors
	virtual ~Triangle();

private:
	// Members
	KPoint3 points[3];

public:
	// Methods
	virtual Object* GetHeapCopy() const;
	virtual void GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const;
	virtual void GetNormal(__in const KPoint3& point, __out KVector3& normal) const;
	virtual void GetNormal(__out KVector3& normal) const;	// overload
};

Triangle::Triangle(__in const KPoint3 points[3], __in const Color& diffuse,
	__in const Color& specular, __in const float shininess,
	__in const float reflectance, __in const float transmittance, __in const float density)
	: Object(diffuse, specular, shininess, reflectance, transmittance, density),
	points{ points[0], points[1], points[2] }
{}

Triangle::Triangle(__in const Triangle& cpy)
	: Object(cpy), points{ cpy.points[0], cpy.points[1], cpy.points[2] }
{}

Triangle::~Triangle()
{}

Object* Triangle::GetHeapCopy() const
{
	return new Triangle(*this);
}

void Triangle::GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const
{
	// get plane normal
	KVector3 n;
	GetNormal(n);

	// p(t) = eye + d*t;
	// solve t
	KVector3 eye = cast_vec3(ray.GetPoint());
	KVector3 dir = ray.GetDirection();
	dir = dir.Normalize();
	double t = -(n * eye - (n * cast_vec3(this->points[0]))) / (n * dir);

	// check whether plane is behind camera
	if (t <= 0)
	{
		is_intersect = false;
		return;
	}

	KPoint3 plane_point = KPoint3(ray.GetPoint() + dir * t);

	// check whether point is inside triangle
	// Compute vectors        
	KVector3 v0 = this->points[2] - this->points[0];
	KVector3 v1 = this->points[1] - this->points[1];
	KVector3 v2 = plane_point - this->points[0];

	// Compute dot products
	double dot00 = v0 * v0;
	double dot01 = v0 * v1;
	double dot02 = v0 * v2;
	double dot11 = v1 * v1;
	double dot12 = v1 * v2;

	// Compute barycentric coordinates
	double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is in triangle
	if ((u >= 0) && (v >= 0) && (u + v < 1))
	{
		intersect_point = plane_point;
		is_intersect = true;
	}
	else
		is_intersect = false;
}

void Triangle::GetNormal(__in const KPoint3& point, __out KVector3& normal) const
{
	// same normals regardless of point
	KVector3 tempV1 = (this->points[0] - this->points[1]).Normalize();
	KVector3 tempV2 = (this->points[2] - this->points[1]).Normalize();

	normal = (tempV2^tempV1).Normalize();
}

void Triangle::GetNormal(__out KVector3& normal) const
{
	// same normals regardless of point
	KVector3 tempV1 = (this->points[0] - this->points[1]).Normalize();
	KVector3 tempV2 = (this->points[2] - this->points[1]).Normalize();

	normal = (tempV2^tempV1).Normalize();
}

#endif