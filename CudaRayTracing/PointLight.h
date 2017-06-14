#ifndef _POINTLIGHT_H_
#define _POINTLIGHT_H_

#include "KMath.h"
#include "Color.h"

#define DIST_VAL0		0.0f
#define DIST_VAL1		0.5f
#define DIST_VAL2		0.25f

#define POINT_LIGHT_TYPE 10

class PointLight {
public:
	// Constructors
	__host__ __device__ PointLight(__in const KPoint3& pos, __in const Color& color);
	__host__ __device__ PointLight(__in const PointLight& cpy);
	// Destructors
	__host__ __device__ virtual ~PointLight();

private:
	// Members
	KPoint3 position;
	Color color;
	// Static Members
	//const float distance_value[3] = { 0.0f, 0.5f, 0.25f };

public:
	// Methods
	__host__ __device__ virtual PointLight* GetHeapCopy() const;
	__host__ __device__ const KPoint3& GetPosition() const;
	__host__ __device__ void GetColor(__out Color& color) const;
	__host__ __device__ float GetDistanceTerm(__in float distance) const;
	__host__ __device__ int GetType() const;
};

// Static Member Initialize
//const float PointLight::distance_value[3] = { 0.0f, 0.5f, 0.25f };

PointLight::PointLight(__in const KPoint3& pos, __in const Color& color)
	: position(pos), color(color)
{}

PointLight::PointLight(__in const PointLight& cpy)
	: position(cpy.position), color(cpy.color)
{}

PointLight::~PointLight()
{}

PointLight* PointLight::GetHeapCopy() const
{
	return new PointLight(*this);
}

const KPoint3& PointLight::GetPosition() const
{
	return this->position;
}

void PointLight::GetColor(__out Color& color) const
{
	color = this->color;
}

float PointLight::GetDistanceTerm(__in float distance) const
{
	float temp =
		1 / (DIST_VAL0 +
			DIST_VAL1 * distance +
			DIST_VAL2 * distance * distance);

	return temp;
}

int PointLight::GetType() const
{
	return POINT_LIGHT_TYPE;
}

#endif