#ifndef _POINTLIGHT_H_
#define _POINTLIGHT_H_

#include "KMath.h"
#include "Color.h"

class PointLight {
public:
	// Constructors
	PointLight(__in const KPoint3& pos, __in const Color& color);
	PointLight(__in const PointLight& cpy);
	// Destructors
	virtual ~PointLight();

private:
	// Members
	KPoint3 position;
	Color color;
	// Static Members
	static const float distance_value[3];

public:
	// Methods
	virtual PointLight* GetHeapCopy() const;
	const KPoint3& GetPosition() const;
	void GetColor(__out Color& color) const;
	float GetDistanceTerm(__in float distance) const;
};

// Static Member Initialize
const float PointLight::distance_value[3] = { 0.0f, 1.0f, 0.5f };

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
		1 / (PointLight::distance_value[0] +
			PointLight::distance_value[1] * distance +
			PointLight::distance_value[2] * distance * distance);

	return temp;
}

#endif