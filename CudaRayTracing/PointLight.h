#pragma once
#ifndef _POINTLIGHT_H_
#define _POINTLIGHT_H_

#include "gmath.h"
#include "Color.h"

class PointLight {
public:
	// Constructors
	PointLight(__in const GPoint3& pos, __in const Color& color);
	PointLight(__in const PointLight& cpy);
	// Destructors
	virtual ~PointLight();

private:
	// Members
	GPoint3 position;
	Color color;
	// Static Members
	static const float distance_value[3];

public:
	// Methods
	virtual PointLight* GetHeapCopy() const;
	const GPoint3& GetPosition() const;
	void GetColor(__out Color& color) const;
	float GetDistanceTerm(__in float distance) const;
};

#endif