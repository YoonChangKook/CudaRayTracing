#pragma once
#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "gmath.h"

class Camera 
{
public:
	// Constructors
	Camera();
	Camera(__in const GPoint3& eye, __in const GVector3& up, __in const GVector3& look, double fovx);
	Camera(__in const Camera& cpy);
	// Destructors
	virtual ~Camera();

private:
	// Members
	GPoint3 eye;
	GVector3 up;
	GVector3 look;
	double fovx;
	GVector3 screenU;
	GVector3 screenV;

public:
	// Methods
	GVector3 GetScreenU() const;
	GVector3 GetScreenV() const;
	GVector3 GetScreenO(__in int width, __in int height) const;
	const GPoint3& GetEyePosition() const;
};

#endif