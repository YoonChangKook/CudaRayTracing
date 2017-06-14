#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "KMath.h"

class Camera 
{
public:
	// Constructors
	__host__ __device__ Camera();
	__host__ __device__ Camera(__in const KPoint3& eye, __in const KVector3& up, __in const KVector3& look, float fovx);
	__host__ __device__ Camera(__in const Camera& cpy);
	// Destructors
	__host__ __device__ ~Camera();

private:
	// Members
	KPoint3 eye;
	KVector3 up;
	KVector3 look;
	float fovx;
	KVector3 screenU;
	KVector3 screenV;

public:
	// Methods
	__host__ __device__ KVector3 GetScreenU() const;
	__host__ __device__ KVector3 GetScreenV() const;
	__host__ __device__ KVector3 GetScreenO(__in int width, __in int height) const;
	__host__ __device__ const KPoint3& GetEyePosition() const;
};

Camera::Camera()
	: eye(0.0f, 0.0f, 0.0f), up(0.0f, 1.0f, 0.0f), look(0.0f, 0.0f, -1.0f), fovx(120.0f)
{
	this->screenU = (look^up).Normalize();
	this->screenV = (look^screenU).Normalize();
}

Camera::Camera(__in const KPoint3& eye, __in const KVector3& up, __in const KVector3& look, float fovx)
	: eye(eye), up(up), look(look), fovx(fovx)
{
	this->screenU = (look^up).Normalize();
	this->screenV = (look^screenU).Normalize();
}

Camera::Camera(__in const Camera& cpy)
	: eye(cpy.eye), up(cpy.up), look(cpy.look), fovx(cpy.fovx)
{
	this->screenU = (look^up).Normalize();
	this->screenV = (look^screenU).Normalize();
}

Camera::~Camera()
{}

KVector3 Camera::GetScreenU() const
{
	return this->screenU;
}

KVector3 Camera::GetScreenV() const
{
	return this->screenV;
}

KVector3 Camera::GetScreenO(__in int width, __in int height) const
{
	return (look.NormalizeCopy() * width / (2.0f * tan(fovx / 2.0f))
		- (width / 2) * screenU - (height / 2) * screenV);
}

const KPoint3& Camera::GetEyePosition() const
{
	return this->eye;
}

#endif