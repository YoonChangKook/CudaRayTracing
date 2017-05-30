#include "Camera.h"
#include "math_functions.h"

Camera::Camera()
	: eye(0, 0, 0), up(0, 1, 0), look(0, 0, -1), fovx(120.0)
{
	this->screenU = (look^up).Normalize();
	this->screenV = (look^screenU).Normalize();
}

Camera::Camera(__in const KPoint3& eye, __in const KVector3& up, __in const KVector3& look, double fovx)
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
	return (look.NormalizeCopy() * width / (2 * tan(fovx / 2))
			- (width / 2) * screenU - (height / 2) * screenV);
}

const KPoint3& Camera::GetEyePosition() const
{
	return this->eye;
}