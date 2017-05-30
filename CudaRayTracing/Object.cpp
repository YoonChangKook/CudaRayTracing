#include "Object.h"
#include "math_functions.h"

// Static Member Initialize
const float Object::ambient_ratio = 20 / 255.0f;

Object::Object(__in const Color& diffuse, __in const Color& specular,
			__in const float shininess, __in const float reflectance,
			__in const float transmittance, __in const float density)
	: ambient(diffuse * ambient_ratio),	diffuse(diffuse), specular(specular),
	shininess(shininess), reflectance(reflectance), transmittance(transmittance), density(density)
{}

Object::Object(__in const Object& cpy)
	: ambient(cpy.ambient), diffuse(cpy.diffuse), specular(cpy.specular), shininess(cpy.shininess), 
	reflectance(cpy.reflectance), transmittance(cpy.transmittance), density(cpy.density)
{}

Object::~Object()
{}

float Object::GetReflectance() const
{
	return this->reflectance;
}

float Object::GetTransmittance() const
{
	return this->transmittance;
}

float Object::GetDensity() const
{
	return this->density;
}

void Object::Local_Illumination(__in const KPoint3& point, __in const KVector3& normal,
								__in const Ray& ray, __in const PointLight& light,
								__out Color& color) const
{
	Color l_color;
	light.GetColor(l_color);

	// Calculate term
	KVector3 L = light.GetPosition() - point;
	L.Normalize();
	KVector3 N = normal;
	N.Normalize();
	KVector3 H = (L + (-ray.GetDirection()).Normalize());
	H.Normalize();

	double diffuse_term = N * L;
	if (diffuse_term < 0)
		diffuse_term = 0;
	double specular_term = pow(N * H, (double)shininess);
	if (specular_term < 0)
		specular_term = 0;
	double distance = dist(point, light.GetPosition());
	double distance_term = light.GetDistanceTerm(distance) * 100;

	// Calculate phong illumination for each color
	for (int i = 0; i < 3; i++)
		color[i] = (unsigned char)MIN(255.0, (distance_term * l_color[i]) * 
												(this->ambient[i] / 255.0 +
												diffuse_term * this->diffuse[i] / 255.0 +
												specular_term * this->specular[i] / 255.0));
}