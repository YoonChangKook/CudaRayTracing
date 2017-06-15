#ifndef _OBJECT_H_
#define _OBJECT_H_

#include "KMath.h"
#include "Ray.h"
#include "Color.h"
#include "PointLight.h"

#define AMBIENT_RATIO	0.0784f

class Object
{
public:
	// Constructors
	__host__ __device__ Object(__in const Color& diffuse, __in const Color& specular,
			__in const float shininess, __in const float reflectance,
			__in const float transmittance, __in const float density);
	__host__ __device__ Object(__in const Object& cpy);
	// Destructors
	__host__ __device__ virtual ~Object();

protected:
	// Static Members
	//const float ambient_ratio = 20 / 255.0f;
	// Members
	Color ambient;
	Color diffuse;
	Color specular;
	float shininess;
	float reflectance;
	float transmittance;
	float density;
	int id;

public:
	// Methods
	__host__ __device__ Color GetDiffuse() const;
	__host__ __device__ Color GetSpecular() const;
	__host__ __device__ float GetShininess() const;
	__host__ __device__ float GetReflectance() const;
	__host__ __device__ float GetTransmittance() const;
	__host__ __device__ float GetDensity() const;
	__host__ __device__ void SetID(int id);
	__host__ __device__ int GetID() const;
	__host__ __device__ Object& operator =(const Object& other);
	// Abstract Method
	__host__ __device__ virtual Object* GetHeapCopy() const = 0;
	__host__ __device__ virtual void GetIntersectionPoint(__in const Ray& ray, __out KPoint3& intersect_point, __out bool& is_intersect) const = 0;
	__host__ __device__ virtual void GetNormal(__in const KPoint3& point, __out KVector3& normal) const = 0;
	__host__ __device__ virtual int GetType() const = 0;
	__host__ __device__ virtual KPoint3 GetPosition() const = 0;
	// Virtual Method
	__host__ __device__ virtual void Local_Illumination(__in const KPoint3& point, __in const KVector3& normal,
									__in const Ray& ray, __in const PointLight& light, 
									__out Color& color) const;
};

// Static Member Initialize
//const float Object::ambient_ratio = 20 / 255.0f;

Object::Object(__in const Color& diffuse, __in const Color& specular,
	__in const float shininess, __in const float reflectance,
	__in const float transmittance, __in const float density)
	: ambient(diffuse * AMBIENT_RATIO), diffuse(diffuse), specular(specular), shininess(shininess), 
	reflectance(reflectance), transmittance(transmittance), density(density), id(-1)
{}

Object::Object(__in const Object& cpy)
	: ambient(cpy.ambient), diffuse(cpy.diffuse), specular(cpy.specular), shininess(cpy.shininess),
	reflectance(cpy.reflectance), transmittance(cpy.transmittance), density(cpy.density)
{}

Object::~Object()
{}

Color Object::GetDiffuse() const
{
	return this->diffuse;
}

Color Object::GetSpecular() const
{
	return this->specular;
}

float Object::GetShininess() const
{
	return this->shininess;
}

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

void Object::SetID(int id)
{
	this->id = id;
}

int Object::GetID() const
{
	return this->id;
}

Object& Object::operator =(const Object& other)
{
	this->ambient = other.ambient;
	this->diffuse = other.diffuse;
	this->specular = other.specular;
	this->shininess = other.shininess;
	this->reflectance = other.reflectance;
	this->transmittance = other.transmittance;
	this->density = other.density;
	this->id = other.id;

	return *this;
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

	float diffuse_term = N * L;
	if (diffuse_term < 0.0f)
		diffuse_term = 0.0f;
	float specular_term = powf(N * H, shininess);
	if (specular_term < 0.0f)
		specular_term = 0.0f;
	float distance = dist(point, light.GetPosition());
	float distance_term = light.GetDistanceTerm(distance) * 100;

	// Calculate phong illumination for each color
	for (int i = 0; i < 3; i++)
	{
		//color[i] = this->diffuse[i];
		color[i] = (unsigned char)MIN(255.0f, (distance_term * l_color[i]) *
			(this->ambient[i] / 255.0f +
				diffuse_term * this->diffuse[i] / 255.0f +
				specular_term * this->specular[i] / 255.0f));
	}
}

#endif