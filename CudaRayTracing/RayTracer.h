#ifndef _RAYTRACER_H_
#define _RAYTRACER_H_

#include <string>
#include <iostream>
//#include <unordered_map>
#include "KMath.h"
#include "Ray.h"
#include "Camera.h"
#include "Color.h"
#include "Object.h"
#include "Sphere.h"
#include "Plane.h"
#include "Triangle.h"
#include "CudaVector.h"

#define RECURSIVE_NUM			5
#define CUDA_FUNC				__host__ __device__
#define CUDA_GLOBAL				__global__

class RayTracer
{
public:
	// Constructors
	__device__ RayTracer();
	// Destructors
	__device__ ~RayTracer();

private:
	// Members
	Color** pixels;
	int width;
	int height;
	Camera camera;
	Cuda::Vector<PointLight*> lights;
	int light_id_counter;
	int light_count;
	Cuda::Vector<Object*> objects;
	int object_id_counter;
	int object_count;
	// Private Methods
	/*
	*	trace with ray and get intersection point, and recursive trace if need.
	*	recur_num is current recursive num, and pass_obj_id is the last object id crashed with ray.
	*	if pass_obj_id is -1, nothing crashed before trace call.
	*/
	__device__ void Trace(__in const Ray& ray, __in const int recur_num,
			__in const int pass_obj_id, __out Color& color);
	/*
	*	called in Trace function.
	*	calculate color in point.
	*	intersected_obj_id is 
	*/
	__device__ void Shade(__in const KPoint3& point, __in const KVector3& normal,
			__in const int intersected_obj_id, __in const Ray& ray,
			__in const int recur_num, __out Color& color);
	/*
	*	find nearest intersection point from ray's origin
	*/
	__device__ void Closest_intersection(__out KPoint3& out_point, __out KVector3& out_normal,
							__out int& intersected_obj_id, __in const Ray& ray, 
							__in const int pass_obj_id);
	//__host__ bool CreateImage(const char *outFileName);

	__device__ void AddObject(__in const Object& obj, __out int* id);			// return id
	__device__ void AddLight(__in const PointLight& light, __out int* id);		// return id

public:
	// Methods
	__global__ friend void RayTrace(__in RayTracer* const ray_tracer, unsigned char* buffer);
	// <<<1, 1>>> global functions
	__global__ friend void CopyColors(__in RayTracer* const ray_tracer, __out unsigned char* buffer);
	__global__ friend void SetImageResolution(__in RayTracer* const ray_tracer, __in int width, __in int height);
	__global__ friend void SetCamera(__in RayTracer* const ray_tracer, __in const KPoint3* const pos,
									__in const KVector3* const up, __in const KVector3* const look, float* fovx);
	
	__global__ friend void AddSphere(__in RayTracer* const ray_tracer, __in const KPoint3* const pos, float r, __in const Color* const diffuse,
							__in const Color* const specular, __in const float shininess,
							__in const float reflectance, __in const float transmittance, __in const float density,
							__out int* id);
	__global__ friend void ModifySphere(__in int id, __in RayTracer* const ray_tracer, __in const KPoint3* const pos, float r, 
									__in const Color* const diffuse, __in const Color* const specular, __in const float shininess,
									__in const float reflectance, __in const float transmittance, __in const float density);

	__global__ friend void AddPlane(__in RayTracer* const ray_tracer, __in const KVector3* const normal, __in const KPoint3* const point, __in const Color* const diffuse,
							__in const Color* const specular, __in const float shininess, __in const float reflectance,
							__in const float transmittance, __in const float density,
							__out int* id);
	__global__ friend void ModifyPlane(__in int id, __in RayTracer* const ray_tracer, __in const KVector3* const normal, __in const KPoint3* const point, __in const Color* const diffuse,
							__in const Color* const specular, __in const float shininess, __in const float reflectance,
							__in const float transmittance, __in const float density);

	__global__ friend void AddPointLight(__in RayTracer* const ray_tracer, __in const KPoint3* const pos, __in const Color* const color,
								__out int* id);
	__global__ friend void ModifyPointLight(__in int id, __in RayTracer* const ray_tracer, __in const KPoint3* const pos, __in const Color* const color);
	
	__global__ friend void DeleteObject(__in RayTracer* const ray_tracer, __in int id);
	__global__ friend void DeleteLight(__in RayTracer* const ray_tracer, __in int id);
	__global__ friend void GetObjectCount(__in RayTracer* const ray_tracer, __out int* count);
	__global__ friend void GetLightCount(__in RayTracer* const ray_tracer, __out int* count);
	__global__ friend void GetAllObjectIDs(__in RayTracer* const ray_tracer, __out int ids[]);
	__global__ friend void GetAllLightIDs(__in RayTracer* const ray_tracer, __out int ids[]);

	__global__ friend void GetClickedObj(__in RayTracer* const ray_tracer, __in int screen_x, __in int screen_y, __out int* id);
	__global__ friend void GetObjectPosition(__in RayTracer* const ray_tracer, __in int id, __out KPoint3* pos);
};

RayTracer::RayTracer()
	: pixels(), width(0), height(0), camera(), light_id_counter(0), object_id_counter(0),
	light_count(0), object_count(0), lights(), objects()
{}

RayTracer::~RayTracer()
{
	for (int i = 0; i < height; i++)
		delete[] pixels[i];
	delete[] pixels;

	for (int i = 0; i < this->objects.size(); i++)
	{
		if (this->objects[i] != NULL)
			delete this->objects[i];
	}
	for (int i = 0; i < this->lights.size(); i++)
	{
		if (this->lights[i] != NULL)
			delete this->lights[i];
	}
}

void RayTracer::Trace(__in const Ray& ray, __in const int recur_num,
	__in const int pass_obj_id, __out Color& color)
{
	int intersected_obj_id = -1;
	KPoint3 intersected_point;
	KVector3 normal;

	// find the nearest intersection point
	Closest_intersection(intersected_point, normal, intersected_obj_id,
		ray, pass_obj_id);

	// check whether the ray crash with any object
	if (intersected_obj_id >= 0)
	{
		Shade(intersected_point, normal, intersected_obj_id,
			ray, recur_num, color);
	}
	else
	{
		color = Color(0, 0, 0);
	}
}

void RayTracer::Shade(__in const KPoint3& point, __in const KVector3& normal,
	__in const int intersected_obj_id, __in const Ray& ray,
	__in const int recur_num, __out Color& color)
{
	Color total;

	for(int i = 0; i < this->lights.size(); i++)
	{
		if (this->lights[i] == NULL)
			continue;

		int shade_inter_obj_id = -1;
		KPoint3 shade_inter_point;
		KVector3 shade_normal;
		KVector3 shade_ray_dir = (this->lights[i]->GetPosition() - point).Normalize();
		Ray shade_ray(point + shade_ray_dir * 0.05f, shade_ray_dir);
		Closest_intersection(shade_inter_point, shade_normal, shade_inter_obj_id,
			shade_ray, intersected_obj_id);

		// check whether this point is shadow
		if (shade_inter_obj_id < 0 && recur_num > 0)
		{
			Color temp;
			this->objects[intersected_obj_id]->
				Local_Illumination(point, normal, ray, *this->lights[i], temp);
		
			total += temp;
		}
		else if (recur_num > 0)
		{
			KVector3 L1 = (this->lights[i]->GetPosition() - shade_inter_point);
			KVector3 L2 = (this->lights[i]->GetPosition() - point);
			if (L1 * L2 < 0)
			{
				Color temp;
				this->objects[intersected_obj_id]->
					Local_Illumination(point, normal, ray, *this->lights[i], temp);
		
				total += temp;
			}
		}
	}

	// check whether this object is reflective
	if (this->objects[intersected_obj_id]->GetReflectance() > 0 && recur_num > 0)
	{
		KVector3 temp_ray_dir = (2 * normal * (normal * (-ray.GetDirection())) + ray.GetDirection()).Normalize();
		Ray temp_ray(point + temp_ray_dir * 0.05f, temp_ray_dir);
		Color temp_color;

		Trace(temp_ray, recur_num - 1, intersected_obj_id, temp_color);

		total += temp_color * this->objects[intersected_obj_id]->GetReflectance();
	}

	// check whether this object is transmissive
	if (this->objects[intersected_obj_id]->GetTransmittance() > 0 && recur_num > 0)
	{
		float n;

		// check whether ray 
		if (normal * ray.GetDirection() > 0)
			n = 1 / this->objects[intersected_obj_id]->GetDensity();
		else
			n = this->objects[intersected_obj_id]->GetDensity() / 1;

		float cosi = normal * (-ray.GetDirection());
		float cost = n * (1 - (1 / (n * n))) * sqrt(1 - cosi * cosi);
		KVector3 T = ((1 / n) * ray.GetDirection() - (cost - (1 / n)*cosi) * normal).Normalize();

		Ray temp_ray(point + T * 0.05f, T);
		Color temp_color;

		Trace(temp_ray, recur_num - 1, intersected_obj_id, temp_color);

		total += temp_color * this->objects[intersected_obj_id]->GetTransmittance();
	}

	color = total;
}

void RayTracer::Closest_intersection(__out KPoint3& out_point, __out KVector3& out_normal,
	__out int& intersected_obj_id, __in const Ray& ray,
	__in const int pass_obj_id)
{
	intersected_obj_id = -1;
	KPoint3 temp_point;
	bool is_intersected = false;

	// loop per each object
	for(int i = 0; i < this->objects.size(); i++)
	{
		if (this->objects[i] == NULL)
			continue;

		bool temp;
		this->objects[i]->GetIntersectionPoint(ray, temp_point, temp);

		// check whether ray crash with object or not.
		if (temp == true)
		{
			// if intersectedObject is null, allocate it
			if (intersected_obj_id < 0)
			{
				intersected_obj_id = i;
				out_point = temp_point;
			}
			// if crash object exist
			else if (dist(ray.GetPoint(), temp_point) <
				dist(ray.GetPoint(), out_point))
			{
				intersected_obj_id = i;
				out_point = temp_point;
			}

			is_intersected = true;
		}
	}

	if (is_intersected == true)
		this->objects[intersected_obj_id]->GetNormal(out_point, out_normal);
}

__global__ void RayTrace(__in RayTracer* const ray_tracer, unsigned char* buffer)
{
	KVector3 o = ray_tracer->camera.GetScreenO(ray_tracer->width, ray_tracer->height);
	int i = (blockIdx.x * blockDim.x + threadIdx.x);
	int j = (blockIdx.y * blockDim.y + threadIdx.y);

	KPoint3 pos = ray_tracer->camera.GetEyePosition();
	KVector3 dir = o +
		i * ray_tracer->camera.GetScreenU() +
		j * ray_tracer->camera.GetScreenV();
	dir = dir.Normalize();
	Ray ray(pos, dir);
	Color pixel;

	// start ray tracing about pixel(i, j)
	ray_tracer->Trace(ray, RECURSIVE_NUM, -1, pixel);

	buffer[4 * (ray_tracer->width * (ray_tracer->height - 1 - j) + i) + 0] = pixel[0];
	buffer[4 * (ray_tracer->width * (ray_tracer->height - 1 - j) + i) + 1] = pixel[1];
	buffer[4 * (ray_tracer->width * (ray_tracer->height - 1 - j) + i) + 2] = pixel[2];
	buffer[4 * (ray_tracer->width * (ray_tracer->height - 1 - j) + i) + 3] = 255;
}

__global__ void CopyColors(__in RayTracer* const ray_tracer, __out unsigned char* buffer)
{
	for (int i = 0; i < ray_tracer->height; i++)
		for (int j = 0; j < ray_tracer->width; j++)
			for(int k = 0; k < 3; k++)
				buffer[k + 3 * j + 3 * ray_tracer->width * i] = ray_tracer->pixels[j][i][k];
}

__global__ void SetImageResolution(__in RayTracer* const ray_tracer, __in int width, __in int height)
{
	ray_tracer->width = width;
	ray_tracer->height = height;
}

__global__ void SetCamera(__in RayTracer* const ray_tracer, __in const KPoint3* const pos,
						__in const KVector3* const up, __in const KVector3* const look, float* fovx)
{
	ray_tracer->camera = Camera(*pos, *up, *look, *fovx);
}

void RayTracer::AddObject(__in const Object& obj, __out int* id)
{
	this->objects.push_back(obj.GetHeapCopy());
	this->objects.back()->SetID(this->objects.size() - 1);
	*id = this->objects.back()->GetID();
}

__global__ void AddSphere(__in RayTracer* const ray_tracer, __in const KPoint3* const pos, float r, __in const Color* const diffuse,
						__in const Color* const specular, __in const float shininess,
						__in const float reflectance, __in const float transmittance, __in const float density,
						__out int* id)
{
	Sphere temp(*pos, r, *diffuse, *specular, shininess,
				reflectance, transmittance, density);

	ray_tracer->AddObject(temp, id);
}
__global__ void ModifySphere(__in int id, __in RayTracer* const ray_tracer, __in const KPoint3* const pos, float r,
						__in const Color* const diffuse, __in const Color* const specular, __in const float shininess,
						__in const float reflectance, __in const float transmittance, __in const float density)
{
	if (id >= ray_tracer->objects.size() || id < 0)
		return;
	if (ray_tracer->objects[id] == NULL)
		return;
	if (ray_tracer->objects[id]->GetType() != SPHERE_TYPE)
		return;

	printf("id, type: %d, %d\n", id, ray_tracer->objects[id]->GetType());
	*((Sphere*)ray_tracer->objects[id]) = Sphere(*pos, r, *diffuse, *specular, shininess,
												reflectance, transmittance, density);
}

__global__ void AddPlane(__in RayTracer* const ray_tracer, __in const KVector3* const normal, __in const KPoint3* const point, __in const Color* const diffuse,
						__in const Color* const specular, __in const float shininess, __in const float reflectance,
						__in const float transmittance, __in const float density,
						__out int* id)
{
	Plane temp(*normal, *point, *diffuse, *specular, 
				shininess, reflectance, transmittance, density);

	ray_tracer->AddObject(temp, id);
}
__global__ void ModifyPlane(__in int id, __in RayTracer* const ray_tracer, __in const KVector3* const normal, __in const KPoint3* const point, __in const Color* const diffuse,
	__in const Color* const specular, __in const float shininess, __in const float reflectance,
	__in const float transmittance, __in const float density)
{
	if (id >= ray_tracer->objects.size() || id < 0)
		return;
	if (ray_tracer->objects[id] == NULL)
		return;
	if (ray_tracer->objects[id]->GetType() != PLANE_TYPE)
		return;
	
	*((Plane*)ray_tracer->objects[id]) = Plane(*normal, *point, *diffuse, *specular,
											shininess, reflectance, transmittance, density);
}

void RayTracer::AddLight(__in const PointLight& light, __out int* id)
{
	this->lights.push_back(light.GetHeapCopy());
	this->lights.back()->SetID(this->lights.size() - 1);
	*id = this->lights.back()->GetID();
}

__global__ void AddPointLight(__in RayTracer* const ray_tracer, __in const KPoint3* const pos, __in const Color* const color, __out int* id)
{
	PointLight temp(*pos, *color);

	ray_tracer->AddLight(temp, id);
}
__global__ void ModifyPointLight(__in int id, __in RayTracer* const ray_tracer, __in const KPoint3* const pos, __in const Color* const color)
{
	if (id >= ray_tracer->lights.size() || id < 0)
		return;
	if (ray_tracer->lights[id] == NULL)
		return;
	if (ray_tracer->lights[id]->GetType() != POINT_LIGHT_TYPE)
		return;

	*((PointLight*)ray_tracer->lights[id]) = PointLight(*pos, *color);
}

__global__ void DeleteObject(__in RayTracer* const ray_tracer, __in int id)
{
	if (id >= ray_tracer->objects.size() || id < 0)
		return;

	if (ray_tracer->objects[id] != NULL)
		delete ray_tracer->objects[id];
}

__global__ void DeleteLight(__in RayTracer* const ray_tracer, __in int id)
{
	if (id >= ray_tracer->lights.size() || id < 0)
		return;

	if (ray_tracer->lights[id] != NULL)
		delete ray_tracer->lights[id];
}

__global__ void GetObjectCount(__in RayTracer* const ray_tracer, __out int* count)
{
	*count = ray_tracer->objects.size();
}

__global__ void GetLightCount(__in RayTracer* const ray_tracer, __out int* count)
{
	*count = ray_tracer->lights.size();
}

__global__ void GetAllObjectIDs(__in RayTracer* const ray_tracer, __out int ids[])
{
	for (int i = 0, idx = 0; i < ray_tracer->objects.size(); i++)
		if (ray_tracer->objects[i] != NULL)
			ids[idx++] = i;
}

__global__ void GetAllLightIDs(__in RayTracer* const ray_tracer, __out int ids[])
{
	for (int i = 0, idx = 0; i < ray_tracer->lights.size(); i++)
		if (ray_tracer->lights[i] != NULL)
			ids[idx++] = i;
}

__global__ void GetClickedObj(__in RayTracer* const ray_tracer, __in int screen_x, __in int screen_y, __out int* id)
{
	KVector3 o = ray_tracer->camera.GetScreenO(ray_tracer->width, ray_tracer->height);
	KPoint3 pos = ray_tracer->camera.GetEyePosition();
	KVector3 dir = o +
		screen_x * ray_tracer->camera.GetScreenU() +
		screen_y * ray_tracer->camera.GetScreenV();
	dir = dir.Normalize();
	Ray ray(pos, dir);

	// find the nearest intersection point
	int intersected_obj_id = -1;
	KPoint3 intersected_point;
	KVector3 normal;
	ray_tracer->Closest_intersection(intersected_point, normal, intersected_obj_id,
									ray, -1);

	// if clicked, return id
	if (intersected_obj_id >= 0)
		*id = intersected_obj_id;
}

__global__ void GetObjectPosition(__in RayTracer* const ray_tracer, __in int id, __out KPoint3* pos)
{
	*pos = ray_tracer->objects[id]->GetPosition();
}

#endif