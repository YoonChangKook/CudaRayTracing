#ifndef _RAYTRACER_H_
#define _RAYTRACER_H_

#include <string>
#include <iostream>
#include <unordered_map>
#include "KMath.h"
#include "Ray.h"
#include "Camera.h"
#include "Color.h"
#include "Object.h"

class RayTracer
{
public:
	// Constructors
	RayTracer();
	// Destructors
	virtual ~RayTracer();

private:
	const int recursive_num = 5;
	// Members
	Color** pixels;
	int width;
	int height;
	Camera camera;
	std::unordered_map<int, PointLight*> lights;
	//std::hash_map<int, PointLight*> lights;
	int light_id_counter;
	//std::hash_map<int, Object*> objects;
	std::unordered_map<int, Object*> objects;
	int object_id_counter;
	// Private Methods
	/*
	*	trace with ray and get intersection point, and recursive trace if need.
	*	recur_num is current recursive num, and pass_obj_id is the last object id crashed with ray.
	*	if pass_obj_id is -1, nothing crashed before trace call.
	*/
	void Trace(__in const Ray& ray, __in const int recur_num, 
			__in const int pass_obj_id, __out Color& color);
	/*
	*	called in Trace function.
	*	calculate color in point.
	*	intersected_obj_id is 
	*/
	void Shade(__in const KPoint3& point, __in const KVector3& normal,
			__in const int intersected_obj_id, __in const Ray& ray,
			__in const int recur_num, __out Color& color);
	/*
	*	find nearest intersection point from ray's origin
	*/
	void Closest_intersection(__out KPoint3& out_point, __out KVector3& out_normal,
							__out int& intersected_obj_id, __in const Ray& ray, 
							__in const int pass_obj_id);
	bool CreateImage(unsigned char* data, const char *outFileName);

public:
	// Methods
	void RayTrace(__in const char* const output_filename);
	void SetImageResolution(__in int width, __in int height);
	void SetCamera(__in const Camera& camera);
	int AddObject(__in const Object& obj);			// return id
	int AddLight(__in const PointLight& light);		// return id
	void DeleteObject(__in int id);
	void DeleteLight(__in int id);
	int GetObjectCount() const;
	int GetLightCount() const;
	void GetAllObjectIDs(__out int ids[]) const;
	void GetAllLightIDs(__out int ids[]) const;

};

RayTracer::RayTracer()
	: pixels(), width(0), height(0), camera(), light_id_counter(0), object_id_counter(0)
{}

RayTracer::~RayTracer()
{
	for (int i = 0; i < height; i++)
		delete[] pixels[i];
	delete[] pixels;

	for (std::unordered_map<int, Object*>::iterator iter = objects.begin();
		iter != objects.end();
		++iter)
		delete iter->second;

	for (std::unordered_map<int, PointLight*>::iterator iter = lights.begin();
		iter != lights.end();
		++iter)
		delete iter->second;
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

	for (std::unordered_map<int, PointLight*>::iterator iter = lights.begin();
		iter != lights.end();
		++iter)
	{
		int shade_inter_obj_id = -1;
		KPoint3 shade_inter_point;
		KVector3 shade_normal;
		Ray shade_ray(point, iter->second->GetPosition() - point);
		Closest_intersection(shade_inter_point, shade_normal, shade_inter_obj_id,
			shade_ray, intersected_obj_id);

		// check whether this point is shadow
		if (shade_inter_obj_id < 0 && recur_num > 0)
		{
			Color temp;
			this->objects[intersected_obj_id]->
				Local_Illumination(point, normal, ray, *(iter->second), temp);

			total += temp;
		}
		else if (recur_num > 0)
		{
			KVector3 L1 = (iter->second->GetPosition() - shade_inter_point);
			KVector3 L2 = (iter->second->GetPosition() - point);
			if (L1 * L2 < 0)
			{
				Color temp;
				this->objects[intersected_obj_id]->
					Local_Illumination(point, normal, ray, *(iter->second), temp);

				total += temp;
			}
		}
	}

	// check whether this object is reflective
	if (this->objects[intersected_obj_id]->GetReflectance() > 0 && recur_num > 0)
	{
		Ray temp_ray(point,
			2 * normal * (normal * (-ray.GetDirection())) + ray.GetDirection());
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
		KVector3 T = (1 / n) * ray.GetDirection() - (cost - (1 / n)*cosi) * normal;

		Ray temp_ray(point, T);
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
	for (std::unordered_map<int, Object*>::iterator iter = objects.begin();
		iter != objects.end();
		++iter)
	{
		bool temp;
		iter->second->GetIntersectionPoint(ray, temp_point, temp);

		// check whether ray crash with object or not.
		if (temp == true)
		{
			// check whether crash object is oneself
			if (pass_obj_id == iter->first)
				continue;
			// if intersectedObject is null, allocate it
			else if (intersected_obj_id < 0)
			{
				intersected_obj_id = iter->first;
				out_point = temp_point;
			}
			// if crash object exist
			else if (dist(ray.GetPoint(), temp_point) <
				dist(ray.GetPoint(), out_point))
			{
				intersected_obj_id = iter->first;
				out_point = temp_point;
			}

			is_intersected = true;
		}
	}

	if (is_intersected == true)
		this->objects[intersected_obj_id]->GetNormal(out_point, out_normal);
}

void RayTracer::RayTrace(__in const char* const output_filename)
{
	KVector3 o = this->camera.GetScreenO(width, height);

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// caculate ray
			KPoint3 pos = this->camera.GetEyePosition();
			KVector3 dir = o +
				i * this->camera.GetScreenU() +
				j * this->camera.GetScreenV();
			dir = dir.Normalize();
			Ray ray(pos, dir);
			Color pixel;

			// start ray tracing about pixel(i, j)
			Trace(ray, RayTracer::recursive_num, -1, pixel);

			this->pixels[i][j] = pixel;
		}

		printf("\r");
		std::cout << (i / (double)width) * 100 << "%";
	}

	unsigned char* tempPixels;

	tempPixels = new unsigned char[width * height * 3];

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				tempPixels[i * (width * 3) + j * 3 + k] = pixels[j][i][k];
			}
		}
	}

	// save image as ppm file
	CreateImage(tempPixels, output_filename);

	// delete tempPixels
	delete[] tempPixels;
}

bool RayTracer::CreateImage(unsigned char* data, const char *outFileName)
{
	// save image as ppm file
	FILE *fp = NULL;
	if (fopen_s(&fp, outFileName, "wb") != 0)
		return false;

	fprintf(fp, "P6\n");
	fprintf(fp, "%d\n", width);
	fprintf(fp, "%d\n", height);
	fprintf(fp, "%d\n", 255);
	fwrite(data, sizeof(unsigned char), width * height * 3, fp);

	fclose(fp);

	return true;
}

void RayTracer::SetImageResolution(__in int width, __in int height)
{
	// delete
	if (this->width != 0 || this->height != 0)
	{
		for (int i = 0; i < this->width; i++)
			delete[] this->pixels[i];
		delete[] this->pixels;
	}

	// new
	this->pixels = new Color*[width];
	for (int i = 0; i < width; i++)
		this->pixels[i] = new Color[height];

	this->width = width;
	this->height = height;
}

void RayTracer::SetCamera(__in const Camera& camera)
{
	this->camera = camera;
}

int RayTracer::AddObject(__in const Object& obj)
{
	this->objects.insert(std::pair<int, Object*>(this->object_id_counter, obj.GetHeapCopy()));
	return this->object_id_counter++;
}

int RayTracer::AddLight(__in const PointLight& light)
{
	this->lights.insert(std::pair<int, PointLight*>(this->light_id_counter, light.GetHeapCopy()));
	return this->light_id_counter++;
}

void RayTracer::DeleteObject(__in int id)
{
	if (this->objects.find(id)->first == id)
	{
		delete this->objects[id];
		this->objects.erase(id);
	}
}

void RayTracer::DeleteLight(__in int id)
{
	if (this->lights.find(id)->first == id)
	{
		delete this->lights[id];
		this->lights.erase(id);
	}
}

int RayTracer::GetObjectCount() const
{
	return this->objects.size();
}

int RayTracer::GetLightCount() const
{
	return this->lights.size();
}

void RayTracer::GetAllObjectIDs(__out int ids[]) const
{
	std::unordered_map<int, Object*>::const_iterator iter;
	int i;

	for (iter = objects.cbegin(), i = 0;
		iter != this->objects.cend();
		++iter, ++i)
		ids[i] = iter->first;
}

void RayTracer::GetAllLightIDs(__out int ids[]) const
{
	std::unordered_map<int, PointLight*>::const_iterator iter;
	int i;

	for (iter = lights.cbegin(), i = 0;
		iter != this->lights.cend();
		++iter, ++i)
		ids[i] = iter->first;
}

#endif