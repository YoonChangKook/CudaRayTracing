#pragma once

#ifndef _RAYTRACER_H_
#define _RAYTRACER_H_

#include <string>
#include <unordered_map>
#include "Ray.h"
#include "gmath.h"
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
	void Shade(__in const GPoint3& point, __in const GVector3& normal,
			__in const int intersected_obj_id, __in const Ray& ray,
			__in const int recur_num, __out Color& color);
	/*
	*	find nearest intersection point from ray's origin
	*/
	void Closest_intersection(__out GPoint3& out_point, __out GVector3& out_normal,
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

#endif