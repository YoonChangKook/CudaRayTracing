
#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <gl/glut.h>
#include <iostream>
#include "RayTracer.h"
#include "Camera.h"
#include "PointLight.h"
#include "Object.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Plane.h"
#include "gmath.h"

using namespace std;

int main()
{
	RayTracer ray_tracer;

	ray_tracer.SetCamera(
		Camera(GPoint3(12.0, 12.0, -20.0), GVector3(0.0, 1.0, 0.0), 
				GVector3(-12.0, -14.0, 20.0), 120.0f));

	ray_tracer.AddLight(PointLight(GPoint3(0.0, 7.0, 0.0), Color(100, 100, 100)));

	for(int i = 0; i < 10; i++)
		for(int j = 0; j < 10; j++)
			ray_tracer.AddObject(Sphere(GPoint3(i * 1.0 - 5.0, -0.2, j * 1.0 - 5.0), 0.5, Color(i * 10 + 50, j * 10 + 50, 0),
				Color(200, 200, 200), 20.0f, 0.2f, 0.0f, 1.2f));
	
	ray_tracer.AddObject(Plane(GVector3(0.0, 1.0, 0.0), GPoint3(0.0, -1.0, 0.0),
								Color(140, 140, 140), Color(140, 140, 140),
								30.0f, 0.0f, 0.0f, 1.2f));

	ray_tracer.SetImageResolution(1280, 720);

	ray_tracer.RayTrace("test.ppm");

	return 0;
}