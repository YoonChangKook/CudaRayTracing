#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
//#include <gl/glew.h>
////#include <gl/GL.h>
//#include <gl/glut.h>
#include <helper_gl.h>
#include <GL/freeglut.h>

//#include "cuda_runtime.h"
//#include "cuda_gl_interop.h"
//#include "device_atomic_functions.h"
//#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include "math_functions.h"

#include "KMath.h"
#include "RayTracer.h"
#include "Camera.h"
#include "PointLight.h"
#include "Object.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Plane.h"

//#pragma comment(lib, "glut32.lib")
//#pragma comment(lib, "glu32.lib")
//#pragma comment(lib, "opengl32.lib")
//#pragma comment(lib, "glut32.lib")
//#pragma comment(lib, "glew32.lib")
//#pragma comment(lib, "glew32s.lib")

using namespace std;

// 픽셀 버퍼를 가르키는 OpenGL 핸들 
GLuint gl_pbo;
// 픽셀 버퍼를 가르키는 CUDA 핸들 
struct cudaGraphicsResource *cuda_pbo_resource;
// 픽셀 버퍼에 대한 실제 메모리 주소
uchar4 *dev_data;
// RayTracer
RayTracer* ray_tracer;
const int width = 1600;
const int height = 900;

// OpenGL
bool is_camera_rotate = false;

//Camera camera;
//KVector3 kvec, kvec2;
//KPoint3 kpos;
//Color diffuse, specular;
//float fovx;
//int id;
Camera* dev_camera;
KVector3* dev_kvec, *dev_kvec2;
KPoint3* dev_kpos;
Color* dev_diffuse, *dev_specular;
float* dev_fovx;
int* dev_id;

// Scene Functions
void scene_set_camera(__in const KPoint3& pos, __in const KVector3& up, 
					__in const KVector3& look, __in const float& fovx);
void scene_add_sphere(__in const KPoint3& pos, __in const Color& diffuse, __in const Color& specular,
					__in const float& r, __in const float& shininess, __in const float& reflect,
					__in const float& refract, __in const float& density, __out int& id);
void scene_add_plane(__in const KVector3& normal, __in const KPoint3& point,
					__in const Color& diffuse, __in const Color& specular, 
					__in const float& shininess, __in const float& reflect,
					__in const float& refract, __in const float& density, __out int& id);
void scene_modify_sphere(__in int id, __in const KPoint3& pos, __in const Color& diffuse, __in const Color& specular,
						__in const float& r, __in const float& shininess, __in const float& reflect,
						__in const float& refract, __in const float& density);
void scene_modify_plane(__in int id, __in const KVector3& normal, __in const KPoint3& point,
						__in const Color& diffuse, __in const Color& specular,
						__in const float& shininess, __in const float& reflect,
						__in const float& refract, __in const float& density);
void scene_add_point_light(__in const KPoint3& point, __in const Color& color, __out int& id);
void scene_modify_point_light(__in int id, __in const KPoint3& point, __in const Color& color);

// OPENGL Functions
void mouse(__in int button, __in int state, __in int x, __in int y);
void motion(__in int _x, __in int _y)

void destroy_buffer(GLuint* buffer)
{
	glBindBuffer(GL_TEXTURE_2D, 0);
	glDeleteBuffers(1, buffer);
	*buffer = 0;
}

void display_func()
{
	// TEST!!!
	static float test_num = 0.0f;
	test_num -= 0.1f;
	scene_set_camera(KPoint3(15.0f + test_num, 5.0f, -25.0f), KVector3(0.0f, 1.0f, 0.0f),
					KVector3(-7.0f, -5.0f, 20.0f), 120.0f);

	glClearColor(0, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	// 커널의 위한 디바이스 메모리의 실제 주소(dev_data)를 구함
	size_t size;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void **)&dev_data, &size, cuda_pbo_resource);
	
	dim3 dimGrim(50, 30);
	dim3 dimBlock(width / dimGrim.x, height / dimGrim.y);
	clock_t time_st = clock();
	RayTrace << <dimGrim, dimBlock >> > (ray_tracer, (unsigned char*)dev_data);

	// 공유자원의 매핑 해제
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, NULL);

	cudaDeviceSynchronize();

	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	clock_t time_ed = clock();
	cout << "Elapsed time: " << time_ed - time_st << "ms" << endl;

	//glFinish();
	glutSwapBuffers();
}

int main(int argc, char* argv[])
{
	// GLUT 초기화
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);

	// 윈도우 크기 설정 및 생성
	glutInitWindowSize(width, height);
	glutCreateWindow("Cuda RayTracer");

	// 콜백 함수 등록
	glutDisplayFunc(display_func);
	//glutReshapeFunc(Reshape);
	//glutCloseFunc(Close);

	// OpenGL 2.0지원 여부를 조사
	if (!isGLVersionSupported(2, 0))
		return 0;	

	cudaSetDevice(0);

	// 버퍼의 핸들을 생성하고, 핸들을 픽셀 버퍼에 바인딩
	glGenBuffers(1, &gl_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pbo);

	// OpenGL에 픽셀 버퍼 할당 요청
	// 파라미터 설명:
	//		NULL: 초기화 데이터 없음
	//		GL_DYNAMIC_DRAW_ARB: 버퍼가 반복적으로 수정될 것임	
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	// OpenGL의 픽셀 버퍼를 그래픽 리소스로서 CUDA 시스템과 공유할 것을 알림
	// 변수 cuda_pbo_resource에 CUDA에서 사용할 픽셀 버퍼에 대한 핸들 값을 저장
	// 상수: cudaGraphicsMapFlagsNone, cudaGraphicsmapFlagsReadOnly, cudaGraphicsMapFlagsWriteDiscard, …
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_pbo, cudaGraphicsMapFlagsNone);

	// 커널의 위한 디바이스 메모리의 실제 주소(dev_data)를 구함
	size_t size;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void **)&dev_data, &size, cuda_pbo_resource);

	// 쿠다 스택, 힙 사이즈 지정
	cudaDeviceSetLimit(cudaLimitStackSize, 10000);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, width * height * sizeof(Color) * 5);

	cudaMalloc((void**)&ray_tracer, sizeof(RayTracer));
	cudaMalloc((void**)&dev_camera, sizeof(Camera));
	cudaMalloc((void**)&dev_kvec, sizeof(KVector3));
	cudaMalloc((void**)&dev_kvec2, sizeof(KVector3));
	cudaMalloc((void**)&dev_kpos, sizeof(KPoint3));
	cudaMalloc((void**)&dev_fovx, sizeof(float));
	cudaMalloc((void**)&dev_diffuse, sizeof(Color));
	cudaMalloc((void**)&dev_specular, sizeof(Color));
	cudaMalloc((void**)&dev_id, sizeof(int));

	// Camera
	scene_set_camera(KPoint3(15.0f, 5.0f, -25.0f), KVector3(0.0f, 1.0f, 0.0f),
					KVector3(-7.0f, -5.0f, 20.0f), 120.0f);

	// Light
	scene_add_point_light(KPoint3(0.0f, 10.5f, 0.0f), Color(200, 200, 200), id);

	// 구 25개
	for(int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
		{
			scene_add_sphere(KPoint3(i * 3.0f - 6.0f, -0.5f, j * 3.0f - 6.0f), Color(i * 30 + 100, j * 30 + 100, 0),
							Color(200, 200, 200), 0.5f, 20.0f, 0.25f, 0.0f, 1.2f, id);
		}

	// 유리구슬 한개
	scene_add_sphere(KPoint3(13.0f, 4.0f, -21.0f), Color(0, 0, 0),
					Color(0, 0, 0), 0.5f, 20.0f, 0.0f, 1.0f, 1.705f, id);

	// Plane
	scene_add_plane(KVector3(0.0f, 1.0f, 0.0f), KPoint3(0.0f, -1.0f, 0.0f),
					Color(140, 140, 140), Color(140, 140, 140), 30.0f, 0.0f, 0.0f, 1.2f, id);

	SetImageResolution<<<1, 1>>>(ray_tracer, width, height);

	// 이벤트 처리 루프 진입
	glutMainLoop();
	
	// delete
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
	destroy_buffer(&gl_pbo);
	cudaFree(ray_tracer);
	cudaFree(dev_camera);
	cudaFree(dev_diffuse);
	cudaFree(dev_specular);
	cudaFree(dev_id);
	cudaFree(dev_kpos);
	cudaFree(dev_kvec);
	cudaFree(dev_kvec2);
	cudaFree(dev_fovx);

	return 0;
}

void scene_set_camera(const KPoint3& pos, const KVector3& up,
					const KVector3& look, const float& fovx)
{
	// Camera
	cudaMemcpy(dev_kpos, &pos, sizeof(KPoint3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kvec, &up, sizeof(KVector3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kvec2, &look, sizeof(KVector3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_fovx, &fovx, sizeof(float), cudaMemcpyHostToDevice);
	SetCamera << <1, 1 >> >(ray_tracer, dev_kpos, dev_kvec, dev_kvec2, dev_fovx);
}

void scene_add_sphere(const KPoint3& pos, const Color& diffuse, const Color& specular,
					const float& r, const float& shininess, const float& reflect,
					const float& refract, const float& density, int& id)
{
	cudaMemcpy(dev_kpos, &pos, sizeof(KPoint3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diffuse, &diffuse, sizeof(Color), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_specular, &specular, sizeof(Color), cudaMemcpyHostToDevice);
	AddSphere << <1, 1 >> >(ray_tracer, dev_kpos, r, dev_diffuse, dev_specular, shininess, reflect, refract, density, dev_id);
	
	cudaMemcpy(&id, dev_id, sizeof(int), cudaMemcpyDeviceToHost);
}

void scene_add_plane(const KVector3& normal, const KPoint3& point,
					const Color& diffuse, const Color& specular,
					const float& shininess, const float& reflect,
					const float& refract, const float& density, int& id)
{
	cudaMemcpy(dev_kvec, &normal, sizeof(KVector3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kpos, &point, sizeof(KPoint3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diffuse, &diffuse, sizeof(Color), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_specular, &specular, sizeof(Color), cudaMemcpyHostToDevice);
	AddPlane << <1, 1 >> >(ray_tracer, dev_kvec, dev_kpos, dev_diffuse, dev_specular, shininess, reflect, refract, density, dev_id);

	cudaMemcpy(&id, dev_id, sizeof(int), cudaMemcpyDeviceToHost);
}

void scene_modify_sphere(__in int id, __in const KPoint3& pos, __in const Color& diffuse, __in const Color& specular,
						__in const float& r, __in const float& shininess, __in const float& reflect,
						__in const float& refract, __in const float& density)
{
	cudaMemcpy(dev_kpos, &pos, sizeof(KPoint3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diffuse, &diffuse, sizeof(Color), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_specular, &specular, sizeof(Color), cudaMemcpyHostToDevice);
	ModifySphere<<<1, 1>>>(id, ray_tracer, dev_kpos, r, dev_diffuse, dev_specular, shininess, reflect, refract, density);
}

void scene_modify_plane(__in int id, __in const KVector3& normal, __in const KPoint3& point,
						__in const Color& diffuse, __in const Color& specular,
						__in const float& shininess, __in const float& reflect,
						__in const float& refract, __in const float& density)
{
	cudaMemcpy(dev_kvec, &normal, sizeof(KVector3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kpos, &point, sizeof(KPoint3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diffuse, &diffuse, sizeof(Color), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_specular, &specular, sizeof(Color), cudaMemcpyHostToDevice);
	ModifyPlane << <1, 1 >> > (id, ray_tracer, dev_kvec, dev_kpos, dev_diffuse, dev_specular, shininess, reflect, refract, density);
}

void scene_add_point_light(__in const KPoint3& point, __in const Color& color, __out int& id)
{
	cudaMemcpy(dev_kpos, &point, sizeof(KPoint3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diffuse, &color, sizeof(Color), cudaMemcpyHostToDevice);
	AddPointLight << <1, 1 >> > (ray_tracer, dev_kpos, dev_diffuse, dev_id);

	cudaMemcpy(&id, dev_id, sizeof(int), cudaMemcpyDeviceToHost);
}

void scene_modify_point_light(__in int id, __in const KPoint3& point, __in const Color& color)
{
	cudaMemcpy(dev_kpos, &point, sizeof(KPoint3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diffuse, &color, sizeof(Color), cudaMemcpyHostToDevice);
	ModifyPointLight << <1, 1 >> > (id, ray_tracer, dev_kpos, dev_diffuse);
}

// OPENGL FUNC
//void mouse(__in int button, __in int state, __in int x, __in int y)
//{
//	// Todo :  회전, 이동, 크기 변환을 위한 마우스 부분
//	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
//	{
//		is_camera_rotate = true;
//		beforePoint[0] = x;
//		beforePoint[1] = y;
//	}
//
//	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
//		is_camera_rotate = false;
//}
//void motion(int _x, int _y)
//{
//	// 오른쪽 클릭하고 드래그 시 카메라 회전
//	if (g_cameraRotate)
//	{
//		GLfloat l[3], r[3], u[3];
//		GLfloat newEye[3];
//		GLfloat eyeToNewEye[2];
//
//		eyeToNewEye[0] = (_x - beforePoint[0]);
//		eyeToNewEye[1] = (_y - beforePoint[1]);
//
//		for (int i = 0; i < 3; i++)
//			l[i] = -g_cameraPoint[i] / g_vectorLength;
//
//		r[0] = -l[2];
//		r[1] = 0;
//		r[2] = l[0];
//
//		u[0] = -l[0] * l[1];
//		u[1] = l[2] * l[2] + l[0] * l[0];
//		u[2] = -l[1] * l[2];
//
//		for (int i = 0; i < 3; i++)
//			newEye[i] = g_cameraPoint[i] + r[i] * (-eyeToNewEye[0]) + u[i] * eyeToNewEye[1];
//
//		GLfloat newEyeLength = sqrt(newEye[0] * newEye[0] + newEye[1] * newEye[1] + newEye[2] * newEye[2]);
//
//		for (int i = 0; i < 3; i++)
//			newEye[i] = g_vectorLength * newEye[i] / newEyeLength;
//
//		for (int i = 0; i < 3; i++)
//			g_cameraPoint[i] = newEye[i];
//
//		beforePoint[0] = _x;
//		beforePoint[1] = _y;
//	}
//}