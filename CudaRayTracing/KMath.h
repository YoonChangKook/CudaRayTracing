#ifndef _K_MATH_H_
#define _K_MATH_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__

// 매크로 상수
#define M_PI       3.14159265358979323846
#define M_PI_2     1.57079632679489661923
#define M_PI_4     0.785398163397448309616

// 매크로 정의
#define SQRT(X)		sqrt((X))
#define SQR(X)		((X) * (X))
#define DEG2RAD(X)	((X) * (M_PI) / (180.0))
#define RAD2DEG(X)	((X) * (180.0) / (M_PI))
#define SWAP(type, x, y) { type temp = (x); (x) = (y); (y) = temp; }
#define MIN(x, y)	((x) > (y) ? (y) : (x))
#define MAX(x, y)	((x) > (y) ? (x) : (y))
#define ABS(X)		(((X) > 0.0) ? (X) : (-(X)))
#define SIGN(a)		((a) > 0.0 ? (1.0) : (-1.0))
#define SIGN1(a, b) ((b) > 0.0 ? ABS(a) : -ABS(a))
#define SIGN2(a, b)	((b) >= 0.0 ? fabs(a) : -fabs(a))
#define PYTHAG(a, b) SQRT((SQR(a) + SQR(b)))
#define EQ(X, Y, EPS)	(ABS((X) - (Y)) < EPS)
#define EQ_ZERO(X, EPS) (ABS(X) < EPS)
#define ARR_ZERO(A, N) memset((A), 0, sizeof(A[0]) * (N))
#define ARR_COPY(D, S, N) memmove((D), (S), sizeof(S[0]) * (N))

class KPoint3;
class KVector3;

class KPoint3
{
	// 프렌드 함수 및 클래스
	__host__ __device__ friend KVector3 operator -(const KPoint3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend KPoint3 operator -(const KPoint3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KPoint3 operator +(const KPoint3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KPoint3 operator +(const KVector3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend bool operator ==(const KPoint3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend bool operator !=(const KPoint3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend float norm(const KPoint3 &p);
	__host__ __device__ friend float dist(const KPoint3 &p, const KPoint3 &q);
	__host__ __device__ friend KPoint3 barycentric_combination(KPoint3 *Points, float *Weights, const int Size);
	__host__ __device__ friend KVector3 cast_vec3(const KPoint3 &p);

public:
	// 생성자 및 소멸자
	__host__ __device__ KPoint3(float x = 0.0, float y = 0.0, float z = 0.0);
	__host__ __device__ KPoint3(const KPoint3 &cpy);
	__host__ __device__ ~KPoint3();

	// 대입연산자
	__host__ __device__ KPoint3 &operator =(const KPoint3 &rhs);

	// 첨자연산자
	__host__ __device__ float &operator [](const int &idx);
	__host__ __device__ const float &operator [](const int &idx) const;

	// 멤버함수
	__host__ __device__ KPoint3 &Set(const float &x, const float &y, const float &z);
	__host__ __device__ float GetPrecision() const;

	// 정적함수
	//__host__ __device__ static void SetPrecision(float error);
	//__host__ __device__ static float GetPrecision();

protected:
	// 데이터 멤버
	float V[3];	/*! \breif 3차원 벡터의 원소를 저장하는 실수배열 */
	const float Precision = 0.0001f;	/*! \breif 등호 및 부등호연산자의 오차한계 */
};

class KVector3
{
	// 프렌드 함수 및 클래스
	__host__ __device__ friend KVector3 operator +(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KPoint3 operator +(const KVector3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend KVector3 operator -(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KVector3 operator *(const KVector3 &lhs, const float &s);
	__host__ __device__ friend KVector3 operator *(const float &s, const KVector3 &rhs);
	__host__ __device__ friend float operator *(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KVector3 operator /(const KVector3 &lhs, const float &s);
	__host__ __device__ friend KVector3 operator ^(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend bool operator ==(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend bool operator !=(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KVector3 proj(const KVector3 &v, const KVector3 &w);
	__host__ __device__ friend float dist(const KVector3 &v, const KVector3 &w);
	__host__ __device__ friend float norm(const KVector3 &v);
	__host__ __device__ friend float angle(const KVector3 &v, const KVector3 &w, bool radian = false);
	__host__ __device__ friend KPoint3 cast_pt3(const KVector3 &v);

public:
	// 생성자 및 소멸자
	__host__ __device__ KVector3(float x = 0.0, float y = 0.0, float z = 0.0);
	__host__ __device__ KVector3(const KVector3 &cpy);
	__host__ __device__ ~KVector3();

	// 대입 및 복합대입연산자
	__host__ __device__ KVector3 &operator =(const KVector3 &rhs);
	__host__ __device__ KVector3 &operator +=(const KVector3 &rhs);
	__host__ __device__ KVector3 &operator -=(const KVector3 &rhs);
	__host__ __device__ KVector3 &operator *=(const float &s);
	__host__ __device__ KVector3 &operator /=(const float &s);
	__host__ __device__ KVector3 &operator ^=(const KVector3 &rhs);

	// 단항연산자
	__host__ __device__ KVector3 operator +() const;
	__host__ __device__ KVector3 operator -() const;

	// 첨자연산자
	__host__ __device__ float &operator [](const int &idx);
	__host__ __device__ const float &operator [](const int &idx) const;

	// 멤버함수
	__host__ __device__ KVector3 &Set(const float &x, const float &y, const float &z);
	__host__ __device__ KVector3 &Normalize();
	__host__ __device__ KVector3 NormalizeCopy() const;
	__host__ __device__ float GetPrecision() const;

	// 정적맴버함수
	//__host__ __device__ static void SetPrecision(float error);
	//__host__ __device__ static float GetPrecision();

protected:
	// 데이터 멤버
	float V[3];
	const float Precision = 0.0001f;
};

inline KPoint3::KPoint3(float x, float y, float z)
	: V{ x, y, z }
{}

inline KPoint3::KPoint3(const KPoint3 &cpy)
	: V{ cpy.V[0], cpy.V[1], cpy.V[2] }
{}

inline KPoint3::~KPoint3()
{}

inline KPoint3 &KPoint3::operator =(const KPoint3 &rhs)
{
	V[0] = rhs.V[0];
	V[1] = rhs.V[1];
	V[2] = rhs.V[2];
	return *this;
}

inline KVector3 operator -(const KPoint3 &lhs, const KPoint3 &rhs)
{
	return KVector3(lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]);
}

inline KPoint3 operator -(const KPoint3 &lhs, const KVector3 &rhs)
{
	return KPoint3(lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]);
}

inline KPoint3 operator +(const KPoint3 &lhs, const KVector3 &rhs)
{
	return KPoint3(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]);
}

inline KPoint3 operator +(const KVector3 &lhs, const KPoint3 &rhs)
{
	return KPoint3(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]);
}

inline bool operator ==(const KPoint3 &lhs, const KPoint3 &rhs)
{
	float error = lhs.Precision;
	return (EQ(lhs.V[0], rhs.V[0], error) && EQ(lhs.V[1], rhs.V[1], error) && EQ(lhs.V[2], rhs.V[2], error));
}

inline bool operator !=(const KPoint3 &lhs, const KPoint3 &rhs)
{
	float error = lhs.Precision;
	return (!EQ(lhs.V[0], rhs.V[0], error) || !EQ(lhs.V[1], rhs.V[1], error) || !EQ(lhs.V[2], rhs.V[2], error));
}

inline float &KPoint3::operator [](const int &idx)
{
	return V[idx];
}

inline const float &KPoint3::operator [](const int &idx) const
{
	return V[idx];
}

inline KPoint3 &KPoint3::Set(const float &x, const float &y, const float &z)
{
	V[0] = x;
	V[1] = y;
	V[2] = z;
	return *this;
}

inline float KPoint3::GetPrecision() const
{
	return this->Precision;
}

/*inline void KPoint3::SetPrecision(float error)
{
Precision = error;
}

inline float KPoint3::GetPrecision()
{
return Precision;
}*/

inline float norm(const KPoint3 &p)
{
	return SQRT(SQR(p.V[0]) + SQR(p.V[1]) + SQR(p.V[2]));
}

inline float dist(const KPoint3 &p, const KPoint3 &q)
{
	return SQRT(SQR(p.V[0] - q.V[0]) + SQR(p.V[1] - q.V[1]) + SQR(p.V[2] - q.V[2]));
}

inline KVector3 cast_vec3(const KPoint3 &pt)
{
	return KVector3(pt[0], pt[1], pt[2]);
}

inline KPoint3 barycentric_combination(KPoint3 *Points, float *Weights, const int Size)
{
	KPoint3 ret;
	for (int i = 0; i < Size; i++)
	{
		ret.V[0] += Points[i][0] * Weights[i];
		ret.V[1] += Points[i][1] * Weights[i];
		ret.V[2] += Points[i][2] * Weights[i];
	}
	return ret;
}

inline KVector3::KVector3(float x, float y, float z)
	: V{ x, y, z }
{}

inline KVector3::KVector3(const KVector3 &cpy)
	: V{ cpy.V[0], cpy.V[1], cpy.V[2] }
{}

inline KVector3::~KVector3()
{}

inline KVector3 &KVector3::operator =(const KVector3 &rhs)
{
	V[0] = rhs.V[0];
	V[1] = rhs.V[1];
	V[2] = rhs.V[2];
	return *this;
}

inline KVector3 &KVector3::operator +=(const KVector3 &rhs)
{
	V[0] += rhs.V[0];
	V[1] += rhs.V[1];
	V[2] += rhs.V[2];
	return *this;
}

inline KVector3 &KVector3::operator -=(const KVector3 &rhs)
{
	V[0] -= rhs.V[0];
	V[1] -= rhs.V[1];
	V[2] -= rhs.V[2];
	return *this;
}

inline KVector3 &KVector3::operator *=(const float &s)
{
	V[0] *= s;
	V[1] *= s;
	V[2] *= s;
	return *this;
}

inline KVector3 &KVector3::operator /=(const float &s)
{
	V[0] /= s;
	V[1] /= s;
	V[2] /= s;
	return *this;
}

inline KVector3 &KVector3::operator ^=(const KVector3 &rhs)
{
	float x = V[0], y = V[1], z = V[2];
	V[0] = y * rhs.V[2] - z * rhs.V[1];
	V[1] = z * rhs.V[0] - x * rhs.V[2];
	V[2] = x * rhs.V[1] - y * rhs.V[0];
	return *this;
}

inline KVector3 KVector3::operator +() const
{
	return *this;
}

inline KVector3 KVector3::operator -() const
{
	return *this * -1;
}

inline KVector3 operator +(const KVector3 &lhs, const KVector3 &rhs)
{
	return KVector3(lhs.V[0] + rhs.V[0], lhs.V[1] + rhs.V[1], lhs.V[2] + rhs.V[2]);
}

inline KVector3 operator -(const KVector3 &lhs, const KVector3 &rhs)
{
	return KVector3(lhs.V[0] - rhs.V[0], lhs.V[1] - rhs.V[1], lhs.V[2] - rhs.V[2]);
}

inline float operator *(const KVector3 &lhs, const KVector3 &rhs)
{
	return lhs.V[0] * rhs.V[0] + lhs.V[1] * rhs.V[1] + lhs.V[2] * rhs.V[2];
}

inline KVector3 operator /(const KVector3 &lhs, const float &s)
{
	return KVector3(lhs.V[0] / s, lhs.V[1] / s, lhs.V[2] / s);
}

inline KVector3 operator ^(const KVector3 &lhs, const KVector3 &rhs)
{
	return KVector3(lhs.V[1] * rhs.V[2] - lhs.V[2] * rhs.V[1], lhs.V[2] * rhs.V[0] - lhs.V[0] * rhs.V[2], lhs.V[0] * rhs.V[1] - lhs.V[1] * rhs.V[0]);
}

inline bool operator ==(const KVector3 &lhs, const KVector3 &rhs)
{
	float error = lhs.Precision;
	return (EQ(lhs.V[0], rhs.V[0], error) && EQ(lhs.V[1], rhs.V[1], error) && EQ(lhs.V[2], rhs.V[2], error));
}

inline bool operator !=(const KVector3 &lhs, const KVector3 &rhs)
{
	float error = lhs.Precision;
	return (!EQ(lhs.V[0], rhs.V[0], error) || !EQ(lhs.V[1], rhs.V[1], error) || !EQ(lhs.V[2], rhs.V[2], error));
}

inline float &KVector3::operator [](const int &idx)
{
	return V[idx];
}

inline const float &KVector3::operator [](const int &idx) const
{
	return V[idx];
}

inline KVector3 &KVector3::Set(const float &x, const float &y, const float &z)
{
	V[0] = x;
	V[1] = y;
	V[2] = z;
	return *this;
}

inline KVector3 &KVector3::Normalize()
{
	float len = norm(*this);
	if (EQ_ZERO(len, Precision))
		return *this;
	V[0] /= len;
	V[1] /= len;
	V[2] /= len;
	return *this;
}

inline KVector3 KVector3::NormalizeCopy() const
{
	KVector3 temp_v(*this);
	float len = norm(temp_v);
	if (EQ_ZERO(len, Precision))
		return temp_v;
	temp_v[0] /= len;
	temp_v[1] /= len;
	temp_v[2] /= len;
	return temp_v;
}

inline float KVector3::GetPrecision() const
{
	return this->Precision;
}

/*inline void KVector3::SetPrecision(float error)
{
Precision = error;
}

inline float KVector3::GetPrecision()
{
return Precision;
}*/

inline KVector3 operator *(const KVector3 &lhs, const float &s)
{
	KVector3 ret(lhs);
	ret *= s;
	return ret;
}

inline KVector3 operator *(const float &s, const KVector3 &rhs)
{
	KVector3 ret(rhs);
	ret *= s;
	return ret;
}

inline KVector3 proj(const KVector3 &v, const KVector3 &w)
{
	return (v * w / (w.V[0] * w.V[0] + w.V[1] * w.V[1] + w.V[2] * w.V[2])) * w;
}

inline float dist(const KVector3 &v, const KVector3 &w)
{
	return norm(v - w);
}

inline float norm(const KVector3 &v)
{
	return SQRT(SQR(v.V[0]) + SQR(v.V[1]) + SQR(v.V[2]));
}

inline float angle(const KVector3 &v, const KVector3 &w, bool radian)
{
	KVector3 p(v);
	KVector3 q(w);
	float cs, sn, theta;

	p.Normalize();
	q.Normalize();

	cs = p * q;
	sn = norm(p ^ q);

	theta = radian ? atan2(sn, cs) : RAD2DEG(atan2(sn, cs));
	return theta;
}

inline KPoint3 cast_pt3(const KVector3 &v)
{
	return KPoint3(v[0], v[1], v[2]);
}

#endif