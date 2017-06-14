#ifndef _K_VECTOR3_H_
#define _K_VECTOR3_H_

#include "KConst.cuh"
#include "KPoint3.cuh"

class KVector3
{
	// 프렌드 함수 및 클래스
	__host__ __device__ friend KVector3 operator +(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KPoint3 operator +(const KVector3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend KVector3 operator -(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KVector3 operator *(const KVector3 &lhs, const double &s);
	__host__ __device__ friend KVector3 operator *(const double &s, const KVector3 &rhs);
	__host__ __device__ friend double operator *(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KVector3 operator /(const KVector3 &lhs, const double &s);
	__host__ __device__ friend KVector3 operator ^(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend bool operator ==(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend bool operator !=(const KVector3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KVector3 proj(const KVector3 &v, const KVector3 &w);
	__host__ __device__ friend double dist(const KVector3 &v, const KVector3 &w);
	__host__ __device__ friend double norm(const KVector3 &v);
	__host__ __device__ friend double angle(const KVector3 &v, const KVector3 &w, bool radian = false);
	__host__ __device__ friend KPoint3 cast_pt3(const KVector3 &v);

public:
	// 생성자 및 소멸자
	__host__ __device__ KVector3(double x = 0.0, double y = 0.0, double z = 0.0);
	__host__ __device__ KVector3(const KVector3 &cpy);
	__host__ __device__ virtual ~KVector3();

	// 대입 및 복합대입연산자
	__host__ __device__ KVector3 &operator =(const KVector3 &rhs);
	__host__ __device__ KVector3 &operator +=(const KVector3 &rhs);
	__host__ __device__ KVector3 &operator -=(const KVector3 &rhs);
	__host__ __device__ KVector3 &operator *=(const double &s);
	__host__ __device__ KVector3 &operator /=(const double &s);
	__host__ __device__ KVector3 &operator ^=(const KVector3 &rhs);

	// 단항연산자
	__host__ __device__ KVector3 operator +() const;
	__host__ __device__ KVector3 operator -() const;

	// 첨자연산자
	__host__ __device__ double &operator [](const int &idx);
	__host__ __device__ const double &operator [](const int &idx) const;

	// 멤버함수
	__host__ __device__ KVector3 &Set(const double &x, const double &y, const double &z);
	__host__ __device__ KVector3 &Normalize();
	__host__ __device__ KVector3 NormalizeCopy() const;

	// 정적맴버함수
	__host__ __device__ static void SetPrecision(double error);
	__host__ __device__ static double GetPrecision();

protected:
	// 데이터 멤버
	double V[3];
	static double Precision;
};

inline KVector3::KVector3(double x, double y, double z)
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

inline KVector3 &KVector3::operator *=(const double &s)
{
	V[0] *= s;
	V[1] *= s;
	V[2] *= s;
	return *this;
}

inline KVector3 &KVector3::operator /=(const double &s)
{
	V[0] /= s;
	V[1] /= s;
	V[2] /= s;
	return *this;
}

inline KVector3 &KVector3::operator ^=(const KVector3 &rhs)
{
	double x = V[0], y = V[1], z = V[2];
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

inline double operator *(const KVector3 &lhs, const KVector3 &rhs)
{
	return lhs.V[0] * rhs.V[0] + lhs.V[1] * rhs.V[1] + lhs.V[2] * rhs.V[2];
}

inline KVector3 operator /(const KVector3 &lhs, const double &s)
{
	return KVector3(lhs.V[0] / s, lhs.V[1] / s, lhs.V[2] / s);
}

inline KVector3 operator ^(const KVector3 &lhs, const KVector3 &rhs)
{
	return KVector3(lhs.V[1] * rhs.V[2] - lhs.V[2] * rhs.V[1], lhs.V[2] * rhs.V[0] - lhs.V[0] * rhs.V[2], lhs.V[0] * rhs.V[1] - lhs.V[1] * rhs.V[0]);
}

inline bool operator ==(const KVector3 &lhs, const KVector3 &rhs)
{
	double error = KVector3::Precision;
	return (EQ(lhs.V[0], rhs.V[0], error) && EQ(lhs.V[1], rhs.V[1], error) && EQ(lhs.V[2], rhs.V[2], error));
}

inline bool operator !=(const KVector3 &lhs, const KVector3 &rhs)
{
	double error = KVector3::Precision;
	return (!EQ(lhs.V[0], rhs.V[0], error) || !EQ(lhs.V[1], rhs.V[1], error) || !EQ(lhs.V[2], rhs.V[2], error));
}

inline double &KVector3::operator [](const int &idx)
{
	return V[idx];
}

inline const double &KVector3::operator [](const int &idx) const
{
	return V[idx];
}

inline KVector3 &KVector3::Set(const double &x, const double &y, const double &z)
{
	V[0] = x;
	V[1] = y;
	V[2] = z;
	return *this;
}

inline KVector3 &KVector3::Normalize()
{
	double len = norm(*this);
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
	double len = norm(temp_v);
	if (EQ_ZERO(len, Precision))
		return temp_v;
	temp_v[0] /= len;
	temp_v[1] /= len;
	temp_v[2] /= len;
	return temp_v;
}

inline void KVector3::SetPrecision(double error)
{
	Precision = error;
}

inline double KVector3::GetPrecision()
{
	return Precision;
}

inline KVector3 operator *(const KVector3 &lhs, const double &s)
{
	KVector3 ret(lhs);
	ret *= s;
	return ret;
}

inline KVector3 operator *(const double &s, const KVector3 &rhs)
{
	KVector3 ret(rhs);
	ret *= s;
	return ret;
}

inline KVector3 proj(const KVector3 &v, const KVector3 &w)
{
	return (v * w / (w.V[0] * w.V[0] + w.V[1] * w.V[1] + w.V[2] * w.V[2])) * w;
}

inline double dist(const KVector3 &v, const KVector3 &w)
{
	return norm(v - w);
}

inline double norm(const KVector3 &v)
{
	return SQRT(SQR(v.V[0]) + SQR(v.V[1]) + SQR(v.V[2]));
}

inline double angle(const KVector3 &v, const KVector3 &w, bool radian)
{
	KVector3 p(v);
	KVector3 q(w);
	double cs, sn, theta;

	p.Normalize();
	q.Normalize();

	cs = p * q;
	sn = norm(p ^ q);

	theta = radian ? atan2(sn, cs) : RAD2DEG(atan2(sn, cs));
	return theta;
}

#endif