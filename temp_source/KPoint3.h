#ifndef _K_POINT3_H_
#define _K_POINT3_H_

#include "KConst.h"
#include "KVector3.h"

class KPoint3
{
	// ������ �Լ� �� Ŭ����
	__host__ __device__ friend KVector3 operator -(const KPoint3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend KPoint3 operator -(const KPoint3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KPoint3 operator +(const KPoint3 &lhs, const KVector3 &rhs);
	__host__ __device__ friend KPoint3 operator +(const KVector3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend bool operator ==(const KPoint3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend bool operator !=(const KPoint3 &lhs, const KPoint3 &rhs);
	__host__ __device__ friend double norm(const KPoint3 &p);
	__host__ __device__ friend double dist(const KPoint3 &p, const KPoint3 &q);
	__host__ __device__ friend KPoint3 barycentric_combination(KPoint3 *Points, double *Weights, const int Size);
	__host__ __device__ friend KVector3 cast_vec3(const KPoint3 &p);

public:
	// ������ �� �Ҹ���
	__host__ __device__ KPoint3(double x = 0.0, double y = 0.0, double z = 0.0);
	__host__ __device__ KPoint3(const KPoint3 &cpy);
	__host__ __device__ virtual ~KPoint3();

	// ���Կ�����
	__host__ __device__ KPoint3 &operator =(const KPoint3 &rhs);

	// ÷�ڿ�����
	__host__ __device__ double &operator [](const int &idx);
	__host__ __device__ const double &operator [](const int &idx) const;

	// ����Լ�
	__host__ __device__ KPoint3 &Set(const double &x, const double &y, const double &z);

	// �����Լ�
	__host__ __device__ static void SetPrecision(double error);
	__host__ __device__ static double GetPrecision();

protected:
	// ������ ���
	double V[3];	/*! \breif 3���� ������ ���Ҹ� �����ϴ� �Ǽ��迭 */
	static double Precision;	/*! \breif ��ȣ �� �ε�ȣ�������� �����Ѱ� */
};

inline KPoint3::KPoint3(double x, double y, double z)
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
	return KVector3(lhs.V[0] - rhs.V[0], lhs.V[1] - rhs.V[1], lhs.V[2] - rhs.V[2]);
}

inline KPoint3 operator -(const KPoint3 &lhs, const KVector3 &rhs)
{
	return KPoint3(lhs.V[0] - rhs[0], lhs.V[1] - rhs[1], lhs.V[2] - rhs[2]);
}

inline KPoint3 operator +(const KPoint3 &lhs, const KVector3 &rhs)
{
	return KPoint3(lhs.V[0] + rhs[0], lhs.V[1] + rhs[1], lhs.V[2] + rhs[2]);
}

inline KPoint3 operator +(const KVector3 &lhs, const KPoint3 &rhs)
{
	return KPoint3(lhs.V[0] + rhs[0], lhs.V[1] + rhs[1], lhs.V[2] + rhs[2]);
}

inline bool operator ==(const KPoint3 &lhs, const KPoint3 &rhs)
{
	double error = KPoint3::Precision;
	return (EQ(lhs.V[0], rhs.V[0], error) && EQ(lhs.V[1], rhs.V[1], error) && EQ(lhs.V[2], rhs.V[2], error));
}

inline bool operator !=(const KPoint3 &lhs, const KPoint3 &rhs)
{
	double error = KPoint3::Precision;
	return (!EQ(lhs.V[0], rhs.V[0], error) || !EQ(lhs.V[1], rhs.V[1], error) || !EQ(lhs.V[2], rhs.V[2], error));
}

inline double &KPoint3::operator [](const int &idx)
{
	return V[idx];
}

inline const double &KPoint3::operator [](const int &idx) const
{
	return V[idx];
}

inline KPoint3 &KPoint3::Set(const double &x, const double &y, const double &z)
{
	V[0] = x;
	V[1] = y;
	V[2] = z;
	return *this;
}

inline void KPoint3::SetPrecision(double error)
{
	Precision = error;
}

inline double KPoint3::GetPrecision()
{
	return Precision;
}

inline double norm(const KPoint3 &p)
{
	return SQRT(SQR(p.V[0]) + SQR(p.V[1]) + SQR(p.V[2]));
}

inline double dist(const KPoint3 &p, const KPoint3 &q)
{
	return SQRT(SQR(p.V[0] - q.V[0]) + SQR(p.V[1] - q.V[1]) + SQR(p.V[2] - q.V[2]));
}

inline KVector3 cast_vec3(const KPoint3 &pt)
{
	return KVector3(pt[0], pt[1], pt[2]);
}

inline KPoint3 barycentric_combination(KPoint3 *Points, double *Weights, const int Size)
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

#endif