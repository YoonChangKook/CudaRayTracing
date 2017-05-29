#ifndef _GMATH_H
#define _GMATH_H

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <iomanip>
#include <vector>
#include "cuda_runtime.h"

using namespace std;

// ��ũ�� ���
#define M_PI       3.14159265358979323846
#define M_PI_2     1.57079632679489661923
#define M_PI_4     0.785398163397448309616

// ��ũ�� ����
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

// �ܺ����� Ŭ����
class GPose;

// Ŭ���� ����.
class GVector3;
class GVector;
class GPoint3;
class GMatrix;
class GQuater;
class GTransf;
class GLine;
class GPlane;

/*!
*	\class GVector3
*	\brief 3���� ���͸� ǥ���ϴ� Ŭ���� (n���� ������ Ư���� ���)
*
*	\author ������ (shyun@dongguk.edu)
*	\date 01 Jan 2001	
*/
class GVector3
{
	// ������ �Լ� �� Ŭ����
	__host__ __device__ friend GVector3 operator +(const GVector3 &lhs, const GVector3 &rhs);
	__host__ __device__ friend GPoint3 operator +(const GVector3 &lhs, const GPoint3 &rhs);
	__host__ __device__ friend GVector3 operator -(const GVector3 &lhs, const GVector3 &rhs);
	__host__ __device__ friend GVector3 operator *(const GVector3 &lhs, const double &s);
	__host__ __device__ friend GVector3 operator *(const double &s, const GVector3 &rhs);
	__host__ __device__ friend double operator *(const GVector3 &lhs, const GVector3 &rhs);
	__host__ __device__ friend GVector3 operator /(const GVector3 &lhs, const double &s);
	__host__ __device__ friend GVector3 operator ^(const GVector3 &lhs, const GVector3 &rhs);
	__host__ __device__ friend bool operator ==(const GVector3 &lhs, const GVector3 &rhs);
	__host__ __device__ friend bool operator !=(const GVector3 &lhs, const GVector3 &rhs);
	__host__ __device__ friend ostream &operator <<(ostream &os, const GVector3 &v);
	__host__ __device__ friend istream &operator >>(istream &is, GVector3 &v);
	__host__ __device__ friend GVector3 proj(const GVector3 &v, const GVector3 &w);
	__host__ __device__ friend double dist(const GVector3 &v, const GVector3 &w);
	__host__ __device__ friend double norm(const GVector3 &v);
	__host__ __device__ friend double angle(const GVector3 &v, const GVector3 &w, bool radian = false);
	__host__ __device__ friend GPoint3 cast_pt3(const GVector3 &v);
	__host__ __device__ friend GVector cast_vec(const GVector3 &v);
	__host__ __device__ friend class GLScene;

public:
	// ������ �� �Ҹ���
	__host__ __device__ GVector3(double x = 0.0, double y = 0.0, double z = 0.0);
	__host__ __device__ GVector3(const GVector3 &cpy);
	__host__ __device__ virtual ~GVector3();

	// ���� �� ���մ��Կ�����
	__host__ __device__ GVector3 &operator =(const GVector3 &rhs);
	__host__ __device__ GVector3 &operator +=(const GVector3 &rhs);
	__host__ __device__ GVector3 &operator -=(const GVector3 &rhs);
	__host__ __device__ GVector3 &operator *=(const double &s);
	__host__ __device__ GVector3 &operator /=(const double &s);
	__host__ __device__ GVector3 &operator ^=(const GVector3 &rhs);

	// ���׿�����
	__host__ __device__ GVector3 operator +() const;
	__host__ __device__ GVector3 operator -() const;

	// ÷�ڿ�����
	__host__ __device__ double &operator [](const int &idx);
	__host__ __device__ const double &operator [](const int &idx) const;

	// ����Լ�
	__host__ __device__ GVector3 &Set(const double &x, const double &y, const double &z);
	__host__ __device__ GVector3 &Normalize();
	__host__ __device__ GVector3 NormalizeCopy() const;
		
	// �����ɹ��Լ�
	__host__ __device__ static void SetPrecision(double error);
	__host__ __device__ static double GetPrecision();

protected:
	// ������ ���
	double V[3];
	static double Precision;
};

/*!
*	\brief ����Ʈ ������
*	
*	\param x x-��ǥ
*	\param y y-��ǥ
*	\param z z-��ǥ
*/
inline
GVector3::GVector3(double x, double y, double z)
{
	V[0] = x;
	V[1] = y;
	V[2] = z;
}

/*!
*	\brief	���������
*
*	\param cpy ����� ��ü
*/
inline
GVector3::GVector3(const GVector3 &cpy)
{
	V[0] = cpy.V[0];
	V[1] = cpy.V[1];
	V[2] = cpy.V[2];
}

/*!
*	\brief �Ҹ���
*/
inline
GVector3::~GVector3()
{
}

/*!
*	\brief ���Կ�����
*
*	\param rhs ������ �ǿ�����
*
*	\return ���Ե� �ڽ�
*/
inline
GVector3 &GVector3::operator =(const GVector3 &rhs)
{
	V[0] = rhs.V[0];
	V[1] = rhs.V[1];
	V[2] = rhs.V[2];
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param rhs ������ �ǿ�����
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GVector3 &GVector3::operator +=(const GVector3 &rhs)
{
	V[0] += rhs.V[0];
	V[1] += rhs.V[1];
	V[2] += rhs.V[2];
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param rhs ������ �ǿ�����
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GVector3 &GVector3::operator -=(const GVector3 &rhs)
{
	V[0] -= rhs.V[0];
	V[1] -= rhs.V[1];
	V[2] -= rhs.V[2];
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param s ���� ���
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GVector3 &GVector3::operator *=(const double &s)
{
	V[0] *= s;
	V[1] *= s;
	V[2] *= s;
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param s ���� ���
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GVector3 &GVector3::operator /=(const double &s)
{
	V[0] /= s;
	V[1] /= s;
	V[2] /= s;
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param rhs ������ ������ �ǿ�����
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GVector3 &GVector3::operator ^=(const GVector3 &rhs)
{
	double x = V[0], y = V[1], z = V[2];
	V[0] = y * rhs.V[2] - z * rhs.V[1];
	V[1] = z * rhs.V[0] - x * rhs.V[2];
	V[2] = x * rhs.V[1] - y * rhs.V[0];
	return *this;
}

/*!
*	\brief ���׿�����
*
*	\return �ݴ��ȣ�� ���� ��ü
*/
inline
GVector3 GVector3::operator +() const
{
	return *this;
}

/*!
*	\brief ���׿�����
*
*	\return �ݴ��ȣ�� ���� ��ü
*/
inline
GVector3 GVector3::operator -() const
{
	return *this * -1;
}

/*!
*	\brief ���������(+)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline 
GVector3 operator +(const GVector3 &lhs, const GVector3 &rhs)
{
	return GVector3(lhs.V[0] + rhs.V[0], lhs.V[1] + rhs.V[1], lhs.V[2] + rhs.V[2]);
}

/*!
*	\brief ���������(-)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline 
GVector3 operator -(const GVector3 &lhs, const GVector3 &rhs)
{
	return GVector3(lhs.V[0] - rhs.V[0], lhs.V[1] - rhs.V[1], lhs.V[2] - rhs.V[2]);
}

/*!
*	\brief ����������(*)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
double operator *(const GVector3 &lhs, const GVector3 &rhs)
{
	return lhs.V[0] * rhs.V[0] + lhs.V[1] * rhs.V[1] + lhs.V[2] * rhs.V[2];
}

/*!
*	\brief ���������(/)
*
*	\param s ���� ���
*	\return ������ ����� ��ȯ
*/
inline
GVector3 operator /(const GVector3 &lhs, const double &s)
{
	return GVector3(lhs.V[0] / s, lhs.V[1] / s, lhs.V[2] / s);
}

/*!
*	\brief ����������(^)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GVector3 operator ^(const GVector3 &lhs, const GVector3 &rhs)
{
	return GVector3(lhs.V[1] * rhs.V[2] - lhs.V[2] * rhs.V[1], lhs.V[2] * rhs.V[0] - lhs.V[0] * rhs.V[2], lhs.V[0] * rhs.V[1] - lhs.V[1] * rhs.V[0]);
}

/*!
*	\brief ��ȣ������
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return �� ����� ��ȯ
*/
inline
bool operator ==(const GVector3 &lhs, const GVector3 &rhs)
{
	double error = GVector3::Precision;
	return (EQ(lhs.V[0], rhs.V[0], error) && EQ(lhs.V[1], rhs.V[1], error) && EQ(lhs.V[2], rhs.V[2], error));
}

/*!
*	\brief �ε�ȣ������
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return �� ����� ��ȯ
*/
inline
bool operator !=(const GVector3 &lhs, const GVector3 &rhs)
{
	double error = GVector3::Precision;
	return (!EQ(lhs.V[0], rhs.V[0], error) || !EQ(lhs.V[1], rhs.V[1], error) || !EQ(lhs.V[2], rhs.V[2], error));
}

/*!
*	\brief ÷�ڿ�����([])
*
*	\param idx ÷��
*	\param ������ \a idx��° ���Ҹ� ��ȯ
*/
inline
double &GVector3::operator [](const int &idx)
{
	assert(idx >= 0 && idx < 3);
	return V[idx];
}

/*!
*	\brief �����ü�� ���� ÷�ڿ�����([])
*
*	\param idx ÷��
*	\param ������ \a idx��° ���Ҹ� ��ȯ
*/
inline
const double &GVector3::operator [](const int &idx) const
{
	assert(idx >= 0 && idx < 3);
	return V[idx];
}

/*!
*	\brief ������ ��ǥ�� ����
*
*	\param x x-��ǥ
*	\param y y-��ǥ
*	\param z z-��ǥ
*/
inline
GVector3 &GVector3::Set(const double &x, const double &y, const double &z)
{
	V[0] = x;
	V[1] = y;
	V[2] = z;
	return *this;
}

/*!
*	\brief ���͸� �����Ͽ� �������ͷ� ��ȯ
*
*	\return ��ȯ�� �ڽ��� ��ȯ
*/
inline
GVector3 &GVector3::Normalize()
{
	double len = norm(*this);
	if (EQ_ZERO(len, Precision))
		return *this;
	V[0] /= len;
	V[1] /= len;
	V[2] /= len;
	return *this;
}

inline
GVector3 GVector3::NormalizeCopy() const
{
	GVector3 temp_v(*this);
	double len = norm(temp_v);
	if (EQ_ZERO(len, Precision))
		return temp_v;
	temp_v[0] /= len;
	temp_v[1] /= len;
	temp_v[2] /= len;
	return temp_v;
}

/*!
*	\brief ��ȣ �� �ε�ȣ �������� ������ �Ѱ� ����
*
*	\param error ������ �Ѱ�
*/
inline
void GVector3::SetPrecision(double error)
{
	Precision = error;
}

/*!
*	\brief ������ ������ �Ѱ踦 ��ȯ
*
*	\return ������ ������ �Ѱ�
*/
inline
double GVector3::GetPrecision()
{
	return Precision;
}

/*!
*	\brief ���������(*)
*
*	\param lhs ���� �ǿ�����
*	\param s ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GVector3 operator *(const GVector3 &lhs, const double &s)
{
	GVector3 ret(lhs);
	ret *= s;
	return ret;
}

/*!
*	\brief ���������(*)
*
*	\param s ���� �ǿ�����
*	\param lhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GVector3 operator *(const double &s, const GVector3 &rhs)
{
	GVector3 ret(rhs);
	ret *= s;
	return ret;
}

/*!
*	\brief ��¿�����(<<)
*
*	\param os ��½�Ʈ��
*	\param v ����� ����
*
*	\return \a v�� ��µ� ��Ʈ���� ��ȯ
*/
inline
ostream &operator <<(ostream &os, const GVector3 &v)
{
	os << "(" << setw(10) << v.V[0] << ", " << setw(10) << v.V[1] << ", " << setw(10) << v.V[2] << ")";
	return os;
}

/*!
*	\brief �Է¿�����(>>)
*
*	\param is �Է½�Ʈ��
*	\param v �Է°��� ����� ����
*
*	\return �Է°��� ���ŵ� �Է½�Ʈ��
*/
inline
istream &operator >>(istream &is, GVector3 &v)
{
	is >> v.V[0] >> v.V[1] >> v.V[2];
	return is;
}

/*!
*	\brief ���� v�� ���� w�� �翵��Ų ���͸� ���Ѵ�.
*
*	\param v ����.
*	\param w ����.
*
*	\return ���� v�� ���� w�� �翵��Ų ���͸� ��ȯ�Ѵ�.
*/
inline
GVector3 proj(const GVector3 &v, const GVector3 &w)
{
	return (v * w / (w.V[0] * w.V[0] + w.V[1] * w.V[1] + w.V[2] * w.V[2])) * w;
}

/*!
*	\brief ���� v, w�� ���̺����� ���̸� ���Ѵ�.
*
*	\param v ����.
*	\param w ����.
*
*	\return  ���� \a v, \a w�� ���̺����� ����
*/
inline
double dist(const GVector3 &v, const GVector3 &w)
{
	return norm(v - w);
}

/*!
*	\brief ���� v�� ũ�⸦ ���Ѵ�.
*
*	\param v ����
*
*	\return ���� \a v�� ũ��
*/
inline double norm(const GVector3 &v)
{
	return SQRT(SQR(v.V[0]) + SQR(v.V[1]) + SQR(v.V[2]));
}

/*!
*	\brief �� ���� ������ ���̰��� ���Ѵ�.
*	\warning 0 <= ���̰� <= 180
*
*	\param v ���� ����
*	\param w �� ����
*	\param radian ������ ����(true: radian, false: degree)
*
*	\return ���� \a v���� ���� \a w������ ���̰�
*/
inline
double angle(const GVector3 &v, const GVector3 &w, bool radian)
{
	GVector3 p(v);
	GVector3 q(w);
	double cs, sn, theta;

	p.Normalize();
	q.Normalize();

	cs = p * q;
	sn = norm(p ^ q);

	theta = radian ? atan2(sn, cs) : RAD2DEG(atan2(sn, cs));
	return theta;
}

/*!
*	\class GVector
*	\brief n���� �����ڷ����� ��Ÿ���� Ŭ����
*
*	\author ������ (shyun@dongguk.edu)
*	\date 01 Jan 2001	
*/
class GVector
{
	// ������ �Լ� �� Ŭ����
	friend GVector operator +(const GVector &lhs, const GVector &rhs);
	friend GVector operator -(const GVector &lhs, const GVector &rhs);
	friend double operator *(const GVector &lhs, const GVector &rsh);
	friend GVector operator *(const double &s, const GVector &rhs);
	friend GVector operator *(const GVector &lhs, const double &s);
	friend GVector operator *(const GMatrix &lhs, const GVector &rhs);
	friend GMatrix operator *(const GVector &lhs, const GMatrix &rhs);
	friend GVector operator /(const GVector &lhs, const double &s);
	friend bool operator ==(const GVector &lhs, const GVector &rhs);
	friend bool operator !=(const GVector &lhs, const GVector &rhs);
	friend ostream &operator <<(ostream &os, const GVector &v);
	friend istream &operator >>(istream &is, GVector &v);
	friend double norm(const GVector &v);
	friend double dist(const GVector &v, const GVector &w);
	friend int dcmp_lu(GMatrix &A, GVector &Idx, double &d);
	friend void dcmp_sv(const GMatrix &A, GMatrix &U, GVector &S, GMatrix &V);
	friend void lin_sys_lu(const GMatrix &A, GVector &x, const GVector &b, const GVector &Idx);
	friend double lin_sys_sv(const GMatrix &A, GVector &x, const GVector &b);
	friend double lin_sys_sv(const GMatrix &U, const GVector &S, const GMatrix &V, GVector &x, const GVector &b);
	friend int lin_sys_cg(const GMatrix &A, GVector &x, const GVector &b, int iter);
	friend int lin_sys_qr(const GMatrix &A, GVector &x, const GVector &b);
	friend int eigen_sys(const GMatrix &A, GVector &D, GMatrix &V);
	friend void pca(GMatrix &Data, GMatrix &PC, GVector &SigVal, GVector &Mean);
	friend double *cast_arr(const GVector &v);
	friend GPoint3 cast_pt3(const GVector &v);
	friend GVector3 cast_vec3(const GVector &v);
	friend GQuater cast_quat(const GVector &v);
	friend class GMatrix;
	friend class RBFinterp;

public:
	// ������ �� �Ҹ���
	GVector(int dim = 3);
	GVector(int dim, double x, ...);
	GVector(int dim, int x, ...);
	GVector(const GVector &cpy);
	virtual ~GVector();

	// ���Կ����� �� ���մ��Կ�����
	GVector &operator =(const GVector &rhs);
	GVector &operator +=(const GVector &rhs);
	GVector &operator -=(const GVector &rhs);
	GVector &operator *=(const double &s);
	GVector	&operator /=(const double &s);

	// ���׿�����
	GVector operator +() const;
	GVector operator -() const;

	// ÷�ڿ�����
	double &operator [](const int &idx);
	const double &operator [](const int &idx) const;

	// ����Լ�
	GVector	&Set(double x, ...);
	GVector	&Set(int x, ...);
	GVector &Set(double *pVal);
	GVector	&Normalize();
	GVector &SetZeros();
	int GetDim() const;

	// �����Լ�
	static void	SetPrecision(double error);
	static double GetPrecision();

protected:
	// ������ ���
	int N;
	double *V;
	static double Precision;
};

/*!
*	\brief n ���� ���͸� �����ϴ� ������
*	\note ��� ���Ҵ� 0���� �ʱ�ȭ �ȴ�.
*	
*	\param dim ������ ����
*/
inline
GVector::GVector(int dim)
{
	this->N = dim;
	V = new double [N];
	ARR_ZERO(V, N);
}

/*!
*	\brief n ���� ���͸� �����ϴ� ������
*	
*	\param dim ������ ����
*	\param x ������ ù���� ���� (�������ڷ� ��� �ݺ���)
*/
inline
GVector::GVector(int dim, double x, ...)
{
	this->N = dim;
	V = new double [N];
	va_list ap;
	va_start(ap, dim);
	for (int i = 0; i < N; i++)
		V[i] = va_arg(ap, double);
	va_end(ap);
}

/*!
*	\brief n ���� ���͸� �����ϴ� ������
*	
*	\param dim ������ ����
*	\param x ������ ù���� ���� (�������ڷ� ��� �ݺ���)
*/
inline
GVector::GVector(int dim, int x, ...)
{
	this->N = dim;
	V = new double [N];
	va_list ap;
	va_start(ap, dim);
	for (int i = 0; i < N; i++)
		V[i] = (double)va_arg(ap, int);
	va_end(ap);
}

/*!
*	\brief ���������
*	
*	\param copy ����� ��ü
*/
inline
GVector::GVector(const GVector &cpy)
{
	N = cpy.N;
	V = new double [N];
	memcpy(V, cpy.V, N * sizeof(double));
}

/*!
*	\brief �Ҹ���
*/
inline
GVector::~GVector()
{
	if (V)
		delete [] V;
	V = NULL;
}

/*!
*	\brief ���Կ����� �ߺ�
*
*	\param rhs ���Կ������� ������ �ǿ�����
*	
*	\return ���Ե� �ڽ��� ��ȯ�Ѵ�
*/
inline
GVector &GVector::operator =(const GVector &rhs)
{
	if (V)
		delete [] V;
	N = rhs.N;
	V = new double [N];
	memcpy(V, rhs.V, N * sizeof(double));
	return *this;
}

/*!
*	\brief ���մ��Կ����� �ߺ�
*
*	\param rhs ������ �ǿ�����
*	
*	\return ���Ե� �ڽ��� ��ȯ�Ѵ�
*/
inline
GVector &GVector::operator +=(const GVector &rhs)
{
	assert(N == rhs.N);
	for (int i = 0; i != N; ++i)
		V[i] += rhs.V[i];
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param rhs ������ �ǿ�����
*	
*	\return ���Ե� �ڽ��� ��ȯ�Ѵ�
*/
inline
GVector &GVector::operator -=(const GVector &rhs)
{
	assert(N == rhs.N);
	for (int i = 0; i != N; ++i)
		V[i] -= rhs.V[i];
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param s ������ �ǿ����� (������ ��)
*	
*	\return ���Ե� �ڽ��� ��ȯ�Ѵ�
*/
inline
GVector &GVector::operator *=(const double &s)
{
	for (int i = 0; i != N; ++i)
		V[i] *= s;
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param s ������ �ǿ����� (division factor)
*	
*	\return ���Ե� �ڽ��� ��ȯ�Ѵ�
*/
inline
GVector &GVector::operator /=(const double &s)
{
	assert(s != 0);
	for (int i = 0; i != N; ++i)
		V[i] /= s;
	return *this;
}

/*!
*	\brief ���׿������ߺ� (+)
*
*	\return ���ο� ��ü�� ��ȯ�Ѵ�
*/
inline
GVector GVector::operator +() const
{
	return *this;
}

/*!
*	\brief ���׿������ߺ� (-)
*
*	\return ���ο� ��ü�� ��ȯ�Ѵ�
*/
inline
GVector GVector::operator -() const
{
	return *this * -1;
}

/*!
*	\brief ���������(+)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GVector operator +(const GVector &lhs, const GVector &rhs)
{
	assert(lhs.N == rhs.N);
	GVector ret(lhs.N);
	for (int i = 0; i != lhs.N; ++i)
		ret.V[i] = lhs.V[i] + rhs.V[i];
	return ret;
}

/*!
*	\brief ���������(-)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GVector operator -(const GVector &lhs, const GVector &rhs)
{
	assert(lhs.N == rhs.N);
	GVector ret(lhs.N);
	for (int i = 0; i != lhs.N; ++i)
		ret.V[i] = lhs.V[i] - rhs.V[i];
	return ret;
}

/*!
*	\brief ����������
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
double operator *(const GVector &lhs, const GVector &rhs)
{
	assert(lhs.N == rhs.N);
	double ret = 0;
	for (int i = 0; i != lhs.N; ++i)
		ret += lhs.V[i] * rhs.V[i];
	return ret;
}

/*!
*	\brief ���������(/)
*
*	\param lhs ���� �ǿ�����
*	\param s ���� ��
*
*	\return ������ ����� ��ȯ
*/
inline
GVector operator /(const GVector &lhs, const double &s)
{
	GVector ret(lhs.N);
	for (int i = 0; i != lhs.N; ++i)
		ret.V[i] = lhs.V[i] / s;
	return ret;
}

/*!
*	\brief ��ȣ������(==)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ���� ����� ��ȯ
*/
inline
bool operator ==(const GVector &lhs, const GVector &rhs)
{
	assert(lhs.N == rhs.N);
	double error = GVector::Precision;
	for (int i = 0; i != lhs.N; ++i)
		if (!EQ(lhs.V[i], rhs.V[i], error))
			return false;
	return true;
}

/*!
*	\brief �ε�ȣ������(!=)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ���� ����� ��ȯ
*/
inline
bool operator !=(const GVector &lhs, const GVector &rhs)
{
	assert(lhs.N == rhs.N);
	double error = GVector::Precision;
	for (int i = 0; i != lhs.N; ++i)
		if (!EQ(lhs.V[i], rhs.V[i], error))
			return true;
	return false;
}

/*!
*	\brief ÷�ڿ�����([])
*
*	\param idx ÷��
*
*	\return ������ \a idx ��° ���Ҹ� ��ȯ
*/
inline
double &GVector::operator [](const int &idx)
{
	assert(idx >=0 && idx < N);
	return V[idx];
}

/*!
*	\brief �����ü�� ÷�ڿ�����([])
*
*	\param idx ÷��
*
*	\return ������ \a idx ��° ���Ҹ� ��ȯ
*/
inline
const double &GVector::operator [](const int &idx) const
{
	assert(idx >=0 && idx < N);
	return V[idx];
}

/*!
*	\brief ������ ���Ҹ� �����Ѵ�.
*	\warning �Ű������� �Ǽ������� �־�� �Ѵ�. ��) 1: �Ұ���, 1.0: ����.
*
*	\param x ��ǥ
*
*	\return ������ �ڽ��� ��ȯ�Ѵ�.
*/
inline
GVector &GVector::Set(double x, ...)
{
	V[0] = x;
	va_list ap;
	va_start(ap, x);
	for (int i = 1; i != N; ++i)
		V[i] = va_arg(ap, double);
	va_end(ap);
	return *this;
}

/*!
*	\brief ������ ���Ҹ� �����Ѵ�.
*	\warning �Ű������� ���������� �־�� �Ѵ�. ��) 1: ����, 1.0: �Ұ���.
*
*	\param x ��ǥ
*
*	\return ������ �ڽ��� ��ȯ�Ѵ�.
*/
inline
GVector &GVector::Set(int x, ...)
{
	V[0] = (double)x;
	va_list ap;
	va_start(ap, x);
	for (int i = 1; i != N; ++i)
		V[i] = (double)va_arg(ap, int);
	va_end(ap);
	return *this;
}

/*!
*	\brief ������ ���Ҹ� �����Ѵ�.
*
*	\param pVal ������ ���Ҹ� ������ N���� �Ǽ� �迭
*
*	\return ������ �ڽ��� ��ȯ�Ѵ�.
*/
inline
GVector &GVector::Set(double *pVal)
{
	memcpy(V, pVal, sizeof(double) * N);
	return *this;
}

/*!
*	\brief ���͸� ����ȭ�Ѵ�.
*
*	\return ����ȭ�� �ڽ��� ��ȯ�Ѵ�.
*/
inline
GVector &GVector::Normalize()
{
	double len = norm(*this);
	for (int i = 0; i != N; ++i)
		V[i] /= len;
	return *this;
}

/*!
*	\brief �����ͷ� �����Ѵ�.
*
*	\return �ڽ��� ��ȯ�Ѵ�.
*/
inline
GVector &GVector::SetZeros()
{
	for (int i = 0; i != N; ++i)
		V[i] = 0.0;
	return *this;
}

/*!
*	\brief ������ ������ ��ȯ�Ѵ�.
*
*	\return ������ ������
*/
inline
int GVector::GetDim() const
{
	return N;
}

/*!
*	\brief �񱳸� ���� ������ �Ѱ踦 �����Ѵ�.
*
*	\param error ������ �Ѱ�
*/
inline
void GVector::SetPrecision(double error)
{
	Precision = error;
}

/*!
*	\breief �񱳸� ���� ������ �Ѱ谪�� ��ȯ�Ѵ�.
*
*	\return �񱳸� ���� ������ �Ѱ谪
*/
inline
double GVector::GetPrecision()
{
	return Precision;
}

/*!
*	\brief ���������(*)
*
*	\param lhs ���� �ǿ�����
*	\param s ���
*	\return ������ ����� ��ȯ
*/
inline
GVector operator *(const GVector &lhs, const double &s)
{
	GVector ret(lhs.N);
	for (int i = 0; i != lhs.N; ++i)
		ret.V[i] = lhs.V[i] * s;
	return ret;
}

/*!
*	\brief ���������(*)
*
*	\param s ���
*	\param rhs ���� �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GVector operator *(const double &s, const GVector &rhs)
{
	GVector ret(rhs.N);
	for (int i = 0; i != rhs.N; ++i)
		ret.V[i] = rhs.V[i] * s;
	return ret;
}

/*!
*	\brief ��¿�����(<<)
*
*	\param os ��½�Ʈ��
*	\param v ����� ����
*
*	\return \a v�� ��µ� ��Ʈ���� ��ȯ
*/
inline
ostream &operator <<(ostream &os, const GVector &v)
{
	for (int i = 0; i != v.N; ++i)
		if (i != v.N -1)
			os << "| " << setw(10) << v.V[i] << " |" << endl;
		else
			os << "| " << setw(10) << v.V[i] << " |";
	return os;
}

/*!
*	\brief �Է¿�����(>>)
*
*	\param is �Է½�Ʈ��
*	\param v �Է°��� ����� ����
*
*	\return �Է°��� ���ŵ� �Է½�Ʈ��
*/
inline
istream &operator >>(istream &is, GVector &v)
{
	for (int i = 0; i != v.N; ++i)
		is >> v.V[i];
	return is;
}

/*!
*	\brief ���� v�� ũ�⸦ ���Ѵ�.
*
*	\param v ����
*
*	\return ���� v�� ũ��
*/
inline
double norm(const GVector &v)
{
	double len = 0;
	for (int i = 0; i != v.N; ++i)
		len += SQR(v.V[i]);
	len = SQRT(len);
	return len;
}

/*!
*	\brief �� ������ ���̺����� ũ�⸦ ���Ѵ�.
*
*	\param v ����
*	\param w ����
*
*	\return �� ������ ���̺����� ũ��
*/
inline
double dist(const GVector &v, const GVector &w)
{
	return norm(v - w);
}

/*!
*	\class GPoint3
*	\brief 3���� ����Ʈ�ڷ����� ��Ÿ���� Ŭ����
*
*	\author ������ (shyun@dongguk.edu)
*	\date 01 Jan 2001	
*/
class GPoint3
{
	// ������ �Լ� �� Ŭ����
	__host__ __device__ friend GVector3 operator -(const GPoint3 &lhs, const GPoint3 &rhs);
	__host__ __device__ friend GPoint3 operator -(const GPoint3 &lhs, const GVector3 &rhs);
	__host__ __device__ friend GPoint3 operator +(const GPoint3 &lhs, const GVector3 &rhs);
	__host__ __device__ friend GPoint3 operator +(const GVector3 &lhs, const GPoint3 &rhs);
	__host__ __device__ friend bool operator ==(const GPoint3 &lhs, const GPoint3 &rhs);
	__host__ __device__ friend bool operator !=(const GPoint3 &lhs, const GPoint3 &rhs);
	__host__ __device__ friend ostream &operator <<(ostream &os, const GPoint3 &p);
	__host__ __device__ friend istream &operator >>(istream &is, GPoint3 &p);
	__host__ __device__ friend double norm(const GPoint3 &p);
	__host__ __device__ friend double dist(const GPoint3 &p, const GPoint3 &q);
	__host__ __device__ friend GPoint3 barycentric_combination(GPoint3 *Points, double *Weights, const int Size);
	__host__ __device__ friend GVector3 cast_vec3(const GPoint3 &p);
	__host__ __device__ friend GVector  cast_vec(const GPoint3 &p);
	__host__ __device__ friend class GLScene;

public:
	// ������ �� �Ҹ���
	__host__ __device__ GPoint3(double x = 0.0, double y = 0.0, double z = 0.0);
	__host__ __device__ GPoint3(const GPoint3 &cpy);
	__host__ __device__ virtual ~GPoint3();

	// ���Կ�����
	__host__ __device__ GPoint3 &operator =(const GPoint3 &rhs);

	// ÷�ڿ�����
	__host__ __device__ double &operator [](const int &idx);
	__host__ __device__ const double &operator [](const int &idx) const;

	// ����Լ�
	__host__ __device__ GPoint3 &Set(const double &x, const double &y, const double &z);
	
	// �����Լ�
	__host__ __device__ static void SetPrecision(double error);
	__host__ __device__ static double GetPrecision();
   
protected:
	// ������ ���
	double V[3];	/*! \breif 3���� ������ ���Ҹ� �����ϴ� �Ǽ��迭 */	
	static double Precision;	/*! \breif ��ȣ �� �ε�ȣ�������� �����Ѱ� */
};

/*!
*	\brief ����Ʈ ������
*	
*	\param x x-��ǥ
*	\param y y-��ǥ
*	\param z z-��ǥ
*/
inline
GPoint3::GPoint3(double x, double y, double z)
{
	V[0] = x;
	V[1] = y;
	V[2] = z;
}

/*!
*	\brief ���������
*
*	\param cpy ����� ��ü
*
*	\return ����� �ڽ��� ��ȯ
*/
inline
GPoint3::GPoint3(const GPoint3 &cpy)
{
	V[0] = cpy.V[0];
	V[1] = cpy.V[1];
	V[2] = cpy.V[2];
}

/*!
*	\brief  �Ҹ���
*/
inline
GPoint3::~GPoint3()
{
}

/*!
*	\brief ���Կ�����
*
*	\param rhs ������ �ǿ�����
*
*	\return ���Ե� �ڽ��� ��ȯ
*/
inline
GPoint3 &GPoint3::operator =(const GPoint3 &rhs)
{
	V[0] = rhs.V[0];
	V[1] = rhs.V[1];
	V[2] = rhs.V[2];
	return *this;
}

/*!
*	\brief ���������(-)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ���(����)�� ��ȯ
*/
inline
GVector3 operator -(const GPoint3 &lhs, const GPoint3 &rhs)
{
	return GVector3(lhs.V[0] - rhs.V[0], lhs.V[1] - rhs.V[1], lhs.V[2] - rhs.V[2]);
}

/*!
*	\brief ���������(-)
*
*	\param lhs ���� �ǿ�����(����Ʈ)
*	\param rhs ������ �ǿ�����(����)
*
*	\return ������ ���(����Ʈ)�� ��ȯ
*/
inline
GPoint3 operator -(const GPoint3 &lhs, const GVector3 &rhs)
{
	return GPoint3(lhs.V[0] - rhs[0], lhs.V[1] - rhs[1], lhs.V[2] - rhs[2]);
}

/*!
*	\brief ���������(+)
*
*	\param lhs ���� �ǿ�����(����Ʈ)
*	\param rhs ������ �ǿ�����(����)
*
*	\return ������ ���(����Ʈ)�� ��ȯ
*/
inline
GPoint3 operator +(const GPoint3 &lhs, const GVector3 &rhs)
{
	return GPoint3(lhs.V[0] + rhs[0], lhs.V[1] + rhs[1], lhs.V[2] + rhs[2]);
}

/*!
*	\brief ���������(+)
*
*	\param lhs ���� �ǿ�����(����)
*	\param rhs ������ �ǿ�����(����Ʈ)

*
*	\return ������ ���(����Ʈ)�� ��ȯ
*/
inline
GPoint3 operator +(const GVector3 &lhs, const GPoint3 &rhs)
{
	return GPoint3(lhs.V[0] + rhs[0], lhs.V[1] + rhs[1], lhs.V[2] + rhs[2]);
}

/*!
*	\brief ��ȣ������
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return �񱳰���� ��ȯ
*/
inline
bool operator ==(const GPoint3 &lhs, const GPoint3 &rhs)
{
	double error = GPoint3::Precision;
	return (EQ(lhs.V[0], rhs.V[0], error) && EQ(lhs.V[1], rhs.V[1], error) && EQ(lhs.V[2], rhs.V[2], error));
}

/*!
*	\brief �ε�ȣ������
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return �񱳰���� ��ȯ
*/
inline
bool operator !=(const GPoint3 &lhs, const GPoint3 &rhs)
{
	double error = GPoint3::Precision;
	return (!EQ(lhs.V[0], rhs.V[0], error) || !EQ(lhs.V[1], rhs.V[1], error) || !EQ(lhs.V[2], rhs.V[2], error));
}

/*!
*	\brief ÷�ڿ�����([])
*
*	\param idx ÷��
*
*	\return \a idx ��° ���Ҹ� ��ȯ
*/
inline
double &GPoint3::operator [](const int &idx)
{
	assert(idx >= 0 && idx < 3);
	return V[idx];
}

/*!
*	\brief �����ü�� ÷�ڿ�����([])
*
*	\param idx ÷��
*
*	\return \a idx ��° ���Ҹ� ��ȯ
*/
inline
const double &GPoint3::operator [](const int &idx) const
{
	assert(idx >= 0 && idx < 3);
	return V[idx];
}

/*!
*	\brief ����Ʈ�� ��ǥ�� �����Ѵ�.
*
*	\param x x-��ǥ
*	\param y y-��ǥ
*	\param z z-��ǥ
*
*	\return ������ �ڽ��� ��ȯ
*/
inline
GPoint3 &GPoint3::Set(const double &x, const double &y, const double &z)
{
	V[0] = x;
	V[1] = y;
	V[2] = z;
	return *this;
}

/*!
*	\brief ��ȣ �� �ε�ȣ �������� ������ �Ѱ� ����
*
*	\param error ������ �Ѱ�
*/
inline
void GPoint3::SetPrecision(double error)
{
	Precision = error;
}

/*!
*	\brief ������ ������ �Ѱ踦 ��ȯ
*
*	\return ������ ������ �Ѱ�
*/
inline
double GPoint3::GetPrecision()
{
	return Precision;
}

/*!
*	\brief ��¿�����(<<)
*
*	\param os ��½�Ʈ��
*	\param p ����� ��ü
*
*	\return ��µ� ��Ʈ�� ��ü�� ��ȯ
*/
inline
ostream &operator <<(ostream &os, const GPoint3 &p)
{
	os << "(" << setw(5) << p.V[0] << ", " << setw(5) << p.V[1] << ", " << setw(5) << p.V[2] << ")";
	return os;
}

/*!
*	\brief �Է¿�����(>>)
*
*	\param is �Է½�Ʈ��
*	\param v �Է°��� ����� ����
*
*	\return �Է°��� ���ŵ� �Է½�Ʈ��
*/
inline
istream &operator >>(istream &is, GPoint3 &p)
{
	is >> p.V[0] >> p.V[1] >> p.V[2];
	return is;
}

/*!
*	\brief �� \a p�� ���̸� ���Ѵ�.
*
*	\param p ��
*
*	\return �� \a p�� ����.
*/
inline
double norm(const GPoint3 &p)
{
	return SQRT(SQR(p.V[0]) + SQR(p.V[1]) + SQR(p.V[2]));
}

/*!
*	\brief �� \a p���� �� \a q������ �Ÿ��� ���Ѵ�.
*
*	\param p ��
*	\param q ��
*
*	\return �� \a p���� �� \a q������ �Ÿ�
*/
inline
double dist(const GPoint3 &p, const GPoint3 &q)
{
	return SQRT(SQR(p.V[0] - q.V[0]) + SQR(p.V[1] - q.V[1]) + SQR(p.V[2] - q.V[2]));
}

/*!
*	\brief �� pt�� ��ǥ�� �������·� ��ȯ�Ѵ�.
*
*	\param pt 3���� ������ ��
*
*	\return pt�� 3���� ���� ǥ��
*/
inline
GVector3 cast_vec3(const GPoint3 &pt)
{
	return GVector3(pt[0], pt[1], pt[2]);
}

/*!
*	\brief �� pt�� ��ǥ�� �������·� ��ȯ�Ѵ�.
*
*	\param pt 3���� ������ ��
*
*	\return pt�� ���� ǥ��
*/
inline
GVector cast_vec(const GPoint3 &pt)
{
	return GVector(3, pt[0], pt[1], pt[2]);
}

/*!
*	\brief ���� v�� ��ǥ�� ����Ʈ ���·� ��ȯ�Ѵ�.
*
*	\param v 3���� ������ ����
*
*	\return v�� ��ǥ�� ����Ʈ ���·� ��ȯ
*/
inline
GPoint3 cast_pt3(const GVector3 &v)
{
	return GPoint3(v[0], v[1], v[2]);
}

/*!
*	\brief 3���� ���� v�� ��ǥ�� �Ϲ����� ���� ���·� ��ȯ�Ѵ�.
*
*	\param v 3���� ������ ����
*
*	\return 3���� ���� v�� ��ǥ�� �Ϲ����� ���� ���·� ��ȯ�Ѵ�.
*/
inline
GVector cast_vec(const GVector3 &v)
{
	return GVector(3, v[0], v[1], v[2]);
}

/*!
*	\brief ������ barycentric combination�� ���Ѵ�
*
*	\param Points ���� �迭�� ���� ������
*	\param Weights combination�� �����
*	\param Size ���� ���� (Points, Weights �迭�� ũ��)
*
*	\return ������ barycentric combination�� ��� (= Point[0] * Weight[0] + Points[1] * Weight[1] + ... + Points[Size-1] * Weight[Size-1])
*/
inline
GPoint3 barycentric_combination(GPoint3 *Points, double *Weights, const int Size)
{
	GPoint3 ret;
	for (int i = 0; i < Size; i++)
	{	
		ret.V[0] += Points[i][0] * Weights[i];
		ret.V[1] += Points[i][1] * Weights[i];
		ret.V[2] += Points[i][2] * Weights[i];
	}
	return ret;
}

/*!
*	\class GMatrix
*	\brief \a m �� \a n ���� ũ�⸦ ���� ����� ǥ���ϴ� Ŭ����
*
*	\author ������ (shyun@dongguk.edu)
*	\date 01 Jan 2001
*/
class GMatrix
{
	// ������ �Լ� �� Ŭ����
	friend GMatrix operator +(const GMatrix &lhs, const GMatrix &rhs);
	friend GMatrix operator -(const GMatrix &lhs, const GMatrix &rhs);
	friend GMatrix operator *(const GMatrix &lhs, const GMatrix &rhs);
	friend GMatrix operator /(const GMatrix &lhs, const double &s);
	friend bool operator ==(const GMatrix &lhs, const GMatrix &rhs);
	friend bool operator !=(const GMatrix &lhs, const GMatrix &rhs);
	friend GVector operator *(const GMatrix &lhs, const GVector &rhs);
	friend GMatrix operator *(const GVector &lhs, const GMatrix &rhs);
	friend GMatrix operator *(const GMatrix &lhs, const double &s);
	friend GMatrix operator *(const double &s, const GMatrix &rhs);
	friend ostream &operator <<(ostream &os, const GMatrix &m);
	friend GMatrix tr(const GMatrix &m);
	friend GMatrix ref(const GMatrix &m);
	friend GMatrix rref(const GMatrix &m);
	friend double *cast_arr(const GMatrix &m);
	friend int grank(const GMatrix &m);
	friend int nullity(const GMatrix &m);
	friend double det(const GMatrix &A);
	friend GMatrix inv(const GMatrix &A);
	friend int dcmp_lu(const GMatrix &A, GMatrix &L, GMatrix &U);
	friend int dcmp_lu(GMatrix &A, GVector &Idx, double &d);
	friend int dcmp_qr(const GMatrix &A, GMatrix &Q, GMatrix &R);
	friend void dcmp_sv(const GMatrix &A, GMatrix &U, GVector &S, GMatrix &V);
	friend void lin_sys_lu(const GMatrix &A, GVector &x, const GVector &b, const GVector &Idx);
	friend double lin_sys_sv(const GMatrix &A, GVector &x, const GVector &b);
	friend double lin_sys_sv(const GMatrix &U, const GVector &S, const GMatrix &V, GVector &x, const GVector &b);
	friend int lin_sys_cg(const GMatrix &A, GVector &x, const GVector &b, int iter);
	friend int lin_sys_qr(const GMatrix &A, GVector &x, const GVector &b);
	friend int eigen_sys(const GMatrix &A, GVector &D, GMatrix &V);
	friend void pca(GMatrix &Data, GMatrix &PC, GVector &SigVal, GVector &Mean);

public:
	// ������ �� �Ҹ���
	GMatrix(int row = 4, int col = 4, double *elem = NULL);
	GMatrix(const GMatrix &cpy);
	virtual ~GMatrix();

	// ���Կ����� �� ���մ��Կ�����
	GMatrix &operator =(const GMatrix &rhs);
	GMatrix &operator +=(const GMatrix &rhs);
	GMatrix &operator -=(const GMatrix &rhs);
	GMatrix &operator *=(const GMatrix &rhs);
	GMatrix &operator *=(const double &s);
	GMatrix &operator /=(const double &s);

	// ���׿�����
	GMatrix operator +() const;
	GMatrix operator -() const;

	// ÷�ڿ�����
	double *operator [](const int idx);
	const double *operator [](const int idx) const;

	// ����Լ�
	GMatrix &SetTranspose();
	GMatrix &SetIdentity();
	GMatrix &SetZeros();
	GMatrix &SetRowVec(const int idx, const GVector &v);
	GMatrix &SetColVec(const int idx, const GVector &v);
	GMatrix &ExchangeRows(const int idx0, const int idx1);
	GMatrix &ExchangeCols(const int idx0, const int idx1);
	int GetRowNum() const;
	int GetColNum() const;
	GVector GetRowVec(const int idx) const;
	GVector GetColVec(const int idx) const;
	bool IsSquare() const;

	// �����Լ�
	static void SetPrecision(double error);
	static double GetPrecision();

protected:
	// ������ ���
	int r;		// �຤���� ����
	int c;		// �������� ����
	double *M;	// ������ �迭
	static double Precision;	// ������ �Ѱ�
};


/*!
*	\brief ������
*
*	\param row �຤���� ����
*	\param col �������� ����
*	\param elem ����� ����
*/
inline
GMatrix::GMatrix(int row, int col, double *elem)
{
	r = row;
	c = col;
	M = new double [r * c];
	if (elem)
		memcpy(M, elem, sizeof(double) * r * c);
	else
		memset(M, 0, sizeof(double) * r * c);
}

/*!
*	\brief ���������
*
*	\param ����� ��ü
*/
inline
GMatrix::GMatrix(const GMatrix &cpy)
{
	r = cpy.r;
	c = cpy.c;
	M = new double [r *c];
	memcpy(M, cpy.M, sizeof(double) * r * c);
}

/*!
*	\brief �Ҹ���
*/
inline
GMatrix::~GMatrix()
{
	if (M)
		delete [] M;
	M = NULL;
}

/*!
*	\brief ���Կ�����
*
*	\param rhs ���Ե� ������ �ǿ�����
*
*	\return ���Ե� �ڽ��� ��ȯ
*/
inline
GMatrix &GMatrix::operator =(const GMatrix &rhs)
{
	if (M)
		delete [] M;

	r = rhs.r;
	c = rhs.c;
	M = new double [r * c];

	memcpy(M, rhs.M, sizeof(double) * r * c);
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param rhs ������ �ǿ�����
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GMatrix &GMatrix::operator +=(const GMatrix &rhs)
{
	assert(r == rhs.r && c == rhs.c);
	for (int i = 0; i < r * c; ++i)
		M[i] += rhs.M[i];
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param rhs ������ �ǿ�����
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GMatrix &GMatrix::operator -=(const GMatrix &rhs)
{
	assert(r == rhs.r && c == rhs.c);
	for (int i = 0; i < r * c; ++i)
		M[i] -= rhs.M[i];
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*	\note ���׹߰� (M[i * c + k] --> M[i * rhs.r + k]) 2010. 6. 15.
*
*	\param rhs ������ �ǿ�����
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GMatrix &GMatrix::operator *=(const GMatrix &rhs)
{
	assert(c == rhs.r);

	c = rhs.c;
	double *newM = new double [r * c];
	memset(newM, 0, sizeof(double) * r * c);

	for (int i = 0; i != r; ++i)
		for (int j = 0; j != c; ++j)
			for (int k = 0; k != rhs.r; ++k)
				newM[i * c + j] += M[i * rhs.r + k] * rhs.M[k * rhs.c + j];

	delete [] M;
	M = newM;

	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param s ������ �ǿ�����(���)
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GMatrix &GMatrix::operator *=(const double &s)
{
	for (int i = 0; i < r * c; i++)
		M[i] *= s;
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param s ������ �ǿ�����(���)
*
*	\return ������ ���Ե� �ڽ�
*/
inline
GMatrix &GMatrix::operator /=(const double &s)
{
	for (int i = 0; i < r * c; i++)
		M[i] /= s;
	return *this;
}

/*!
*	\brief ���׿����� (+)
*
*	\return ���� ��ȣ �� ���� ���� ���� ��ü�� ��ȯ
*/
inline
GMatrix GMatrix::operator +() const
{
	return *this;
}

/*!
*	\brief ���׿����� (-)
*
*	\return �ݴ��ȣ�� ���� ���� ���� ��ü�� ��ȯ
*/
inline
GMatrix GMatrix::operator -() const
{
	return *this * -1;
}

/*!
*	\brief ���������(+)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GMatrix operator +(const GMatrix &lhs, const GMatrix &rhs)
{
	assert(lhs.r == rhs.r && lhs.c == rhs.c);
	GMatrix ret(lhs.r, lhs.c);
	ret += rhs;	
	return ret;
}

/*!
*	\brief ���������(-)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GMatrix operator -(const GMatrix &lhs, const GMatrix &rhs)
{
	assert(lhs.r == rhs.r && lhs.c == rhs.c);
	GMatrix ret(lhs);
	ret -= rhs;	
	return ret;
}

/*!
*	\brief ���������(*)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GMatrix operator *(const GMatrix &lhs, const GMatrix &rhs)
{
	assert(lhs.c == rhs.r);
	GMatrix ret(lhs);
	ret *= rhs;	
	return ret;
}

/*!
*	\brief ���������(/)
*
*	\param lhs ���� �ǿ�����
*	\param s ���
*
*	\return ������ ����� ��ȯ
*/
inline
GMatrix operator /(const GMatrix &lhs, const double &s)
{
	GMatrix ret(lhs);
	ret /= s;
	return ret;
}

/*!
*	\brief ��ȣ������(==)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ���� ����� ��ȯ
*/
inline
bool operator ==(const GMatrix &lhs, const GMatrix &rhs)
{
	assert(lhs.r == rhs.r && lhs.c == rhs.c);
	double error = GMatrix::Precision;
	for (int i = 0; i != lhs.r * lhs.c; ++i)
		if (!EQ(lhs.M[i], rhs.M[i], error))
			return false;
	return true;
}

/*!
*	\brief �ε�ȣ������(!=)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ���� ����� ��ȯ
*/
inline
bool operator !=(const GMatrix &lhs, const GMatrix &rhs)
{
	assert(lhs.r == rhs.r && lhs.c == rhs.c);
	double error = GMatrix::Precision;
	for (int i = 0; i != lhs.r * lhs.c; ++i)
		if (EQ(lhs.M[i], rhs.M[i], error))
			return true;
	return false;
}

/*!
*	\brief ÷�ڿ�����([])
*
*	\param idx ���� ÷��
*
*	\return \a idx ���� �迭�� ��ȯ
*/
inline
double *GMatrix::operator [](const int idx)
{
	assert(idx >= 0 && idx < r);
	return &M[idx * c];
}

/*!
*	\brief �����ü�� ÷�ڿ�����([])
*
*	\param idx ���� ÷��
*
*	\return \a idx ���� �迭�� ��ȯ
*/
inline
const double *GMatrix::operator [](const int idx) const
{
	assert(idx >= 0 && idx < r);
	return &M[idx * c];
}

/*!
*	\brief ����� ��ġ��ķ� ��ȯ�Ѵ�.
*
*	\return ��ȯ�� �ڽ��� ��ȯ
*/
inline
GMatrix &GMatrix::SetTranspose()
{
	int i, j;
	if (r == c)	// Square matrix
	{
		for (i = 0; i != r; ++i)
			for (j = i + 1; j != c; ++j)
				SWAP(double, M[i * c + j], M[j * c + i]);
	}
	else		// Rectangular matrix
	{ 
		double *buffer = new double [r * c];
		memcpy(buffer, M, sizeof(double) * r * c);
		SWAP(int, r, c);
		for (i = 0; i != r; ++i)
			for (j = 0; j != c; ++j)
				M[i * c + j] = buffer[j * r + i];
		delete [] buffer;
	}
	return *this;
}

/*!
*	\brief ������ķ� ��ȯ�Ѵ�.
*
*	\return ��ȯ�� �ڽ��� �����Ѵ�.
*/
inline
GMatrix &GMatrix::SetIdentity()
{
	memset(M, 0, sizeof(double) * r * c);
	int min = MIN(r, c);
	for (int i = 0; i < min; i++)
		M[i * c + i] = 1.0;
	return *this;
}

/*!
*	\brief ����ķ� ��ȯ�Ѵ�.
*
*	\return ��ȯ�� �ڽ��� �����Ѵ�.
*/
inline
GMatrix &GMatrix::SetZeros()
{
	memset(M, 0, sizeof(double) * r * c);
	return *this;
}

/*!
*	\brief idx ��°�� �຤�͸� v�� �����Ѵ�.
*
*	\param idx �຤���� �ε���
*	\param v ������ ����
*
*	\return ��ȯ�� �ڽ��� �����Ѵ�.
*/
inline
GMatrix &GMatrix::SetRowVec(const int idx, const GVector &v)
{
	assert(idx < r);
	assert(v.N == c);
	for (int i = 0; i < c; i++)
		M[idx * c + i] = v.V[i];
	return *this;
}

/*!
*	\brief idx ��°�� �����͸� v�� �����Ѵ�.
*
*	\param idx �������� �ε���
*	\param v ������ ����
*
*	\return ��ȯ�� �ڽ��� �����Ѵ�.
*/
inline
GMatrix &GMatrix::SetColVec(const int idx, const GVector &v)
{
	assert(idx < c);
	assert(v.N == r);
	for (int i = 0; i < r; i++)
		M[i * c + idx] = v.V[i];
	return *this;
}

/*!
*	\brief idx0��° �຤�Ϳ� idx1��° �຤�͸� ��ȯ�Ѵ�.
*
*	\param idx0 �຤���� �ε���
*	\param idx1 �຤���� �ε���
*
*	\return ��ȯ�� �ڽ��� �����Ѵ�.
*/
inline
GMatrix &GMatrix::ExchangeRows(const int idx0, const int idx1)
{
	GVector tmp(c);
	tmp = GetRowVec(idx0);
	SetRowVec(idx0, GetRowVec(idx1));
	SetRowVec(idx1, tmp);
	return *this;
}
/*!
*	\brief idx0��° �����Ϳ� idx1��° �����͸� ��ȯ�Ѵ�.
*
*	\param idx0 �������� �ε���
*	\param idx1 �������� �ε���
*
*	\return ��ȯ�� �ڽ��� �����Ѵ�.
*/
inline
GMatrix &GMatrix::ExchangeCols(const int idx0, const int idx1)
{
	GVector tmp(r);
	tmp = GetColVec(idx0);
	SetColVec(idx0, GetColVec(idx1));
	SetColVec(idx1, tmp);
	return *this;
}

/*!
*	\brief ���� ���� �����Ѵ�.
*
*	\return ���� ��
*/
inline
int GMatrix::GetRowNum() const
{
	return r;
}

/*!
*	\brief ���� ���� �����Ѵ�.
*
*	\return ���� ��
*/
inline
int GMatrix::GetColNum() const
{
	return c;
}

/*!
*	\brief �຤�͸� ��ȯ�Ѵ�.
*
*	\param idx �຤���� �ε���
*
*	\return idx ��° �຤�͸� ��ȯ�Ѵ�.
*/
inline
GVector GMatrix::GetRowVec(const int idx) const
{
	assert(idx < r);
	GVector ret(c);
	for (int i = 0; i < c; i++)
		ret.V[i] = M[idx * c + i];
	return ret;
}

/*!
*	\brief �����͸� ��ȯ�Ѵ�.
*
*	\param idx �������� �ε���
*
*	\return idx ��° �����͸� ��ȯ�Ѵ�.
*/
inline
GVector GMatrix::GetColVec(const int idx) const
{
	assert(idx < c);
	GVector ret(r);
	for (int i = 0; i < r; i++)
		ret.V[i] = M[i * c + idx];
	return ret;
}

/*!
*	\brief ������� (square matrix)������ �����Ѵ�.
*
*	\return ��������� ��� true, ���簢����� ��� false.
*/
inline
bool GMatrix::IsSquare() const
{
	return (r == c) ? true : false;
}

/*!
*	\brief Set ��ȣ �� �ε�ȣ�������� ��Ȯ���� �����Ѵ�.
*
*	\param error ������ �Ѱ�
*/
inline
void GMatrix::SetPrecision(double error)
{
	Precision = error;
}

/*!
*	\brief ������ ������ �Ѱ踦 ��ȯ�Ѵ�.
*
*	\return ������ �Ѱ�
*/
inline
double GMatrix::GetPrecision()
{
	return Precision;
}

/*!
*	\brief ��������� (*)
*
*	\param lhs ���� �ǿ����� (���)
*	\param rhs ������ �ǿ����� (����)
*
*	\return ����� ����� ��ȯ
*/
inline
GVector operator *(const GMatrix &lhs, const GVector &rhs)
{
	assert(lhs.c == rhs.N);
	GVector ret(lhs.r);
	for (int i = 0; i != lhs.r; ++i)		// for each row.
		for (int j = 0; j != lhs.c; ++j)	// for each col.
			ret.V[i] += lhs.M[i * lhs.c + j] * rhs.V[j];
	return ret;
}

/*!
*	\brief ��������� (*)
*
*	\param lhs ���� �ǿ����� (����)
*	\param rhs ������ �ǿ����� (���)
*
*	\return ����� ����� ��ȯ
*/
inline
GMatrix operator *(const GVector &lhs, const GMatrix &rhs)
{
	assert(rhs.r == 1);
	GMatrix ret(lhs.N, rhs.c);
	for (int i = 0; i != lhs.N; ++i)		// for each row.
		for (int j = 0; j != rhs.c; ++j)	// for each col.
			ret.M[i * rhs.c + j] = lhs.V[i] * rhs.M[j];
	return ret;
}

/*!
*	\brief ��������� (*)
*
*	\param lhs ���� �ǿ����� (���)
*	\param rhs ������ �ǿ����� (���)
*
*	\return ����� ����� ��ȯ
*/
inline
GMatrix operator *(const GMatrix &lhs, const double &s)
{
	GMatrix ret(lhs);
	ret *= s;
	return ret;
}

/*!
*	\brief ��������� (*)
*
*	\param lhs ���� �ǿ����� (���)
*	\param rhs ������ �ǿ����� (���)
*
*	\return ����� ����� ��ȯ
*/
inline
GMatrix operator *(const double &s, const GMatrix &rhs)
{
	GMatrix ret(rhs);
	ret *= s;
	return ret;
}

/*!
*	\brief ��¿�����(<<)
*
*	\param os ��½�Ʈ��
*	\param m ����� ��ü
*
*	\return ��µ� ��Ʈ�� ��ü�� ��ȯ
*/
inline
ostream &operator <<(ostream &os, const GMatrix &m)
{
	for (int i = 0; i != m.r; i++)
	{
		os << "|";
		for (int j = 0; j != m.c; j++)
			os << setw(5) << m.M[i * m.c + j] << " ";
		os << "|" << endl;
	}
	return os;
}

/*!
*	\brief ��ġ����� ��ȯ�Ѵ�.
*
*	\param m ���
*
*	\return ��� \a m�� ��ġ���
*/
inline GMatrix tr(const GMatrix &m)
{
	GMatrix ret(m);
	ret.SetTranspose();
	return ret;
}

/*!
*	\brief ����� row-echelon ���� ��ȯ�Ѵ�.
*
*	\param m ���
*
*	\return row-echelon ���� ��ȯ�Ѵ�.
*/
inline GMatrix ref(const GMatrix &m)
{
	int i, j, k; // �ݺ����� ����
	int r = m.GetRowNum();
	int c = m.GetColNum();
	int n = MIN(r, c);
	GMatrix T(m);

	int shift = 0;
	for (i = 0; i < n; i++)
	{
		// pivoting.
		double maxi = ABS(T[i][i + shift]);
		int pivot_idx = i;
		for (j = i + 1; j < r; j++)
		{
			if (maxi < ABS(T[j][i + shift]))
			{
				maxi = ABS(T[j][i + shift]);
				pivot_idx = j;
			}
		}

		if (EQ_ZERO(maxi, GMatrix::Precision))
		{
			shift++;
			i--;
			continue;
		}

		if (i != pivot_idx)
			T.ExchangeRows(i, pivot_idx);          

		double s = T[i][i + shift];
		for (j = i + shift; j < c; j++)
			T[i][j] = T[i][j] / s;

		for (j = i + 1; j < r; j++)
		{
			s = T[j][i + shift];
			for (k = i + shift; k < c; k++)
			{
				T[j][k] = T[j][k] - s * T[i][k];
			}
		}
	}

	return T;
}

/*!
*	\brief ����� reduced row-echelon ���� ��ȯ�Ѵ�.
*
*	\param m ���
*
*	\return reduced row-echelon ���� ��ȯ�Ѵ�.
*/
inline GMatrix rref(const GMatrix &m)
{
	int i, j, k; // �ݺ����� ����
	int r = m.GetRowNum();
	int c = m.GetColNum();
	int n = MIN(r, c);
	GMatrix T(m);

	int shift = 0;
	for (i = 0; i < n; i++)
	{
		// pivoting.
		double maxi = ABS(T[i][i + shift]);
		int pivot_idx = i;
		for (j = i + 1; j < r; j++)
		{
			if (maxi < ABS(T[j][i + shift]))
			{
				maxi = ABS(T[j][i + shift]);
				pivot_idx = j;
			}
		}

		if (EQ_ZERO(maxi, GMatrix::Precision))
		{
			shift++;
			i--;
			continue;
		}

		if (i != pivot_idx)
			T.ExchangeRows(i, pivot_idx);          

		double s = T[i][i + shift];
		for (j = i + shift; j < c; j++)
			T[i][j] = T[i][j] / s;

		for (j = 0; j < r; j++)
		{
			if (i == j)
				continue;

			s = T[j][i + shift];
			for (k = i + shift; k < c; k++)
			{
				T[j][k] = T[j][k] - s * T[i][k];
			}
		}
	}

	return T;
}

/*!
*	\brief ����� ���Ҹ� ���� �Ǽ��迭�� ���Ѵ�.
*
*	\param m ���
*
*	\return ����� ���Ҹ� ������ �Ǽ� �迭
*/
inline
double *cast_arr(const GMatrix &m)
{
	return m.M;
}

/*!
*	\brief ����� grank (������ ������ Ȥ�� �຤���� ��)�� ���Ѵ�.
*
*	\param m ���
*
*	\return ������ ����� rank.
*/
inline 
int grank(const GMatrix &m)
{
	int i, r, rank = 0;
	r = m.GetRowNum();

	GMatrix T = ref(m);
	for (i = 0; i < r; i++)
	{
		GVector rVec = T.GetRowVec(i);
		if (!EQ_ZERO(norm(rVec), rVec.GetPrecision()))
			rank++;
	}

	return rank;
}

/*!
*	\brief ����� nullity �� ���Ѵ�.
*
*	\param m ���
*
*	\return ������ ����� nullity.
*/
inline 
int nullity(const GMatrix &m)
{
	int rnk = grank(m);
	int c = m.GetColNum();
	return (c - rnk);
}

/*!
*	\brief	��� A�� �ϻﰢ���(lower triangle matrix) L�� ��ﰢ��� (upper triangle matrix) U�� �����Ѵ�.\n
*			���ذ������� �κ��Ǻ��� �� �౳ȯ�� �������� �ʴ´�.
*
*	\param A �����ϰ��� �ϴ� �������.
*	\param L ������ �ϻﰢ����� �����.
*	\param U ������ ��ﰢ����� �����.
*
*	\return ����: true, ����: false.
*/
inline
int dcmp_lu(const GMatrix &A, GMatrix &L, GMatrix &U)
{
	assert(A.r == A.c);
	assert(L.r == A.r);
	assert(L.c == A.c);
	assert(U.r == A.r);
	assert(U.c == A.c);
	int i, j, k, n;
	double big, dum, sum;
	n = A.r;

	GVector vv(n);  // vv stores the implicit scaling of each row
	GMatrix AA(A);
	L.SetIdentity();
	U.SetZeros();

	for (i = 0; i < n; i++)
	{
		big = 0.0;
		for (j = 0; j < n; j++)
			if (ABS(AA[i][j]) > big) 
				big = ABS(AA[i][j]);
		if (big == 0.0)
		{
			printf("Singular matrix in dcmp_lu()...\n");
			return false;
		}
	}

	for (j = 0; j < n; j++)			// loop over columns of Crout's method
	{
		for (i = 0; i < j; i++)		// equation (2.3.12) except i=j
		{
			sum = AA[i][j];
			for (k = 0; k < i; k++) 
                sum -= AA[i][k] * AA[k][j];
			AA[i][j] = sum;
		}
		for (i = j; i < n; i++) 	// i=j of equation (2.3.12) and i= j+1 .. N
		{							// of equation (2.3.13)
			sum = AA[i][j];
			for (k = 0; k < j; k++)
				sum -= AA[i][k] * AA[k][j];
			AA[i][j] = sum;
		}
		if (AA[j][j] == 0.0) 
			AA[j][j] = 1.0e-15;
		/* 
		if the pivot element is zero the matrix is singular (at least the
		precision of the algorithm). For some applications on singular 
		matrices, it is desirable to substitute TINY for zero
		*/
		if (j != n - 1)				// now finally, divide by the pivot element
		{
			dum = 1.0 / AA[j][j];
			for (i = j + 1; i < n; i++) 
				AA[i][j] *= dum;
		}
	}		// go back for the next column in the reduction

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (j >= i)
				U[i][j] = AA[i][j];
			else
				L[i][j] = AA[i][j];
		}
	}
	return true;
}

/*!
*	\brief	��� A�� �ϻﰢ���(lower triangle matrix) L�� ��ﰢ��� (upper triangle matrix) U�� ��ȯ�Ѵ�.\n
*			���ذ������� �κ��Ǻ��� (�౳ȯ)�� �����Ѵ�. ����, Idx = (0, 1, 2, ...)�� ���� ������, L*U �� A�� ���� �ʴ�.\n
*			���� ��, lin_sys_lu(A, x, b, Idx)�� ȣ���Ͽ� �����ý��� Ax = b �� �����ϴ� ���� x�� ���Ѵ�.
*
*	\param A n�� n���� ������� �Լ�ȣ�� ��, LU ���·� ��ȯ�ȴ�.
*	\param Idx �� ������ ��ȯ������ ����ȴ�.
*	\param d �� ������ ��ȯ������ ¦���̸� 1, Ȧ���̸� -1 �� ����ȴ�.
*
*	\return ����: true, ����: false.
*/
inline 
int dcmp_lu(GMatrix &A, GVector &Idx, double &d)
{
	assert(A.r == A.c);
	assert(Idx.N == A.r);
	int i, imax, j, k, n;
	double big, dum, sum, temp;
	n = A.r;
	GVector vv(n);  // vv stores the implicit scaling of each row

	d = 1.0;
	for (i = 0; i < n; i++)
	{
		big = 0.0;
		for (j = 0; j < n; j++)
			if ((temp = ABS(A[i][j])) > big ) 
				big = temp;
		if (big == 0.0)
		{
			printf("Singular matrix in dcmp_lu()...\n");
			return false;
		}
		// No nonzero largest element
		vv[i] = 1.0 / big;		// save the scaling
	}

	for (j = 0; j < n; j++)			// loop over columns of Crout's method
	{
		for (i = 0; i < j; i++)		// equation (2.3.12) except i=j
		{
			sum = A[i][j];
			for (k = 0; k < i; k++) 
				sum -= A[i][k] * A[k][j];
			A[i][j] = sum;
		}
		big = 0.0;					// initialize for the search for largest pivot element
		for (i = j; i < n; i++) 	// i=j of equation (2.3.12) and i= j+1 .. N
		{							// of equation (2.3.13)
			sum = A[i][j];
			for (k = 0; k < j; k++)
				sum -= A[i][k] * A[k][j];
			A[i][j] = sum;
			if ((dum = vv[i] * ABS(sum)) >= big)
			{
				// is the figure of merit for the pivot better than the best so far
				big = dum;
				imax = i;
			}
		}
		if (j != imax) 			// Do we need to interchange rows?
		{
			A.ExchangeRows(j, imax);
			d = - d;			// and change the parity of d
			vv[imax] = vv[j];	// also interchange the scale factor
		} 

		Idx[j] = imax;
		if (A[j][j] == 0.0) 
			A[j][j] = 1e-15;
        /* 
		if the pivot element is zero the matrix is singular (at least the
		precision of the algorithm). For some applications on singular 
		matrices , it is desirable to substitute TINY for zero
		*/
		if (j != n - 1)				// now finally, divide by the pivot element
		{
			dum = 1.0 / (A[j][j]);
			for (i = j + 1; i < n; i++) 
				A[i][j] *= dum;
		}
	}		// go back for the next column in the reduction
	return true;
}

/*!
*	\brief	���������� Ax = b �� �����ϴ� x�� ���Ѵ�.\n
*			- ��� A�� dcmp_lu(A, Idx, d)�� LU�� ���ص� ����̴�.\n
*			- ���� Idx�� dcmp_lu(A, Idx, d)�� �౳ȯ���� ����̴�.\n
*
*	\param A LU�� ���ص� �࿭.
*	\param x �ذ� ����� ����.
*	\param b ��� ����.
*	\param Idx �� ������ ��ȯ������ ������ ����.
*/
inline
void lin_sys_lu(const GMatrix &A, GVector &x, const GVector &b, const GVector &Idx)
{
	assert(A.r == A.c);
	assert(A.r == x.N);
	assert(x.N == b.N);
	assert(Idx.N == x.N);
	x = b;

	int i, ii = -1, ip, j, n;
	double sum;
	n = A.r;

	for (i = 0; i < n; i++)			// when ii is set to a positive value, it
	{								// will become the index of the first non-
		ip = (int)Idx[i];			// vanishing element of b. We now do the
		sum = x[ip];				// forward substitution, equation (2.3.6).
		x[ip] = x[i];				// The only new wrinkle is to unscalable 
		if (ii >= 0)				// the permutation as we go
			for (j = ii; j < i; j++) 
				sum -= A[i][j] * x[j];
		else if (sum)				// A nonzero element was encountered, so 
			ii = i;					// from now on we will have to do the sums
		x[i] = sum;					// in the loop above
	}

	for (i = n - 1; i >= 0; i--) // Now we do the back-substitution equation
	{							 // (2.3.7)
		sum = x[i];
		for (j = i + 1; j < n; j++) 
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];		// Store a component of the solution vector X
	}
}

/*!
*	\brief	��� A�� Householder ����� ����Ͽ� Q * R�� �����Ѵ�.
*
*	\param A m x n ���.
*	\param Q m x m orthogonal ���.
*	\param R m x n ��ﰢ���.
*
*	\return ����: -1, ����: householder ����� ������ ȸ��.
*/
inline
int dcmp_qr(const GMatrix &A, GMatrix &Q, GMatrix &R)
{
	if (Q.r != A.r || Q.c != A.r)
		return -1;
	if (R.r != A.r || R.c != A.c)
		return -1;

	int i, j, k ;
	double alpha, beta;
	Q.SetIdentity();
	R = A;
	GVector u(A.r);			// Target column vector 	
	GMatrix H(A.r, A.r);	// HouseHolder Matrix.

	for (k = 0; k < A.c; k++) // for each column
	{
		alpha = 0.0;
		for (i = k; i < A.r; i++) 
		{
			alpha += SQR(R[i][k]);
			u[i] = R[i][k];
		}
		alpha = SQRT(alpha);
		if (EQ_ZERO(alpha, GMatrix::Precision))
			break;
		beta = 1.0 / (alpha * alpha + SIGN1(alpha, u[k]) * u[k]);
		u[k] += SIGN1(alpha, u[k]);

		H.SetIdentity();
		for (i = k; i < A.r; i++)
			for (j = k; j < A.r; j++)
				H[i][j] -= beta * u[i] * u[j]; // Construct HouseHolder Matrix. -> Beautiful Computation.

		Q = Q * H;
		R = H * R;
	}
	return k;
}

/*!
*	\brief	��� A�� singular value decomposition�� �����Ѵ�.\n
*			- A = U * W * tr(V) �� ���صȴ�.\n
*
*	\param A m x n ��� (m >= n)
*	\param U m x n ��ķμ� �����ʹ� A�� �������� ���������� ����ȴ�.
*	\param S Ư�̰��� ����� n ��������. W�� S�� ���ҷ� �����Ǵ� �밢�� ���. 
*	\param V n x n ��ķμ� �����ʹ� A�� ����� �� �ΰ����� ���������� ����ȴ�.
*/
inline
void dcmp_sv(const GMatrix &A, GMatrix &U, GVector &S, GMatrix &V)
{
	int flag, i, its, j, jj, k, l, nm;
	double c, f, h, s, x, y, z;
	double anorm = 0.0, g = 0.0, scale = 0.0;
	int m = A.r;
	int n = A.c;
	assert(m >= n);
	assert(m == U.r && n == U.c);
	GVector rv1(n);
	U = A;
	for (i = 0; i < n; i++) 
	{
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m) 
		{
			for (k = i; k < m; k++)
				scale += fabs(U[k][i]);
			if (scale)
			{
				for (k = i; k < m; k++) 
				{
					U[k][i] /= scale;
					s += U[k][i] * U[k][i];
				}
				f = U[i][i];
				g = -SIGN2(SQRT(s), f);
				h = f * g - s;
				U[i][i] = f - g;
				if (i != n - 1) 
				{
					for (j = l; j < n; j++) 
					{
						for (s = 0.0, k = i; k < m; k++)
							s += U[k][i] * U[k][j];
						f = s / h;
						for (k = i; k < m; k++)
							U[k][j] += f * U[k][i];
					}
				}
				for (k = i; k < m; k++)
					U[k][i] *= scale;
			}
		}
		S[i] = scale * g;
		g = s = scale = 0.0;
		if (i <= m - 1 && i != n - 1) 
		{
			for (k = l; k < n; k++)
				scale += fabs(U[i][k]);
			if (scale) 
			{
				for (k = l; k < n; k++) 
				{
					U[i][k] /= scale;
					s += U[i][k] * U[i][k];
				}
				f = U[i][l];
				g = -SIGN2(SQRT(s), f);
				h = f * g - s;
				U[i][l] = f - g;
				for (k = l; k < n; k++)
					rv1[k] = U[i][k] / h;
				if (i != m - 1) 
				{
					for (j = l;j < m; j++) 
					{
						for (s = 0.0, k=l; k < n; k++)
							s += U[j][k] * U[i][k];
						for (k = l; k < n; k++)
							U[j][k] += s * rv1[k];
					}
				}
				for (k = l; k < n; k++) 
					U[i][k] *= scale;
			}
		}
		s = fabs(S[i]) + fabs(rv1[i]);
		anorm = MAX(anorm, s);
	}
	for (i = n - 1; i >= 0; i--) 
	{
		if (i < n - 1) 
		{
			if (g) 
			{
				for (j = l; j < n; j++)
					V[j][i] = (U[i][j] / U[i][l]) / g;
				for (j = l; j < n; j++) 
				{
					for (s = 0.0, k = l; k < n; k++)
						s += U[i][k] * V[k][j];
					for (k = l; k < n; k++)
						V[k][j] += s * V[k][i];
				}
			}
			for (j = l; j < n; j++)
				V[i][j] = V[j][i] = 0.0;
		}
		V[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}
	for (i = n - 1; i >= 0; i--) 
	{
		l = i + 1;
		g = S[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				U[i][j] = 0.0;
		if (g) 
		{
			g = 1.0 / g;
			if (i != n - 1)	
			{
				for (j = l; j < n; j++) 
				{
					for (s = 0.0, k = l; k < m; k++)
						s += U[k][i] * U[k][j];
					f = (s / U[i][i]) * g;
					for (k = i; k < m; k++)
						U[k][j] += f * U[k][i];
				}
			}
			for (j = i; j < m; j++)
				U[j][i] *= g;
		}
		else 
		{
			for (j = i; j < m; j++)
				U[j][i] = 0.0;
		}
		++U[i][i];
	}
	for (k = n - 1; k >= 0; k--) 
	{
		for (its = 1; its <= 30; its++) 
		{
			flag = 1;
			for (l = k; l >= 0; l--) 
			{
				nm = l - 1;
				if (fabs(rv1[l]) + anorm  ==  anorm) 
				{
					flag = 0;
					break;
				}
				if (fabs(S[nm]) + anorm == anorm)
					break;
			}
			if (flag) 
			{
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++) 
				{
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm) 
					{
						g = S[i];
						h = PYTHAG(f, g);
						S[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (-f * h);
						for (j = 0; j < m; j++) 
						{
							y = U[j][nm];
							z = U[j][i];
							U[j][nm] = y * c + z * s;
							U[j][i] = z * c - y * s;
						}
					}
				}
			}
			z = S[k];
			if (l == k) 
			{
				if (z < 0.0) 
				{
					S[k] = -z;
					for (j = 0; j < n; j++)
						V[j][k] = (-V[j][k]);
				}
				break;
			}
			//if (its == 30)
			//	cerr << "No convergence in 30 SVDCMP iterations";
			x = S[l];
			nm = k - 1;
			y = S[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = PYTHAG(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN2(g, f))) - h)) / x;
			c = s = 1.0;
			for (j = l; j <= nm; j++) 
			{
				i = j + 1;
				g = rv1[i];
				y = S[i];
				h = s * g;
				g = c * g;
				z = PYTHAG(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for (jj = 0; jj < n; jj++) 
				{
					x = V[jj][j];
					z = V[jj][i];
					V[jj][j] = x * c + z * s;
					V[jj][i] = z * c - x * s;
				}
				z = PYTHAG(f, h);
				S[j] = z;
				if (z) 
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++) 
				{
					y = U[jj][j];
					z = U[jj][i];
					U[jj][j] = y * c + z * s;
					U[jj][i] = z * c - y * s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			S[k] = x;
		}
	}
}

/*!
*	\brief	���������� Ax = b �� �����ϴ� minimum length least square �� x�� ���Ѵ�.
*
*	\param A m x n ��� (m >= n).
*	\param x �ذ� ����� n ���� ����.
*	\param b m ���� �������.
*
*	\return condition number.
*/
inline
double lin_sys_sv(const GMatrix &A, GVector &x, const GVector &b)
{
	int i, j;
	int m = A.r;
	int n = A.c;
	double min, max, cnum;

	GMatrix U(m, n);
	GVector S(n);
	GMatrix V(n, n);
	dcmp_sv(A, U, S, V);
	min = max = S[0];

	for (i = 0; i < n; i++) 
	{
		min = MIN(min, S[i]);
		max = MAX(max, S[i]);
		cnum = min / max;
	}

	GVector tmp(n);
	for (j = 0; j < n; j++) 
	{
		double s = 0.0;
		if (!EQ_ZERO(S[j], GMatrix::Precision))
		{
			for (i = 0; i < m; i++)
				s += U[i][j] * b[i];
			s /= S[j];
		}
		tmp[j] = s;
	}
	for (j = 0; j < n; j++) 
	{
		double s = 0.0;
		for (i = 0; i < n; i++)
			s += V[j][i] * tmp[i];
		x[j] = s;
	}
	return cnum;
}

/*!
*	\brief	���������� Ax = b �� �����ϴ� minimum length least square �� x�� ���Ѵ�.\n
*			- �� �Լ��� ȣ���ϱ� ����, dcmp_sv(A, U, S, V)�� ȣ���Ͽ� ��� A�� ���ص� ����� ����Ѵ�.
*			- �ѹ� ���ص� ��� A�� ������� b�� �ٲٸ鼭 �ظ� ���ϰ� �ȴ�.
*
*	\param U m x n ��ķ� dcmp_sv(A, U, S, V)�� ���.
*	\param S n ���� ���ͷ� dcmp_sv(A, U, S, V)�� ���.
*	\param V n x n ��ķ� dcmp_sv(A, U, S, V)�� ���.
*	\param x �ذ� ����� n ���� ����.
*	\param b m ���� �������.
*
*	\return ���Ǽ� (condition number).
*/
inline
double lin_sys_sv(const GMatrix &U, const GVector &S, const GMatrix &V, GVector &x, const GVector &b)
{
	int i, j;
	int m = U.r;
	int n = U.c;
	double min, max, cnum;
	min = max = S[0];

	for (i = 0; i < n; i++) 
	{
		min = MIN(min, S[i]);
		max = MAX(max, S[i]);
		cnum = min / max;
	}
	GVector tmp(n);
	for (j = 0; j < n; j++) 
	{
		double s = 0.0;
		if (!EQ_ZERO(S[j], GMatrix::Precision)) 
		{
			for (i = 0; i < m; i++)
				s += U[i][j] * b[i];
			s /= S[j];
		}
		tmp[j] = s;
	}
	for (j = 0; j < n; j++) 
	{
		double s = 0.0;
		for (i = 0; i < n; i++)
			s += V[j][i] * tmp[i];
		x[j] = s;
	}
	return cnum;
}

/*!
*	\brief	LU ���ظ� ����, ��� A�� ��Ľ��� ���Ѵ�.
*
*	\param A n x n ���.
*
*	\return ��Ľ��� ��.
*/
inline
double det(const GMatrix &A)
{
	assert(A.IsSquare());
	int i, n = A.r;
	double d;
	GMatrix LU(A);
	GVector Idx(n);

	dcmp_lu(LU, Idx, d);

	for (i = 0; i < n; i++)
		d *= LU[i][i];

	return d;
}

/*!
*	\brief	LU ���ظ� ����, ��� A�� ������� ���Ѵ�.
*
*	\param A n x n ���.
*
*	\return ��� A�� �����.
*/
inline
GMatrix inv(const GMatrix &A)
{
	assert(A.IsSquare());
	int n = A.r;
	GMatrix ret(n, n);    
	double d;
	GVector Idx(n);
	GMatrix LU(A);
	
	// for fast inverse computation remove determinant check.
	d = det(LU);
	assert(!EQ_ZERO(d, GMatrix::Precision));

	dcmp_lu(LU, Idx, d);
	for (int j = 0; j < n; j++)
	{
		GVector x(n);
		GVector b(n);
		b[j] = 1.0;
		lin_sys_lu(LU, x, b, Idx);
		ret.SetColVec(j, x);
	}

	return ret;
}

/*!
*	\brief	Conjugate gradient ����� ����Ͽ� ���������� Ax = b �� Ǭ��.\n
*			- ��� A�� ��Ī����̾�� �Ѵ�.
*			- sparse �����ý����� Ǯ������ ȿ������ ����̴�.
*
*	\param A n x n ��Ī���.
*	\param x n ���� �ʱ� �� ����.
*	\param b n ���� ��� ����.
*	\param iter �ݺ�ȸ��, ���� �ݺ�ȸ���� n�̸� ��Ȯ�� �ظ� ���Ѵ�.
*
*	\return ���� �ݺ�ȸ��.
*/
inline
int lin_sys_cg(const GMatrix &A, GVector &x, const GVector &b, int iter)
{
	int n = A.r;
	assert(A.IsSquare());
	assert(n == x.N);
	assert(n == b.N);
	if (iter < 0 || iter > n)
		iter = n;

	GVector tmp0(n);
	double alpha, beta, tmp1;
	GVector r(b);	// initial residual.
	GVector d(b);	// initial conjugate direction.

	// Conjugate Gradient Iteration
	// Such a simple algorithm... 
	// but great theorem....(Expanding Subspace Theorem)
	int i;
	for (i = 0; i < iter; ++i)
	{
		tmp0 = A * d;
		tmp1 = r * r;

		alpha = tmp1 / (d * tmp0);
		x = x + alpha * d;
		r = r - alpha * tmp0;
		beta = (r * r) / tmp1;
		d = r + beta * d;
	}

	return i;
}

/*!
*	\brief	QR ���ع��� �̿��Ͽ� ���������� Ax = b�� least square �ظ� ã�´�.
*
*	\param A m x n ���.
*	\param x n ���� �� ����.
*	\param b n ���� ��� ����.
*
*	\return 1: ����, 0: ����.
*/
inline
int lin_sys_qr(const GMatrix &A, GVector &x, const GVector &b)
{
	int m, n, i, j, scs;
	m = A.r;
	n = A.c;

	GMatrix Q(m, m);
	GMatrix R(m, n);
	GVector c(m);

	scs =dcmp_qr(A, Q, R);

	if (scs != A.c)
		return false;

	Q.SetTranspose();
	c = Q * b;

	for (i = n - 1; i >= 0; i--)
	{
		double sum = 0.0;
		for (j = i + 1; j < n; j++)
			sum += (R[i][j] * x[j]);
		x[i] = (c[i] - sum) / R[i][i];
	}
	return true;
}

#define ROTATE(a, i, j, k, l)\
{ \
	g = a[i][j]; \
	h = a[k][l]; \
	a[i][j] = g - s * (h + g * tau); \
	a[k][l] = h + s * (g - h * tau); \
}
/*!
*	\brief	��纯ȯ (similarity transformation)�� ����Ͽ� �Ǽ���Ī����� ��� eigenvector�� eigenvalue�� ���Ѵ�.
*
*	\param A n x n ��Ī���.
*	\param D n���� eigenvalue�� ������ ����.
*	\param V n���� eigenvector (������)�� ������ ���.
*
*	\return ��纯ȯ�� ȸ��.
*/
inline
int eigen_sys(const GMatrix &A, GVector &D, GMatrix &V)
{
	int n = A.r;
	assert(A.c == n);
	assert(D.N == n);
	assert(V.r == n);
	assert(V.c == n);
	int j, iq, ip, i, nrot;
	double tresh, theta, tau, t, sm, s, h, g, c, *b, *z;

	b = new double [n];
	z = new double [n];

	GMatrix B(A);
	V.SetIdentity();

	for (ip = 0; ip < n; ip++) 
	{
		b[ip] = D[ip] = B[ip][ip];
		z[ip] = 0.0;
	}

	nrot = 0;

	for (i = 1; i <= 50; i++) 
	{
		sm = 0.0;
		for (ip = 0; ip < n - 1; ip++) 
		{
			for (iq = ip + 1; iq < n; iq++)
				sm += ABS(B[ip][iq]);
		}
		if (sm == 0.0) 
		{
			delete [] b;
			delete [] z;
			return nrot;
		}
		if (i < 4)
			tresh = 0.2 * sm / (n * n);
		else
			tresh = 0.0;
		for (ip = 0; ip < n - 1; ip++) 
			for (iq = ip + 1; iq < n; iq++) 
			{
				g = 100.0 * ABS(B[ip][iq]);
				if (i > 4 && ABS((D[ip]) + g) == ABS(D[ip]) && (ABS(D[iq]) + g) == ABS(D[iq]))
					B[ip][iq] = 0.0;
				else if (ABS(B[ip][iq]) > tresh) 
				{
					h = D[iq] - D[ip];
					if ((ABS(h) + g) == ABS(h))
						t = (B[ip][iq]) / h;
					else 
					{
						theta = 0.5 * h / B[ip][iq];
						t = 1.0 / (ABS(theta) + SQRT((1.0 + SQR(theta))));
						if (theta < 0.0) 
							t = -t;
					}
					c = 1.0 / SQRT(1 + SQR(t));
					s = t * c;
					tau = s / (1.0 + c);
					h = t * B[ip][iq];
					z[ip] -= h;
					z[iq] += h;
					D[ip] -= h;
					D[iq] += h;
					B[ip][iq] = 0.0;
					for (j = 0; j < ip; j++) 
						ROTATE(B, j, ip, j, iq)
						for (j = ip + 1; j < iq; j++) 
							ROTATE(B, ip, j, j, iq)
							for (j = iq + 1; j < n; j++) 
								ROTATE(B, ip, j, iq, j)
								for (j = 0; j < n; j++) 
									ROTATE(V, j, ip, j, iq)
									++nrot;
				}
			}
			for (ip = 0; ip < n; ip++) 
			{
				b[ip] += z[ip];
				D[ip] = b[ip];
				z[ip] = 0.0;
			}
	}
	return nrot;
}

/*!
*	\brief dcmp_sv �� ����Ͽ� �ּ��� (principal component)�� ã�´�.
*	\note �Լ�ȣ�� ��, Data ������ ����� ������ ��ġ�ϵ��� �̵��ȴ�.
*
*	\param Data m x n ������ ��� (m: �������� ����, n: �������� ����)
*	\param PC �����ͷ� ǥ���� �ּ���.
*	\param SigVal �ּ��к����� Ư�̰��� ������ n ���� ����.
*	\param Mean ��պ���.
*/
inline
void pca(GMatrix &Data, GMatrix &PC, GVector &SigVal, GVector &Mean)
{
	int i;
	int num = Data.r;
	int dim = Data.c;
	assert(PC.r == dim);
	assert(PC.c == dim);
	assert(SigVal.N == dim);

	// 1. Get mean vector
	GVector data(dim);
	for (i = 0; i < num; i++)
	{
		data = Data.GetRowVec(i);
		Mean += data;
	}
	Mean = Mean / (double)num;
	// for debug.
	// cerr << "mean vector:" << endl << Mean;

	// 2. Translate to mean vector.
	for (i = 0; i < num; i++)
	{
		data = Data.GetRowVec(i);
		data -= Mean;
		Data.SetRowVec(i, data);
	}
	Data /= SQRT((double)num);
	// for debug.
	//printf( "translate matrix:\n" );
	//Data.Print();

	// 3. Do singular Value Decomposition.
	GMatrix U(num, dim), V(dim, dim);
	GVector W(dim);
	dcmp_sv(Data, U, W, V);

	// 4. Sorting by its singular value size.
	for (i = 0; i < dim - 1; i++)
	{
		if (W[i] < W[i + 1])
		{
			// swap eigenvalue.
			SWAP(double, W[i], W[i + 1]);
			// swap eigenvector.
			V.ExchangeCols(i, i + 1);
			U.ExchangeCols(i, i + 1);
		}
	}

	PC = V;
	SigVal = W;
}

/*!
*	\brief ������ ���Ҹ� ���� �Ǽ��迭�� ���Ѵ�.
*
*	\param v ����
*
*	\return ������ ���Ҹ� ������ �Ǽ� �迭
*/
inline
double *cast_arr(const GVector &v)
{
	return v.V;
}

/*!
*	\brief �Ϲ����� ����ǥ���� 3���� ���ͷ� ��ȯ�Ѵ�.
*
*	\param v ����
*
*	\return v�� 3���� ����ǥ��
*/
inline
GVector3 cast_vec3(const GVector &v)
{
	return GVector3(v[0], v[1], v[2]);
}

/*!
*	\brief �Ϲ����� ����ǥ���� 3���� ����Ʈ�� ���·� ��ȯ�Ѵ�.
*
*	\param v ����
*
*	\return v�� 3���� ����Ʈ ǥ��
*/
inline
GPoint3 cast_pt3(const GVector &v)
{
	return GPoint3(v[0], v[1], v[2]);
}

/*!
*	\class GQuater		   
*	\brief ���ʹϿ� (q = W + iX + jY + kZ)�� ��Ÿ���� Ŭ����.
*	\warning �������ʹϿ��� �ƴѰ�쵵 ����.
*
*	\author ������ (shyun@dongguk.edu)
*	\date 01 Jan 2001	
*/
enum EulerType
{
	EULER_XYZ = 0,
	EULER_ZYX = 1,
};

enum ArcBallType
{ 
	ARCBALL_FREE = 0, 
	ARCBALL_X, 
	ARCBALL_Y, 
	ARCBALL_Z 
};
class GQuater
{
	// ����Ʈ �Լ� �� Ŭ����
	friend GQuater operator +(const GQuater &lhs, const GQuater &rhs);
	friend GQuater operator -(const GQuater &lhs, const GQuater &rhs);
	friend GQuater operator *(const GQuater &lhs, const GQuater &rhs);
	friend GQuater operator /(const GQuater &lhs, const GQuater &rhs);
	friend GQuater operator /(const GQuater &lhs, const double s);
	friend GPoint3 operator *(const GQuater &lhs, const GPoint3 &rhs);
	friend GVector3 operator *(const GQuater &lhs, const GVector3 &rhs);
	friend bool operator ==(const GQuater &lhs, const GQuater &rhs);
	friend bool operator !=(const GQuater &lhs, const GQuater &rhs);
	friend GQuater operator *(const GQuater &lhs, const double &s);
	friend GQuater operator *(const double &s, const GQuater &rhs);
	friend ostream &operator <<(ostream &os, const GQuater &q);
	friend istream &operator >>(istream &is, GQuater &q);
	friend double norm(const GQuater &q);
	friend GQuater inv(const GQuater &q);
	friend GQuater exp(const GVector3 &v);
	friend GVector3 log(const GQuater &q);
	friend double quat_dot(const GQuater &q1, const GQuater &q2);
	friend GQuater slerp(const GQuater &q1, const GQuater &q2, const double t);
	friend GMatrix cast_mat(const GQuater &q, bool isGL = true);
	friend GVector cast_vec(const GQuater &q);

public:
	// ������ �� �Ҹ���
	GQuater(double w = 1.0, double x = 0.0, double y = 0.0, double z = 0.0);
	GQuater(const GQuater &cpy);
	GQuater(const double *q, const bool invOrder = false);
	GQuater(GVector3 axis, double theta, bool radian = false);
	GQuater(double theta1, double theta2, double theta3, EulerType eulerType = EULER_XYZ);
	virtual ~GQuater();

	// ���Կ����� �ߺ�
	GQuater &operator =(const GQuater &rhs);

	// ���մ��Կ����� �ߺ�
	GQuater &operator +=(const GQuater &rhs);
	GQuater &operator -=(const GQuater &rhs);
	GQuater &operator *=(const GQuater &rhs);
	GQuater &operator /=(const GQuater &rhs);
	GQuater &operator *=(const double s);
	GQuater &operator /=(const double s);

	// ���׿����� �ߺ�
	GQuater operator +() const;
	GQuater operator -() const;

	// Set ����Լ�
	GQuater &Set(const double w, const double x, const double y, const double z);
	GQuater &Set(double *q, bool invOrder = false);
	GQuater &SetIdentity();
	GQuater &SetInverse();
	GQuater &SetConjugate();
	GQuater &SetNegate();
	GQuater &SetFromArcBall(GVector3 p, GVector3 q, ArcBallType arcType = ARCBALL_FREE);
	GQuater &SetFromAngleAxis(const double theta, GVector3 axis, bool radian = false);
	GQuater &SetFromEulerAngle(double theta1, double theta2, double theta3, EulerType eulerType = EULER_XYZ);
	GQuater &SetFromFrameXY(const GVector3 &x_axis, const GVector3 &y_axis);
	GQuater &SetFromFrameYZ(const GVector3 &y_axis, const GVector3 &z_axis);
	GQuater &SetFromFrameZX(const GVector3 &z_axis, const GVector3 &x_axis);
	GQuater &SetFromMatrix(const GMatrix &m);
	GQuater &Normalize();

	// Get ����Լ�
	void	GetAngleAxis(GVector3 &axis, double &angle, bool radian = false) const;
	void	GetEulerAngle(double &theta1, double &theta2, double &theta3, bool radian = false, EulerType eulerType = EULER_XYZ);
	double	GetW() const;
	double	GetX() const;
	double	GetY() const;
	double	GetZ() const;
	
	// ���� ����Լ�
	bool IsUnitQuater() const;
	bool IsIdentity() const;

	// ���� ����Լ�
	static void SetPrecision(double error);
	static double GetPrecision();
	
protected:
	// ������ ���
	/*! \brief ���ʹϿ��� �Ǽ��� */
	double W;

	/*! \brief ���ʹϿ��� ����� */
	double X, Y, Z;

	/*! \brief ��ȣ �� ��Һ񱳸� ���� ��Ȯ�� */
	static double Precision;
};


/*!
*	\brief ���ʹϿ� q = w + xi + yj + zk �� �����ϴ� ������
*	
*	\param w �Ǽ���
*	\param x i�� �����
*	\param y j�� �����
*	\param z k�� �����
*/
inline
GQuater::GQuater(double w, double x, double y, double z)
{
	this->W = w;
	this->X = x;
	this->Y = y;
	this->Z = z;
}

/*!
*	\brief ��������� (copy constructor)
*	
*	\param cpy ����� ���ʹϿ� ��ü
*/
inline
GQuater::GQuater(const GQuater &cpy)
{
	this->W = cpy.W;
	this->X = cpy.X;
	this->Y = cpy.Y;
	this->Z = cpy.Z;
}

/*!
*	\brief ������ (constructor)
*	
*	\param q ������ ���ʹϿ��� ���Ҹ� ������ �迭
*	\param invOrder ������ �������. 
*          true: W = q[1], X = q[2], Y = q[3], W = q[0].
*		   false: W = q[0], X = q[1], Y = q[2], W = q[3].
*/
inline
GQuater::GQuater(const double *q, const bool invOrder)
{
	if (invOrder)
	{
		this->W = q[1];
		this->X = q[2];
		this->Y = q[3];
		this->Z = q[0];
	} 
	else
	{
		this->W = q[0];
		this->X = q[1];
		this->Y = q[2];
		this->Z = q[3];
	}
}

/*!
*	\brief ȸ���� axis�� �߽����� theta��ŭ ȸ���ϴ� �������ʹϿ��� �����Ѵ�.
*	
*	\param axis ȸ���� �߽���.
*	\param theta ȸ������
*	\param radian true: theta�� �����, false: theta�� degree.
*/
inline
GQuater::GQuater(GVector3 axis, double theta, bool radian /* = false */)
{
	double rad, sn, cs;
	axis.Normalize();
	if (!radian)
		rad = theta * M_PI / 360.0;

	sn = sin(rad);
	cs = cos(rad);

	sn = (abs(sn) < Precision) ? 0.0 : sn;
	cs = (abs(cs) < Precision) ? 0.0 : cs;

	W = cs;
	X = sn * axis[0];
	Y = sn * axis[1];
	Z = sn * axis[2];
}

/*!
*	\brief ���Ϸ��� (degree) �̿��Ͽ� ���ʹϿ��� �����Ѵ�.
*
*	\param theta1 ù��°�࿡ ���� ȸ������ (degree).
*	\param theta2 �ι�°�࿡ ���� ȸ������ (degree).
*	\param theta3 ����°�࿡ ���� ȸ������ (degree).
*	\param eulerType ���Ϸ��� (XYZ = RxRyRz, ZYX = RzRyRx, ...)
*/
inline
GQuater::GQuater(double theta1, double theta2, double theta3, EulerType eulerType)
{
	double c1, c2, c3;
	double s1, s2, s3;
	theta1 = DEG2RAD(theta1);
	theta2 = DEG2RAD(theta2);
	theta3 = DEG2RAD(theta3);
	c1 = cos(theta1);
	c2 = cos(theta2);
	c3 = cos(theta3);
	s1 = sin(theta1);
	s2 = sin(theta2);
	s3 = sin(theta3);

	GMatrix mat;
	switch (eulerType)
	{
	case EULER_XYZ:
		mat[0][0] = c2 * c3;	
		mat[0][1] = -c2 * s3;	
		mat[0][2] = s2;
		mat[0][3] = 0.0;

		mat[1][0] = s1 * s2 * c3 + c1 * s3;
		mat[1][1] = -s1 * s2 * s3 + c1 * c3;
		mat[1][2] = -s1 * c2;
		mat[1][3] = 0.0;

		mat[2][0] = -c1 * s2 * c3 + s1 * s3;
		mat[2][1] = c1 * s2 * s3 + s1 * c3;
		mat[2][2] = c1 * c2;
		mat[2][3] = 0.0;

		mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
		mat[3][3] = 1.0;
		break;
	case EULER_ZYX:
		mat[0][0] = c3 * c2;	
		mat[0][1] = -s3 * c1 + c3 * s2 * s1;
		mat[0][2] = s3 * s1 + c3 * s2 * c1;
		mat[0][3] = 0.0;

		mat[1][0] = s3 * c2;
		mat[1][1] = c3 * c1 + s3 * s2 * s1;
		mat[1][2] = -c3 * s1 + s3 * s2 * c1;
		mat[1][3] = 0.0;

		mat[2][0] = -s2;
		mat[2][1] = c2 * s1;
		mat[2][2] = c2 * c1;
		mat[2][3] = 0.0;

		mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
		mat[3][3] = 1.0;
		break;
	}
	SetFromMatrix(mat);
}

/*!
*	\brief �Ҹ��� (destructor)
*/
inline
GQuater::~GQuater()
{
}

/*!
*	\brief ���Կ�����
*
*	\param rhs ���Կ������� ������ �ǿ�����
*
*	\return ���Ե� �ڽ��� ��ȯ
*/
inline
GQuater &GQuater::operator =(const GQuater &rhs)
{
	W = rhs.W;
	X = rhs.X;
	Y = rhs.Y;
	Z = rhs.Z;
	return *this;
}

/*!
*	���մ��Կ�����
*
*	\param rhs ���մ��Կ������� ������ �ǿ�����.
*
*	\return ������ ���Ե� �ڽ��� ��ȯ
*/
inline
GQuater &GQuater::operator +=(const GQuater &rhs)
{
	W += rhs.W;
	X += rhs.X;
	Y += rhs.Y;
	Z += rhs.Z;
	return *this;
}

/*!
*	���մ��Կ�����
*
*	\param rhs ���մ��Կ������� ������ �ǿ�����.
*
*	\return ������ ���Ե� �ڽ��� ��ȯ
*/
inline
GQuater &GQuater::operator -=(const GQuater &rhs)
{
	W -= rhs.W;
	X -= rhs.X;
	Y -= rhs.Y;
	Z -= rhs.Z;
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*	\warning ���� ���ʹϿ��� ����, ������ ���������� �����ϴ� ���� ��������ʹϿ��� �� �� ����.
*
*	\param rhs ���մ��Կ������� ������ �ǿ�����. 
*
*	\return ������ ���Ե� �ڽ��� ��ȯ
*/
inline
GQuater &GQuater::operator *=(const GQuater &rhs)
{
	double w = W, x = X, y = Y, z = Z;
	this->W = w * rhs.W - x * rhs.X - y * rhs.Y - z * rhs.Z;
	this->X = w * rhs.X + rhs.W * x + y * rhs.Z - z * rhs.Y;
	this->Y = w * rhs.Y + rhs.W * y + z * rhs.X - x * rhs.Z;
	this->Z = w * rhs.Z + rhs.W * z + x * rhs.Y - y * rhs.X;
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*	\usage q1 /= q2;	// q2���� q1���� ������� ȸ���� ���Ѵ�
*
*	\param rhs ���մ��Կ������� �������� �ǿ�����.
*
*	\return ������ ���Ե� �ڽ��� ��ȯ
*/
inline
GQuater &GQuater::operator /=(const GQuater &rhs)
{
	(*this) = inv(rhs) * (*this);
	Normalize();
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param s ���մ��Կ������� �������� �ǿ����� (�Ǽ�).
*
*	\return ������ ���Ե� �ڽ��� ��ȯ
*/
inline
GQuater &GQuater::operator *=(const double s)
{
	W *= s;
	X *= s;
	Y *= s;
	Z *= s;
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param s ���մ��Կ������� �������� �ǿ����� (�Ǽ�).
*
*	\return ������ ���Ե� �ڽ��� ��ȯ
*/
inline
GQuater &GQuater::operator /=(const double s)
{
	W /= s;
	X /= s;
	Y /= s;
	Z /= s;
	return *this;
}


/*!
*	\brief ���׿�����
*/
inline
GQuater GQuater::operator -() const
{
	return *this * -1;
}

/*!
*	\brief ���׿�����
*/
inline
GQuater GQuater::operator +() const
{
	return *this;
}

/*!
*	\brief ���������(+)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GQuater operator +(const GQuater &lhs, const GQuater &rhs)
{
	GQuater ret(lhs);
	ret += rhs;
	return ret;
}

/*!
*	\brief ���������(-)
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GQuater operator -(const GQuater &lhs, const GQuater &rhs)
{
	GQuater ret(lhs);
	ret -= rhs;
	return ret;
}

/*!
*	\brief ���׿����� (*) �ߺ�
*	\warning ���� ���ʹϿ��� ����, ������ ���������� �����ϴ� ���� ��������ʹϿ��� �� �� ����.
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return �� ���ʹϿ��� ���� ��ȯ�Ѵ�.
*/
inline
GQuater operator *(const GQuater &lhs, const GQuater &rhs)
{
	GQuater ret(lhs);
	ret *= rhs;
	return ret;
}

/*!
*	\brief (/) ������ �ߺ�
*	\note q1 / q2;	// q2���� q1���� ������� ȸ���� ���Ѵ�
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return ������ ����� ��ȯ
*/
inline
GQuater operator /(const GQuater &lhs, const GQuater &rhs)
{
	GQuater ret(lhs);
	ret /= rhs;
	return ret;
}

/*!
*	\brief ���׿����� (*) �ߺ�
*
*	\param lhs ���� �ǿ�����
*	\param s ������ �ǿ����� (�Ǽ�)
*
*	\return ���ʹϿ��� �Ǽ��� (1/s)�� ��ȯ�Ѵ�.
*/
inline
GQuater operator /(const GQuater &lhs, const double s)
{
	GQuater ret(lhs);
	ret /= s;
	return ret;
}

/*!
*	\brief ���׿����� (*) �ߺ�
*		    (*) ����: ���� ���ʹϿ��� �������ʹϿ� �̾�� �Ѵ�.
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ����� (����)
*
*	\return ���� rhs�� ���ʹϿ����� ȸ���� ���͸� ����Ͽ� �����Ѵ�. 
*/
inline
GVector3 operator *(const GQuater &lhs, const GVector3 &rhs)
{
	assert(lhs.IsUnitQuater());
	GVector3 ret;
	GQuater v(0.0, rhs[0], rhs[1], rhs[2]);
	GQuater rq = lhs * v * inv(lhs);
	ret.Set(rq.X, rq.Y, rq.Z);
	return ret;
}

/*!
*	\brief ���׿����� (*) �ߺ�
*		    (*) ����: ���� ���ʹϿ��� �������ʹϿ� �̾�� �Ѵ�.
*
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ����� (��)
*
*	\return �� rhs�� ���ʹϿ����� ȸ���� ���� ����Ͽ� �����Ѵ�. 
*/
inline
GPoint3 operator *(const GQuater &lhs, const GPoint3 &rhs)
{
	assert(lhs.IsUnitQuater());
	GPoint3 ret;
	GQuater v(0.0, rhs[0], rhs[1], rhs[2]);
	GQuater rq = lhs * v * inv(lhs);
	ret.Set(rq.X, rq.Y, rq.Z);
	return ret;
}

/*!
*	\brief ��ġ������ ������ �����ϴ� ��������Լ�
*
*	\param error ��ġ������ ����
*/
inline
void GQuater::SetPrecision(double error)
{
	Precision = error;
}

/*!
*	\brief ��ġ������ ������ ����
*
*	\return ������ ��ġ������ ����
*/
inline
double GQuater::GetPrecision()
{
	return Precision;
}

/*!
*	\brief ��ȣ������ (==) �ߺ�
*
*	\param lhs ���� �ǿ�����
*	\param rhs ��ȣ�� ������ �ǿ�����
*
*	\return �� ���ʹϿ��� ������������ ������ true, �ٸ��� false ��ȯ
*/
inline
bool operator ==(const GQuater &lhs, const GQuater &rhs)
{
	double error = GQuater::Precision;
	return (EQ(lhs.W, rhs.W, error) && EQ(lhs.X, rhs.X, error) && EQ(lhs.Y, rhs.Y, error) && EQ(lhs.Z, rhs.Z, error));
}

/*!
*	\brief ��ȣ������ (!=) �ߺ�
*
*	\param lhs ���� �ǿ�����
*	\param rhs ��ȣ�� ������ �ǿ�����
*
*	\return �� ���ʹϿ��� ������������ �ٸ��� true, ������ false ��ȯ
*/
inline 
bool operator !=(const GQuater &lhs, const GQuater &rhs)
{
	double error = GQuater::Precision;
	return (!EQ(lhs.W, rhs.W, error) || !EQ(lhs.X, rhs.X, error) || !EQ(lhs.Y, rhs.Y, error) || !EQ(lhs.Z, rhs.Z, error));
}


/*!
*	\brief Set ����Լ�
*	
*	\param w ���ʹϿ��� �Ǽ���
*	\param x ���ʹϿ��� i �����
*	\param y ���ʹϿ��� j �����
*	\param z ���ʹϿ��� k �����
*/
inline
GQuater &GQuater::Set(const double w, const double x, const double y, const double z)
{
	this->W = w;
	this->X = x;
	this->Y = y;
	this->Z = z;
	return *this;
}

/*!
*	\brief Set ����Լ�
*	
*	\param q ������ ���ʹϿ��� ���Ҹ� ������ �迭
*	\param invOrder ������ �������. 
*          true: W = q[3], X = q[0], Y = q[1], W = q[2].
*		   false: W = q[0], X = q[1], Y = q[2], W = q[3].
*/
inline
GQuater &GQuater::Set(double *q, bool invOrder)
{
	if (invOrder)
	{
		this->W = q[3];
		this->X = q[0];
		this->Y = q[1];
		this->Z = q[2];
	} 
	else
	{
		this->W = q[0];
		this->X = q[1];
		this->Y = q[2];
		this->Z = q[3];
	}
}

/*!
*	\brief axis ���� �߽����� theta ��ŭ ȸ���ϴ� ���ʹϿ��� �����Ѵ�.
*
*	\param theta ȸ���� ������ ��Ÿ����.
*	\param axis ȸ���� �߽����� ��Ÿ����.
*	\param radian theta�� ������̸� true, �����̸� false.
*
*	\return ������ �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetFromAngleAxis(const double theta, GVector3 axis, bool radian)
{
	double rad, sn, cs;
	axis.Normalize();
	if (!radian)
		rad = theta * M_PI / 360.0;
	else
		rad = theta / 2.0;

	sn = sin(rad);
	cs = cos(rad);

	sn = (abs(sn) < Precision) ? 0.0 : sn;
	cs = (abs(cs) < Precision) ? 0.0 : cs;

	W = cs;
	X = sn * axis[0];
	Y = sn * axis[1];
	Z = sn * axis[2];

	return *this;
}

/*!
*	\brief ��ũ������ �� p�� �� q�� ȸ���ϴ� ���ʹϿ��� �����Ѵ�.
*
*	\param p ��ũ������ ��.
*	\param q ��ũ������ ��.
*	\param arcType ��ũ���� Ÿ��.
*
*	\return ������ �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetFromArcBall(GVector3 p, GVector3 q, ArcBallType arcType)
{
	double ang, nm;
	GVector3 axis;

	p.Normalize();
	q.Normalize();

	double dot = p * q;
	if (EQ(dot, -1.0, Precision))	// p = -q �ΰ��
	{
		int idx = 0;
		if (p[idx] >= p[1])
			idx = 1;

		if (p[idx] >= p[2])
			idx = 2;

		switch (idx)
		{
		case 0:
			W = 0.0;
			X = 0.0;
			Y = p[2];
			Z = -p[1];
			break;
		case 1:
			W = 0.0;
			X = p[2];
			Y = 0.0;
			Z = -p[0];
			break;
		case 2:
			W = 0.0;
			X = p[1];
			Y = -p[0];
			Z = 0.0;
			break;
		}

		return *this;
	}

	axis = p ^ q;
	nm = norm(axis);

	if (EQ_ZERO(nm, Precision))	// p == q �ΰ��
	{
		SetIdentity();
		return *this;
	}

	switch (arcType)
	{
	case ARCBALL_X:
		if (axis[0] < 0.0)
			axis.Set(-1.0, 0.0, 0.0);
		else
			axis.Set(1.0, 0.0, 0.0);
		break;
	case ARCBALL_Y:
		if (axis[1] < 0.0)
			axis.Set(0.0, -1.0, 0.0);
		else
			axis.Set(0.0, 1.0, 0.0);
		break;
	case ARCBALL_Z:
		if (axis[1] < 0.0)
			axis.Set(0.0, 0.0, -1.0);
		else
			axis.Set(0.0, 0.0, 1.0);
		break;
	}

	ang = angle(p, q, false);
	SetFromAngleAxis(ang, axis);
	return *this;
}

/*!
*	\brief ���Ϸ��� (degree) �̿��Ͽ� ���ʹϿ��� �����Ѵ�.
*
*	\param theta1 ù��°�࿡ ���� ȸ������ (degree).
*	\param theta2 �ι�°�࿡ ���� ȸ������ (degree).
*	\param theta3 ����°�࿡ ���� ȸ������ (degree).
*	\param eulerType ���Ϸ��� (XYZ, ZYX, ...)
*
*	\return ������ �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetFromEulerAngle(double theta1, double theta2, double theta3, EulerType eulerType)
{
	double c1, c2, c3;
	double s1, s2, s3;
	theta1 = DEG2RAD(theta1);
	theta2 = DEG2RAD(theta2);
	theta3 = DEG2RAD(theta3);
	c1 = cos(theta1);
	c2 = cos(theta2);
	c3 = cos(theta3);
	s1 = sin(theta1);
	s2 = sin(theta2);
	s3 = sin(theta3);

	GMatrix mat;
	switch (eulerType)
	{
	case EULER_XYZ:
		mat[0][0] = c2 * c3;	
		mat[0][1] = -c2 * s3;	
		mat[0][2] = s2;
		mat[0][3] = 0.0;

		mat[1][0] = s1 * s2 * c3 + c1 * s3;
		mat[1][1] = -s1 * s2 * s3 + c1 * c3;
		mat[1][2] = -s1 * c2;
		mat[1][3] = 0.0;

		mat[2][0] = -c1 * s2 * c3 + s1 * s3;
		mat[2][1] = c1 * s2 * s3 + s1 * c3;
		mat[2][2] = c1 * c2;
		mat[2][3] = 0.0;

		mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
		mat[3][3] = 1.0;
		break;
	case EULER_ZYX:
		mat[0][0] = c3 * c2;	
		mat[0][1] = -s3 * c1 + c3 * s2 * s1;
		mat[0][2] = s3 * s1 + c3 * s2 * c1;
		mat[0][3] = 0.0;

		mat[1][0] = s3 * c2;
		mat[1][1] = c3 * c1 + s3 * s2 * s1;
		mat[1][2] = -c3 * s1 + s3 * s2 * c1;
		mat[1][3] = 0.0;

		mat[2][0] = -s2;
		mat[2][1] = c2 * s1;
		mat[2][2] = c2 * c1;
		mat[2][3] = 0.0;

		mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
		mat[3][3] = 1.0;
		break;
	}
	return SetFromMatrix(mat);
}

/*!
*	\brief �� ���� �������� ȸ���ϴ� ���ʹϿ��� �����Ѵ�.
*
*	\param x_axis x ���� ��Ÿ����.
*	\param y_axis y ���� ��Ÿ����.
*
*	\return ������ �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetFromFrameXY(const GVector3 &x_axis, const GVector3 &y_axis)
{
	GVector3 X(x_axis), Y(y_axis), Z;
	Z = X ^ Y;
	X.Normalize();
	Y.Normalize();
	Z.Normalize();

	GMatrix mat;
	mat.SetIdentity();

	mat[0][0] = X[0];
	mat[1][0] = X[1];
	mat[2][0] = X[2];

	mat[0][1] = Y[0];
	mat[1][1] = Y[1];
	mat[2][1] = Y[2];

	mat[0][2] = Z[0];
	mat[1][2] = Z[1];
	mat[2][2] = Z[2];

	return SetFromMatrix(mat);
}

/*!
*	\brief �� ���� �������� ȸ���ϴ� ���ʹϿ��� �����Ѵ�.
*
*	\param y_axis y ���� ��Ÿ����.
*	\param z_axis z ���� ��Ÿ����.
*
*	\return ������ �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetFromFrameYZ(const GVector3 &y_axis, const GVector3 &z_axis)
{
	GVector3 X, Y(y_axis), Z(z_axis);
	X = Y ^ Z;
	X.Normalize();
	Y.Normalize();
	Z.Normalize();

	GMatrix mat;
	mat.SetIdentity();

	mat[0][0] = X[0];
	mat[1][0] = X[1];
	mat[2][0] = X[2];

	mat[0][1] = Y[0];
	mat[1][1] = Y[1];
	mat[2][1] = Y[2];

	mat[0][2] = Z[0];
	mat[1][2] = Z[1];
	mat[2][2] = Z[2];

	return SetFromMatrix(mat);
}

/*!
*	\brief �� ���� �������� ȸ���ϴ� ���ʹϿ��� �����Ѵ�.
*
*	\param z_axis z ���� ��Ÿ����.
*	\param x_axis x ���� ��Ÿ����.
*
*	\return ������ �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetFromFrameZX(const GVector3 &z_axis, const GVector3 &x_axis)
{
	GVector3 X(x_axis), Y, Z(z_axis);
	Y = Z ^ X;
	X.Normalize();
	Y.Normalize();
	Z.Normalize();

	GMatrix mat;
	mat.SetIdentity();

	mat[0][0] = X[0];
	mat[1][0] = X[1];
	mat[2][0] = X[2];

	mat[0][1] = Y[0];
	mat[1][1] = Y[1];
	mat[2][1] = Y[2];

	mat[0][2] = Z[0];
	mat[1][2] = Z[1];
	mat[2][2] = Z[2];

	return SetFromMatrix(mat);
}

/*!
*	\brief ȸ����ķκ��� ���ʹϿ��� �����Ѵ�.
*
*	\param m ȸ����� (3 by 3, 4 by 4)
*
*	\return ������ �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetFromMatrix(const GMatrix &m)
{
	int r, c;
	r = m.GetRowNum();
	c = m.GetColNum();
	assert(r == 3 || r == 4);
	assert(c == 3 || c == 4);

	double q[4];
	double tr, s;
	int i,j,k;
	int nxt[3] = {1, 2, 0};

	tr = m[0][0] +  m[1][1] + m[2][2];

	if (tr > 0.0) 
	{
		s = SQRT(tr + 1.0);
		W = s * 0.5;
		s = 0.5 / s;
		X = (m[2][1] - m[1][2]) * s;
		Y = (m[0][2] - m[2][0]) * s;
		Z = (m[1][0] - m[0][1]) * s;
	} 
	else 
	{
		i = 0;
		if (m[1][1] > m[0][0]) 
			i = 1;
		if (m[2][2] > m[i][i]) 
			i = 2;
		j = nxt[i]; 
		k = nxt[j];
		s = SQRT((m[i][i] - (m[j][j] + m[k][k])) + 1.0);
		q[i] = s * 0.5;
		s = 0.5 / s;
		W = (m[k][j] - m[j][k]) * s;
		q[j] = (m[j][i] + m[i][j]) * s;
		q[k] = (m[k][i] + m[i][k]) * s;
		X = q[0];
		Y = q[1];
		Z = q[2];
	}
	Normalize();
	return *this;
}

/*!
*	\brief ���ʹϿ��� �׵������ ��ȯ�Ѵ�.
*
*	\return �׵������ ��ȯ�� �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetIdentity()
{
	this->W = 1.0;
	this->X = 0.0;
	this->Y = 0.0;
	this->Z = 0.0;
	return *this;
}

/*!
*	\brief ���ʹϿ��� �������� ��ȯ�Ѵ�.
*
*	\return �������� ��ȯ�� �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetInverse()
{
	if (!IsUnitQuater())
	{
		double norm_sqr = SQR(W) + SQR(X) + SQR(Y) + SQR(Z);
		*this /= norm_sqr;
	}
	SetConjugate();
	return *this;
}

/*!
*	\brief ���ʹϿ��� conjugate �Ѵ�.
*
*	\return conjugate�� �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetConjugate()
{
	this->X *= -1.0;
	this->Y *= -1.0;
	this->Z *= -1.0;
	return *this;
}

/*!
*	\brief ���ʹϿ��� negate �Ѵ�.
*
*	\return negate�� �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::SetNegate()
{
	this->W *= -1.0;
	this->X *= -1.0;
	this->Y *= -1.0;
	this->Z *= -1.0;
	return *this;
}

/*!
*	\brief ���ʹϿ��� �������ʹϿ����� ��ȯ�Ѵ�.
*
*	\return �������ʹϿ����� ��ȯ�� �ڽ��� �����Ѵ�.
*/
inline
GQuater &GQuater::Normalize()
{
	double len = norm(*this);
	this->W /= len;
	this->X /= len;
	this->Y /= len;
	this->Z /= len;
	return *this;
}

/*!
*	\brief ���ʹϿ����� ǥ���Ǵ� ȸ���� ����ȸ����� ������ ���Ѵ�.
*
*	\param axis ȸ������ ����ȴ�.
*	\param angle ������ ����ȴ�.
*	\param radian \a angle�� ����� ������ ������̸� true, �ƴϸ� false.
*/
inline
void GQuater::GetAngleAxis(GVector3 &axis, double &angle, bool radian) const
{
	if (IsIdentity() || !IsUnitQuater())
	{
		angle = 0.0;
		axis.Set(1.0, 0.0, 0.0);
		return;
	}

	double cs, sn;
	cs = W;
	sn = SQRT(SQR(X) + SQR(Y) + SQR(Z));
	angle = atan2(sn, cs);
	axis.Set(X / sn, Y / sn, Z / sn);
	if (radian)
		angle = angle * 2.0;
	else
		angle = angle * 360.0 / M_PI;
}

/*!
*	\brief ���ʹϿ����� ǥ���Ǵ� ȸ���� ���Ϸ� ǥ���� ���Ѵ�.
*
*	\param theta1 ù��° ����
*	\param theta2 �ι�° ����
*	\param theta3 ����° ����
*	\param radian true: ����, false: ����
*	\param eulerType EULER_XYZ or EULER_ZYX
*/
inline
void GQuater::GetEulerAngle(double &theta1, double &theta2, double &theta3, bool radian, EulerType eulerType)
{
	GMatrix mat = cast_mat(*this, false);
	switch (eulerType)
	{
	case EULER_XYZ:
		theta1 = atan2(-mat[1][2], mat[2][2]);
		theta2 = asin(mat[0][2]);
		theta3 = atan2(-mat[0][1], mat[0][0]);
		break;
	case EULER_ZYX:
		theta1 = atan2(mat[2][1], mat[2][2]);
		theta2 = asin(-mat[2][0]);
		theta3 = atan2(mat[1][0], mat[0][0]);
	    break;
	}
	if (!radian)
	{
		theta1 = RAD2DEG(theta1);
		theta2 = RAD2DEG(theta2);
		theta3 = RAD2DEG(theta3);
	}
}

/*!
*	\brief ���ʹϿ��� W ������ ��ȯ�Ѵ�.
*
*	\return  ���ʹϿ��� W ����
*/
inline
double GQuater::GetW() const
{
	return W;
}

/*!
*	\brief ���ʹϿ��� X ������ ��ȯ�Ѵ�.
*
*	\return  ���ʹϿ��� X ����
*/
inline
double GQuater::GetX() const
{
	return X;
}

/*!
*	\brief ���ʹϿ��� Y ������ ��ȯ�Ѵ�.
*
*	\return  ���ʹϿ��� Y ����
*/
inline
double GQuater::GetY() const
{
	return Y;
}

/*!
*	\brief ���ʹϿ��� Z ������ ��ȯ�Ѵ�.
*
*	\return  ���ʹϿ��� Z ����
*/
inline
double GQuater::GetZ() const
{
	return Z;
}

/*!
*	\brief �������ʹϿ� ���θ� �����Ѵ�.
*
*	\return true: �������ʹϿ�, false: �������ʹϿ��� �ƴѰ��
*/
inline
bool GQuater::IsUnitQuater() const
{
	double norm = SQR(W) + SQR(X) + SQR(Y) + SQR(Z);
	return EQ(norm, 1.0, Precision) ? true : false;
}

/*!
*	\brief ���� ���ʹϿ��� �׵������ ���θ� �����Ѵ�.
*
*	\return true: �׵���� ���, false: �׵���� �ƴѰ��
*/
inline
bool GQuater::IsIdentity() const
{
	return (EQ(W, 1.0, 1.0e-5) && EQ(X, 0.0, 1.0e-5) && 
		EQ(Y, 0.0, 1.0e-5) && EQ(Z, 0.0, 1.0e-5));
}

/*!
*	\brief ������ (*) �ߺ�
*
*	\param lhs ���� �ǿ�����
*	\param s ������ �ǿ����� (�Ǽ�)
*
*	\return \a s�� ������ ���ʹϿ��� �����Ѵ�.
*/
inline
GQuater operator *(const GQuater &lhs, const double &s)
{
	GQuater ret(lhs);
	ret *= s;
	return ret;
}

/*!
*	\brief ������ (*) �ߺ�
*
*	\param s ���� �ǿ����� (�Ǽ�)
*	\param rhs ������ �ǿ����� (���ʹϿ�)
*
*	\return \a s �� ������ ���ʹϿ�
*/
inline
GQuater operator *(const double &s, const GQuater &rhs)
{
	GQuater ret(rhs);
	ret *= s;
	return ret;
}

/*!
*	\brief ���ʹϿ��� ũ�⸦ ���Ѵ�.
*
*	\param q ũ�⸦ ���� ���ʹϿ�
*
*	\return ���ʹϿ� \a q�� ũ��
*/
inline
double norm(const GQuater &q)
{
	return SQRT(SQR(q.W) + SQR(q.X) + SQR(q.Y) + SQR(q.Z));
}

/*!
*	\brief ���ʹϿ��� ������ ���ϴ� �Լ�.
*
*	\param q ������ ���� ���ʹϿ�.
*
*	\return ���ʹϿ� \a q�� ����.
*/
inline
GQuater inv(const GQuater &q)
{
	GQuater ret(q);
	if (!ret.IsUnitQuater())
	{
		double norm_sqr = SQR(ret.W) + SQR(ret.X) + SQR(ret.Y) + SQR(ret.Z);
		ret /= norm_sqr;
	}
	ret.SetConjugate();
	return ret;
}

/*!
*	\brief �־��� ���� (pure imaginary ���ʹϿ�)�� exponential mapping�� ���Ѵ�.
*
*	\param v pure imaginary ���ʹϿ� q = (0, v).
*
*	\return ����ȸ���� v/||v|| �� �߽����� 2 * ||v|| �� ȸ������ ���� ���ʹϿ�.
*/
inline
GQuater exp(const GVector3 &v)
{
	double nm, s;
	GQuater rq;

	nm = norm(v);
	if (EQ(nm, 0.0, GQuater::Precision))
	{
		rq.SetIdentity();
		return rq;
	}

	rq.W = cos(nm);
	s = sin(nm) / nm;
	rq.X = s * v[0];
	rq.Y = s * v[1];
	rq.Z = s * v[2];

	return rq;
}

/*!
*	\brief ���ʹϿ� q�� �α��Լ����� ���Ѵ�.
*
*	\param q ���ʹϿ�
*
*	\return ����� ���ʹϿ��� ����κ� Log(q) = (0, x, y, z).
*/
inline
GVector3 log(const GQuater &q)
{
	GVector3 rv;
	if (q.IsIdentity())
	{
		rv.Set(0.0, 0.0, 0.0);
		return rv;
	}
	double theta, s;
	theta = acos(q.W);
	s = theta / sin(theta);
	rv.Set(s * q.X, s * q.Y, s * q.Z);
	return rv;
}

/*!
*	\brief ���ʹϿ� q1, q2�� ���� �Ѵ�.
*
*	\param q1 ���ʹϿ�
*	\param q2 ���ʹϿ�
*
*	\return ������ ��
*/
inline
double quat_dot(const GQuater &q1, const GQuater &q2)
{
	return q1.W * q2.W + q1.X * q2.X + q1.Y * q2.Y + q1.Z * q2.Z;
}

/*!
*	\brief ���ʹϿ� q1, q2�� Spherical linear interpolation �Ѵ�.
*
*	\param q1 ���ʹϿ�
*	\param q2 ���ʹϿ�
*	\param t ������ �ð�
*
*	\return ������ ���ʹϿ�
*/
inline
GQuater slerp(const GQuater &q1, const GQuater &q2, const double t)
{
	GQuater q;
	double c = q1.W * q2.W + q1.X * q2.X + q1.Y * q2.Y + q1.Z * q2.Z;
	
	if (1.0 + c > 1.0e-5)
	{
		if (1.0 - c > 1.0e-5)
		{
			double theta = acos(c);
			q = (q1 * sin((1.0 - t) * theta) + q2 * sin(t * theta)) / sin(theta);
		}
		else
			q = q1 * (1.0 - t) + q2 * t;
	}
	else	
		q = q1 * sin((0.5 - t) * M_PI) + q2 * sin(t * M_PI);
	
	q.Normalize();
	return q;
}

/*!
*	\brief ���ʹϿ����� ǥ���Ǵ� ȸ���� 4�� 4���� ������ǥ��ȯ��ķ� ��ȯ�Ѵ�.
*
*	\param q ������ʹϿ�
*	\param isGL ���� true �̸� ��Ŀ� OpenGL ���·� �����Ѵ�.
*
*	\return ȸ���� ǥ���ϴ� ���
*/
inline
GMatrix cast_mat(const GQuater &q, bool isGL)
{
	GMatrix ret(4, 4);

	double s = 2.0 / (q.W * q.W + q.X * q.X + q.Y * q.Y + q.Z * q.Z);
	double xs = q.X * s, ys = q.Y * s, zs = q.Z * s;
	double wx = q.W * xs, wy = q.W * ys, wz = q.W * zs;
	double xx = q.X * xs, xy = q.X * ys, xz = q.X * zs;
	double yy = q.Y * ys, yz = q.Y * zs, zz = q.Z * zs;

	ret[0][0] = 1.0 - (yy + zz); 
	ret[1][0] = xy + wz; 
	ret[2][0] = xz - wy;

	ret[0][1] = xy - wz;	
	ret[1][1] = 1.0 - (xx + zz); 
	ret[2][1] = yz + wx;

	ret[0][2] = xz + wy;	
	ret[1][2] = yz - wx; 
	ret[2][2] = 1.0 - (xx + yy);

	ret[0][3] = ret[1][3] = ret[2][3] = 0.0;
	ret[3][0] = ret[3][1] = ret[3][2] = 0.0;
	ret[3][3] = 1.0;

	if (isGL)
		ret.SetTranspose();

	return ret;
}
// inline
// GMatrix cast_mat(const GQuater &q, bool isGL)
// {
// 	GMatrix ret(4, 4);
// 
// 	if (q.IsUnitQuater())
// 	{
// 		double xx = SQR(q.X), yy = SQR(q.Y), zz = SQR(q.Z), ww = SQR(q.W);
// 		double Nq = xx + yy + zz + ww;
// 		double wx2 = 2.0 * q.W * q.X, wy2 = 2.0 * q.W * q.Y;
// 		double wz2 = 2.0 * q.W * q.Z, xy2 = 2.0 * q.X * q.Y;
// 		double xz2 = 2.0 * q.X * q.Z, yz2 = 2.0 * q.Y * q.Z;
// 
// 		ret[0][0] = ww + xx - yy - zz; 
// 		ret[1][0] = xy2 + wz2; 
// 		ret[2][0] = xz2 - wy2;
// 
// 		ret[0][1] = xy2 - wz2;	
// 		ret[1][1] = ww - xx + yy - zz;
// 		ret[2][1] = yz2 + wx2;
// 
// 		ret[0][2] = xz2 + wy2;	
// 		ret[1][2] = yz2 - wx2; 
// 		ret[2][2] = ww - xx - yy + zz;
// 
// 		ret[0][3] = ret[1][3] = ret[2][3] = 0.0;
// 		ret[3][0] = ret[3][1] = ret[3][2] = 0.0;
// 		ret[3][3] = Nq;
// 	}
// 	else
// 	{
// 		double Nq = q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W;
// 		double s = (Nq > 0.0) ? (2.0 / Nq) : 0.0;
// 		double xs = q.X * s, ys = q.Y * s, zs = q.Z * s;
// 		double wx = q.W * xs, wy = q.W * ys, wz = q.W * zs;
// 		double xx = q.X * xs, xy = q.X * ys, xz = q.X * zs;
// 		double yy = q.Y * ys, yz = q.Y * zs, zz = q.Z * zs;
// 
// 		ret[0][0] = 1.0 - (yy + zz); 
// 		ret[1][0] = xy + wz; 
// 		ret[2][0] = xz - wy;
// 
// 		ret[0][1] = xy - wz;	
// 		ret[1][1] = 1.0 - (xx + zz); 
// 		ret[2][1] = yz + wx;
// 
// 		ret[0][2] = xz + wy;	
// 		ret[1][2] = yz - wx; 
// 		ret[2][2] = 1.0 - (xx + yy);
// 
// 		ret[0][3] = ret[1][3] = ret[2][3] = 0.0;
// 		ret[3][0] = ret[3][1] = ret[3][2] = 0.0;
// 		ret[3][3] = 1.0;
// 	}
// 	if (isGL)
// 		ret.SetTranspose();
// 
// 	return ret;
// }

/*!
*	\brief ���ʹϿ��� 4���� ���ͷ� ��ȯ�Ѵ�.
*
*	\param q ���ʹϿ�
*
*	\return 4���� ����
*/
inline
GVector cast_vec(const GQuater &q)
{
	return GVector(4, q.W, q.X, q.Y, q.Z);
}

/*!
*	\brief ��¿����� �ߺ�
*
*	\param os ��½�Ʈ��
*	\param q ����� ���ʹϿ�
*
*	\return ��µ� ��Ʈ��
*/
inline
ostream &operator <<(ostream &os, const GQuater &q)
{
	os << "(" << q.W << ") + " << "(" << q.X << ")i + " << "(" << q.Y << ")j + " << "(" << q.Z << ")k" << endl;
	return os;
}

/*!
*	\brief �Է¿����� �ߺ�
*
*	\param is �Է½�Ʈ��
*	\param q �Է¹��� ���ʹϿ�
*
*	\return �Էµ� ��Ʈ��
*/
inline
istream &operator >>(istream &is, GQuater &q)
{
	is >> q.W >> q.X >> q.Y >> q.Z;
	return is;
}

/*!
*	\brief 4���� ���͸� ���ʹϿ��� ���·� ��ȯ�Ѵ�.
*
*	\param v 4���� ����
*
*	\return ��ȯ�� ���ʹϿ�ǥ��
*/
inline
GQuater cast_quat(const GVector &v)
{
	return GQuater(v[0], v[1], v[2], v[3]);
}

/*!
*	\class GLine
*	\brief 3���� �������� ������ ��Ÿ���� Ŭ����
*
*	\author ������ (shyun@dongguk.edu)
*	\date 01 Jan 2001	
*/
class GLine
{
	// ������ �Լ� �� Ŭ����
	friend bool operator ==(const GLine &lhs, const GLine &rhs);
	friend bool operator !=(const GLine &lhs, const GLine &rhs);
	friend ostream &operator <<(ostream &os, const GLine &l);
	friend double dist(const GLine &l, const GPoint3 &q);
	friend bool intersect_line_plane(GPoint3 &p, const GLine &l, const GPlane &pi);
	friend bool intersect_line_triangle(GPoint3 &q, const GLine &l, const GPoint3 &p1, const GPoint3 &p2, const GPoint3 &p3, double &t, double &alpha, double &beta, const bool bCull = true);
	friend bool intersect_planes(GLine &l, const GPlane &pi1, const GPlane &pi2);
	friend GLine operator *(const GTransf &X, const GLine &l);

public:
	// ������ �� �Ҹ���
	GLine(const GPoint3 &_p = GPoint3(0, 0, 0), const GVector3 &_v = GVector3(0, 0, 0));
	GLine(const GPoint3 &_p, const GPoint3 &_q);
	GLine(const GLine &cpy);
	virtual ~GLine();

	// ���Կ�����
	GLine &operator =(const GLine &rhs);
	
	// ����Լ�
	GLine &SetPt(const GPoint3 &_p);
	GLine &SetDir(const GVector3 &_v);
	GPoint3 GetPt() const;
	GVector3 GetDir() const;
	bool IsOnLine(const GPoint3 &q) const;
	GPoint3 Eval(const double t) const;

	// ��������Լ�
	static void SetPrecision(double err);
	static double GetPrecision();

protected:
	// ������ ���.
	GPoint3 p;
	GVector3 v;		// �����Ǵ� ������ ������: l(t) = p + v * t
	static double Precision;
};

/*!
*	\brief �������� �� ���� �������� ������ �����ϴ� ������.
*	
*	\param _p �������� �� ��
*	\param _v ������ ���� ����.
*/
inline
GLine::GLine(const GPoint3 &_p, const GVector3 &_v)
: p(_p), v(_v)
{
}

/*!
*	\brief 3���� �������� �� ���� ������ ������ �����ϴ� ������.
*	
*	\param _p �������� ��, l(0) = p.
*	\param _q �������� ��, l(1) = q.
*/
inline
GLine::GLine(const GPoint3 &_p, const GPoint3 &_q)
{
	p = _p;
	v = _q - _p;
}

/*!
*	\brief ���������.
*	
*	\param copy ����� ������ ��ü.
*/
inline
GLine::GLine(const GLine &copy)
: p(copy.p), v(copy.v)
{
}

/*!
*	\brief �Ҹ���.
*/
inline
GLine::~GLine()
{
}

/*!
*	\brief	���Կ����� �ߺ�
*	
*	\param rhs ������ �ǿ�����.
*
*	\return ���Ե� �ڽ��� ��ȯ�Ѵ�.
*/
inline
GLine &GLine::operator =(const GLine &rhs)
{
	this->p = rhs.p;
	this->v = rhs.v;
	return *this;
}

/*!
*	\brief	�Լ�ȣ�⿬���� �ߺ�
*			- �������� �Ű����� t������ �� l(t)�� ���Ѵ�.
*	
*	\param t �Ű�����.
*
*	\return �������� �� l(t)�� ��ȯ�Ѵ�.
*/
inline
GPoint3 GLine::Eval(const double t) const
{
	return p + t * v;
}

/*!
*	\brief	��ȣ�Կ����� �ߺ�.
*	
*	\param rhs ������ �ǿ�����.
*
*	\return true: ������, false: �ٸ���.
*/
inline
bool operator ==(const GLine &lhs, const GLine &rhs)
{
	if (lhs.IsOnLine(rhs.p) && EQ_ZERO(norm(lhs.v ^ rhs.v), GLine::Precision))
		return true;
	else
		return false;
}

/*!
*	\brief	�ε�ȣ�Կ����� �ߺ�.
*	
*	\param rhs ������ �ǿ�����.
*
*	\return true: �ٸ���, false: ������.
*/
inline
bool operator !=(const GLine &lhs, const GLine &rhs)
{
	return !(lhs == rhs);
}

/*!
*	\brief �� q�� �������� ������ �����Ѵ�.
*	
*	\param q ������ ��.
*
*	\return true: q�� �������� �� �ϰ��, false: �ƴҰ��.
*/
inline
bool GLine::IsOnLine(const GPoint3 &q) const
{
	GVector3 w = q - p;
	double s = norm(v ^ w);
	if (EQ_ZERO(s, Precision))
		return true;
	else
		return false;
}

/*!
*	\brief ��¿����� �ߺ�.
*
*	\param os ��½�Ʈ��
*	\param l ����� ���ΰ�ü
*
*	\return ��µ� ��Ʈ��
*/
inline
ostream &operator <<(ostream &os, const GLine &l)
{
	os << "(" 
		<< l.p[0] << " + (" << l.v[0] << ") * t, "
		<< l.p[1] << " + (" << l.v[1] << ") * t, "
		<< l.p[2] << " + (" << l.v[2] << ") * t)";
	return os;
}

/*!
*	\brief �� ������ ���������� �Ÿ��� ���Ѵ�.
*
*	\param l ����
*	\param q �� ��
*
*	\return �� \a q���� ���� \a l������ �����Ÿ�
*/
inline
double dist(const GLine &l, const GPoint3 &q)
{
	return norm(proj(q - l.p, l.v) - (q - l.p));
}

/*!
*	\brief ���ο� ���� �������� ������ �������� �����Ѵ� (������ �������).
*
*	\param _p ���ο� ��
*
*	\return ����� �ڽ��� ��ȯ�Ѵ�.
*/
inline
GLine &GLine::SetPt(const GPoint3 &_p)
{
	p = _p;
	return *this;
}

/*!
*	\brief ������ ������ �����Ѵ�.
*
*	\param _v ���ο� ����.
*
*	\return ����� �ڽ��� ��ȯ�Ѵ�.
*/
inline
GLine &GLine::SetDir(const GVector3 &_v)
{
	v = _v;
	return *this;
}

/*!
*	\brief ������ �������� �����Ѵ�.
*
*	\return ������ ������.
*/
inline
GPoint3 GLine::GetPt() const
{
	return p;
}

/*!
*	\brief ������ ������ �����Ѵ�.
*
*	\return ������ ����.
*/
inline
GVector3 GLine::GetDir() const
{
	return v;
}

/*!
*	\brief ��ġ������ �Ѱ�ġ�� �����Ѵ�.
*/
inline
void GLine::SetPrecision(double err)
{
	Precision = err;
}

/*!
*	\brief ��ġ������ �Ѱ�ġ�� �����Ѵ�.
*
*	\return �Ѱ�ġ.
*/
inline
double GLine::GetPrecision()
{
	return Precision;
}

/*!
*	\class GPlane
*	\brief ����� ǥ���ϴ� Ŭ����
*
*	\author ������ (shyun@dongguk.edu)
*	\date 01 Jan 2001	
*/
class GPlane
{
	// ������ �Լ� �� Ŭ����
	friend bool operator ==(const GPlane &lhs, const GPlane &rhs);
	friend bool operator !=(const GPlane &lhs, const GPlane &rhs);
	friend ostream &operator <<(ostream &os, const GPlane &pi);
	friend GPlane operator *(const GTransf &X, const GPlane &pi);
	friend double dist(const GPlane &pi, const GPoint3 &p);
	friend GPoint3 proj(const GPoint3 &p, const GPlane &pi);
	friend bool intersect_line_plane(GPoint3 &p, const GLine &l, const GPlane &pi);
	friend bool intersect_planes(GLine &l, const GPlane &pi1, const GPlane &pi2);
	friend bool intersect_planes(GPoint3 &p, const GPlane &pi1, const GPlane &pi2, const GPlane &pi3);

public:
	// ������ �� �Ҹ���
	GPlane(const GVector3 &_n, const GPoint3 &_p);
	GPlane(const GPoint3 &p1, const GPoint3 &p2, const GPoint3 &p3);
	GPlane(const GVector3 &_n, const double &_d);
	GPlane(const double &a, const double &b, const double &c, const double &d);
	GPlane(const GPlane &cpy);
	virtual ~GPlane();

	// ���Կ�����
	GPlane &operator =(const GPlane &rhs);
	
	// ����Լ�
	GVector3 GetNormal() const;
    bool IsOnPlane(const GPoint3 &p) const;
	bool IsAbovePlane(const GPoint3 &p) const;
	bool IsBelowPlane(const GPoint3 &p) const;

	// �����Լ�
	static void SetPrecision(const double err);
	static double GetPrecision();
    
protected:
	// ������ ���
	GVector3 n;		// ����� ������ ax + by + cz + d = 0 ���� n = (a, b, c)
	double d;		// n = (a, b, c), d = -n * p (p: ������� ������ ��)
	static double Precision;
};

/*!
*	\brief ����� ������.
*	
*	\param _n �����Ǵ� ����� ��������
*	\param _p �����Ǵ� ������� ������ ����
*/
inline
GPlane::GPlane(const GVector3 &_n, const GPoint3 &_p)
{
	n = _n;
	d = -(n[0] * _p[0] + n[1] * _p[1] + n[2] * _p[2]);
}

/*!
*	\brief �� �� p1, p2, p3 �� ������ ����� ������
*	
*	\param p1 ������ ��.
*	\param p2 ������ ��.
*	\param p3 ������ ��.
*/
inline
GPlane::GPlane(const GPoint3 &p1, const GPoint3 &p2, const GPoint3 &p3)
{
	n = (p2 - p1) ^ (p3 - p1);
	d = -(n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2]);
}

/*!
*	\brief ����� ���������
*	
*	\param copy ����� ����� ��ü�� ���� ���۷���
*/
inline
GPlane::GPlane(const GPlane &cpy)
: n(cpy.n), d(cpy.d)
{
}

/*!
*	\brief ����� ������.
*	
*	\param _n ��������.
*	\param _d ��� d = - n * p (p: ������� ������ ��).
*/
inline
GPlane::GPlane(const GVector3 &_n, const double &_d)
: n(_n), d(_d)
{
}

/*!
*	\brief ����� ������.
*	
*	\param a, b, c, d ��� ������ ax + by + cz +d = 0 �� ���
*/
inline
GPlane::GPlane(const double &a, const double &b, const double &c, const double &d)
{
	this->n = GVector3(a, b, c);
	this->d = d;
}

/*!
*	\brief ����� �Ҹ���.
*/
inline
GPlane::~GPlane()
{
}

/*!
*	\brief ���Կ����� �ߺ�
*	
*	\param rhs ������ �ǿ�����
*
*	\return ���Ե� �ڽ��� ��ȯ�Ѵ�.
*/
inline
GPlane &GPlane::operator =(const GPlane &rhs)
{
	this->n = rhs.n;
	this->d = rhs.d;
	return *this;
}

/*!
*	\brief ��ȣ�Կ����� �ߺ�.
*	
*	\param lhs ���� �ǿ�����.
*	\param rhs ������ �ǿ�����.
*
*	\return true: ������, false: �ٸ���.
*/
inline
bool operator ==(const GPlane &lhs, const GPlane &rhs)
{
	GVector pi1(4), pi2(4);
	pi1.Set(lhs.n[0], lhs.n[1], lhs.n[2], lhs.d);
	pi2.Set(rhs.n[0], rhs.n[1], rhs.n[2], rhs.d);
	double s1 = norm(pi1) * norm(pi2);
	double s2 = pi1 * pi2;

	if (EQ(s1, s2, GPlane::Precision))
		return true;
	else
		return false;
}

/*!
*	\brief �ε�ȣ�Կ����� �ߺ�
*	
*	\param lhs ���� �ǿ�����
*	\param rhs ������ �ǿ�����
*
*	\return true: �ٸ���, false: ������.
*/
inline
bool operator !=(const GPlane &lhs, const GPlane &rhs)
{
	return !(lhs == rhs);
}

/*!
*	\brief �������͸� ��ȯ�Ѵ�.
*
*	\return ��������.
*/
inline
GVector3 GPlane::GetNormal() const
{
	return n;
}

/*!
*	\brief �� p�� ������ �� ������ ����.
*	
*	\param p ������ ��
*
*	\return true: �� p�� ������ �� �ϰ��, false: �׷��� �������.
*/
inline
bool GPlane::IsOnPlane(const GPoint3 &p) const
{
	double s;
	s = -(n[0] * p[0] + n[1] * p[1] + n[2] * p[2]);
	if (EQ(s, d, GPlane::Precision))
		return true;
	else
		return false;
}

/*!
*	\brief �� p�� ����� ���� �������� ����.
*	
*	\param p ������ ��
*
*	\return true: �� p�� ����� ���� �� �ϰ��, false: �׷��� �������.
*/
inline
bool GPlane::IsAbovePlane(const GPoint3 &p) const
{
	double s;
	s = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d;
	if (s > 0.0)
		return true;
	else
		return false;
}

/*!
*	\brief �� p�� ����� �Ʒ��� �������� ����.
*	
*	\param p ������ ��
*
*	\return true: �� p�� ����� �Ʒ��� �� �ϰ��, false: �׷��� �������.
*/
inline
bool GPlane::IsBelowPlane(const GPoint3 &p) const
{
	double s;
	s = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d;
	if (s < 0.0)
		return true;
	else
		return false;
}

/*!
*	\brief ��ġ������ �Ѱ�ġ�� �����Ѵ�.
*/
inline
void GPlane::SetPrecision(const double err)
{
	Precision = err;
}

/*!
*	\brief ��ġ������ �Ѱ�ġ�� �����Ѵ�.
*
*	\return �Ѱ�ġ.
*/
inline
double GPlane::GetPrecision()
{
	return Precision;
}

/*!
*	\brief ��¿����� �ߺ�.
*
*	\param os ��½�Ʈ��.
*	\param pi ����� ��鰴ü.
*
*	\return ��½�Ʈ��
*/
inline
ostream &operator <<(ostream &os, const GPlane &pi)
{
	os << "(" << pi.n[0] << ") * x + (" 
		<< pi.n[1] << ") * y + ("
        << pi.n[2] << ") * z + ("
		<< pi.d << ") = 0";
	return os;
}

/*!
*	\brief	��� pi���� �� p������ ��ȣ�ִ� �Ÿ��� ����Ѵ�.
*
*	\param pi ����� ��ü.
*	\param p ��� ���� ����.
*
*	\return ��� \a pi���� �� \a p������ ��ȣ�ִ� �Ÿ�
*/
inline
double dist(const GPlane &pi, const GPoint3 &p)
{
    double D;
	D = (p[0] * pi.n[0] + p[1] * pi.n[1] + p[2] * pi.n[2] + pi.d) / norm(pi.n);
	return D;
}

/*!
*	\brief	�� p�� ��� pi�� �翵��Ų ���� ���Ѵ�.
*
*	\param p ��� ���� ��.
*	\param pi ��鰴ü.
*/
inline
GPoint3 proj(const GPoint3 &p, const GPlane &pi)
{
	double s = -dist(pi, p) / norm(pi.n);
	return p + s * pi.n;
}

/*!
*	\brief	������ ����� �������� ����Ѵ�.
*
*	\param p �������� ����� ����.
*	\param l ���� ��ü.
*	\param pi ��� ��ü.
*
*	\return true: ����, false: ������ ���� �����ϰų�, ����� ������ �����ϴ� ���.
*/
inline
bool intersect_line_plane(GPoint3 &p, const GLine &l, const GPlane &pi)
{
	if (EQ_ZERO(l.v * pi.n, GPlane::Precision))
	{
		cout << "line is parallel to plane !" << endl;
		return false;
	}

	double t = -(l.p[0] * pi.n[0] + l.p[1] * pi.n[1] + l.p[2] * pi.n[2] + pi.d) / (l.v * pi.n);
	p = l.Eval(t);
    return true;
}

/*!
*	\brief	������ �ﰢ�� (p1p2p3)�� �������� ����Ѵ�.
*
*	\param q �������� ����� ��.
*	\param l ������ü.
*	\param p1 �ﰢ���� ����.
*	\param p2 �ﰢ���� ����.
*	\param p3 �ﰢ���� ����.
*	\param t q = l(t)�� �����ϴ� ������ �Ķ����
*	\param alpha barycentric ��ǥ
*	\param beta barycentric ��ǥ
*	\param bCull ������ �ﰢ���� �޸�� ������ ��츦 �����Ϸ���. (����Ʈ: true).
*
*	\return true: ����, false: ������ �ﰢ���� �����ϰų� �������� �ﰢ���� �ܺο� �����ϴ� ���.
*/
inline
bool intersect_line_triangle(GPoint3 &q, const GLine &l, const GPoint3 &p1, const GPoint3 &p2, const GPoint3 &p3, double &t, double &alpha, double &beta, const bool bCull)
{
	GVector3 e1, e2, u, v, w;
	double det;
	e1 = p2 - p1;
	e2 = p3 - p1;
	u = l.v ^ e2;
	det = e1 * u;

	if (bCull)	// ������ �ﰢ���� �޸�� ������ ���� ����.
	{
		if (det < GLine::Precision)
			return false;

		w = l.p - p1;
		alpha = w * u;
		if (alpha < 0.0 || alpha > det)
			return false;

		v = w ^ e1;
		beta = l.v * v;
		if (beta < 0.0 || alpha + beta > det)
			return false;

		t = e2 * v;

		alpha /= det;
		beta /= det;
		t /= det;
	}
	else
	{
		if (det > -GLine::Precision && det < GLine::Precision)
			return false;

		w = l.p - p1;
		alpha = w * u / det;
		if (alpha < 0.0 || alpha > 1.0)
			return false;

		v = w ^ e1;
		beta = l.v * v / det;
		if (beta < 0.0 || alpha + beta > 1.0)
			return false;

		t = e2 * v / det;
	}

	q = l.Eval(t);
	return true;
}

/*!
*	\brief	�� ����� ���������� ����Ѵ�.
*
*	\param l ���������� ����� ������ü.
*	\param pi1 ���1.
*	\param pi2 ���2.
*
*	\return true: ����, false: �� ����� ������ ���.
*/
inline
bool intersect_planes(GLine &l, const GPlane &pi1, const GPlane &pi2)
{
	double s = norm(pi1.n ^ pi2.n);
	if (EQ_ZERO(s, GPlane::Precision))
	{	
		cout << "two planes are parallel !" << endl;
		return false;
	}

	l.v = pi1.n ^ pi2.n;
	
	GPoint3 p(0, 0, 0);
	GPlane pi3(l.v, p);
	intersect_planes(p, pi1, pi2, pi3);

	l.p = p;
    return true;
}

/*!
*	\brief	�� ����� �������� ����Ѵ�.
*
*	\param p �������� ����� �� ��ü.
*	\param pi1 ���1.
*	\param pi2 ���2.
*	\param pi3 ���3.
*
*	\return true: ����, false: �� ��� �� ��� �� ����� ��ġ�ϰų� �����ϴ� ���.
*/
inline
bool intersect_planes(GPoint3 &p, const GPlane &pi1, const GPlane &pi2, const GPlane &pi3)
{
	GMatrix A(3, 3);
	GVector x(3), b(3), Idx(3);
	double tmp;

	A.SetRowVec(0, cast_vec(pi1.n));
	A.SetRowVec(1, cast_vec(pi2.n));
	A.SetRowVec(2, cast_vec(pi3.n));

	if (grank(A) != 3)
	{
		cout << "at least two planes are parallel !" << endl;
		return false;
	}

	b.Set(-pi1.d, -pi2.d, -pi3.d);
	dcmp_lu(A, Idx, tmp);

	lin_sys_lu(A, x, b, Idx);
	p[0] = x[0];
	p[1] = x[1];
	p[2] = x[2];
	return true;
}

/*!
*	\class GTransf
*	\brief ��ü��ȯ (rigid-body)�� ��Ÿ���� Ŭ����.
*
*	\author ������ (shyun@dongguk.edu)
*	\date 2001�� 1�� 1��.
*/
class GTransf
{
	// ������ �Լ� �� Ŭ����
	friend GTransf operator *(const GTransf &lhs, const GTransf &rhs);
	friend bool operator ==(const GTransf &lhs, const GTransf &rhs);
	friend bool operator !=(const GTransf &lhs, const GTransf &rhs);
	friend ostream &operator <<(ostream &os, const GTransf &X);
	friend GTransf operator *(const GTransf &lhs, const double &s);
	friend GTransf operator *(const double &s, const GTransf &rhs);
	friend GPoint3 operator *(const GTransf &X, const GPoint3 &p);
	friend GLine operator *(const GTransf &X, const GLine &l);
	friend GPlane operator *(const GTransf &X, const GPlane &pi);
	friend GVector3 operator *(const GTransf &X, const GVector3 &v);
	friend GTransf inv(const GTransf &X);
	friend GVector3 part_trans(const GTransf &X);
	friend GQuater part_rot(const GTransf &X);
	friend GMatrix cast_mat(const GTransf &X, bool isGL = true);
	friend double norm(const GTransf &X);
	friend double dist(const GTransf &X1, const GTransf &X2);
	friend GTransf ad(const GTransf &X1, const GTransf &dX, const GTransf &X2);
	friend void lerp(GTransf &X, const GTransf &X1, const GTransf &X2, const double t);
	friend void lerp(GPose &P, const GPose &P1, const GPose &P2, const double t);
	friend class GJoint;

public:
	// ������ �� �Ҹ���
	GTransf(const GVector3 &_t = GVector3(0.0, 0.0, 0.0), const GQuater &_q = GQuater(1.0, 0.0, 0.0, 0.0));
	GTransf(const GTransf &cpy);
	virtual ~GTransf();

	// ���Կ����� �� ���մ��Կ�����
	GTransf &operator =(const GTransf &rhs);
	GTransf &operator *=(const GTransf &rhs);
	GTransf &operator *=(const double &s);

	// ����Լ�
	void SetNegate();
	void SetTrans(const GVector3 &_t);
	void SetRot(const GQuater &_q);
   
protected:
	// ������ ���
	/*! \breif 3���� �̵���ȯ */
    GVector3 t;

	/*! \breif 3���� ȸ����ȯ */
	GQuater q;
};

/*!
*	\brief ����Ʈ ������.
*
*	\param _t �̵�����.
*	\param _q ȸ���� ��Ÿ���� ���ʹϿ�.
*/
inline
GTransf::GTransf(const GVector3 &_t, const GQuater &_q)
: t(_t), q(_q)
{
}

/*!
*	\brief ���� ������.
*
*	\param copy ����� ��ü.
*/
inline
GTransf::GTransf(const GTransf &cpy)
: t(cpy.t), q(cpy.q)
{
}

/*!
*	\brief �Ҹ���.
*/
inline
GTransf::~GTransf()
{
}

/*!
*	\brief ���Կ����� �ߺ�.
*
*	\param rhs ���Ե� �ǿ�����.
*
*	\return ����� �ڽ��� ��ȯ�Ѵ�.
*/
inline
GTransf &GTransf::operator =(const GTransf &rhs)
{
	this->t = rhs.t;
	this->q = rhs.q;
	return *this;
}

/*!
*	\brief ���մ��Կ�����
*
*	\param rhs ������ �ǿ�����.
*
*	\return ���Ե� �ڽ��� ��ȯ.
*/
inline
GTransf &GTransf::operator *=(const GTransf &rhs)
{
	t += q * rhs.t;
	q *= rhs.q;
	q.Normalize();
	return *this;
}

/*!
*	\brief ���� ���Կ�����
*
*	\param s ��� ��
*
*	\return ���Ե� �ڽ��� ��ȯ
*/
inline
GTransf &GTransf::operator *=(const double &s)
{
	t *= s;
	q *= s;
	return *this;
}

/*!
*	\brief ���������
*
*	\param lhs ���� �ǿ�����.
*	\param rhs ������ �ǿ�����.
*
*	\return ����� ��ü��ȯ�� ��ȯ
*/
inline
GTransf operator *(const GTransf &lhs, const GTransf &rhs)
{
	GTransf X(lhs);
	X *= rhs;
	return X;
}


/*!
*	\brief ��ȣ������ �ߺ�.
*
*	\param lhs �����ǿ�����
*	\param rhs �������ǿ�����
*
*	\return true: ������, false: �ٸ���.
*/
inline
bool operator ==(const GTransf &lhs, const GTransf &rhs)
{
    if (lhs.t == rhs.t && lhs.q == rhs.q)
		return true;
	else
		return false;
}

/*!
*	\brief �ε�ȣ������ �ߺ�.
*
*	\param lhs �����ǿ�����
*	\param rhs �������ǿ�����
*
*	\return true: �ٸ���, false: ������.
*/
inline
bool operator !=(const GTransf &lhs, const GTransf &rhs)
{
	return !(lhs == rhs);
}

/*!
*	\brief ȸ���� negate�Ѵ�.
*/
inline
void GTransf::SetNegate()
{
	q.SetNegate();
}

/*!
*	\brief �̵���ȯ�� �����Ѵ�.
*
*	\param _t ������ �̵���ȯ
*/
inline
void GTransf::SetTrans(const GVector3 &_t)
{
	t = _t;
}

/*!
*	\brief ȸ����ȯ�� �����Ѵ�.
*
*	\param _q ������ ȸ����ȯ
*/
inline
void GTransf::SetRot(const GQuater &_q)
{
	q = _q;
}

/*!
*	\brief ��¿����� �ߺ�.
*
*	\param os ��½�Ʈ��.
*	\param X ����� ��ü��ȯ.
*
*	\return ��µ� ��Ʈ��
*/
inline
ostream &operator <<(ostream &os, const GTransf &X)
{
	os << cast_mat(X, false);
	return os;
}

/*!
*	\brief ���������
*
*	\param lhs ���� �ǿ�����
*	\param s ���
*
*	\return \a s��ŭ ������ �� ��ȯ
*/
inline
GTransf operator *(const GTransf &lhs, const double &s)
{
	GTransf ret(lhs);
	ret *= s;
	return ret;
}

/*!
*	\brief * ������ �ߺ�
*
*	\param s ���
*	\param rhs ���� �ǿ�����
*	\return \a s��ŭ ������ �� ��ȯ
*/
inline
GTransf operator *(const double &s, const GTransf &rhs)
{
	GTransf ret(rhs);
	ret *= s;
	return ret;
}

/*!
*	\brief ���������� �ߺ�.
*
*	\param X ��ü��ȯ ��ü.
*	\param p ������ǥ����� �� ��.
*
*	\return ���� (����) ��ǥ�迡�� �� p�� ��ǥ.
*/
inline
GPoint3 operator *(const GTransf &X, const GPoint3 &p)
{
	return X.q * p + X.t;
}

/*!
*	\brief ���������� �ߺ�.
*
*	\param X ��ü��ȯ ��ü.
*	\param v ������ǥ����� �� ����.
*
*	\return ���� (����) ��ǥ�迡�� ���� v�� ����.
*	\warning ������ ���, �̵������� ������ ���� �ʴ´�.
*/
inline
GVector3 operator *(const GTransf &X, const GVector3 &v)
{
	return X.q * v;
}

/*!
*	\brief ���������� �ߺ�.
*
*	\param X ��ü��ȯ ��ü.
*	\param l ������ǥ��󿡼� ǥ���� ���� ��ü.
*
*	\return ���� (����) ��ǥ�迡�� ǥ���� ���� l.
*/
inline
GLine operator *(const GTransf &X, const GLine &l)
{
	GPoint3 p = X * l.p;
	GVector3 v = X * l.v;
	return GLine(p, v);
}

/*!
*	\brief ���������� �ߺ�.
*
*	\param X ��ü��ȯ ��ü.
*	\param pi ������ǥ��� ǥ���� ����� ��ü.
*
*	\return ���� (����) ��ǥ�迡�� ǥ���� ��� pi.
*/
inline
GPlane operator *(const GTransf &X, const GPlane &pi)
{
	GVector3 n = X * pi.n;
	double d = pi.d - pi.n * (inv(X.q) * X.t);
    return GPlane(n, d);
}

/*!
*	\brief ��ü��ȯ�� ����ȯ�� ���Ѵ�.
*
*	\param X ��ü��ȯ ��ü.
*
*	\return ��ü��ȯ X�� ����ȯ.
*/
inline
GTransf inv(const GTransf &X)
{
	GQuater q = inv(X.q);
	GVector3 t = -(q * X.t);
	return GTransf(t, q);
}

/*!
*	\brief ��ü��ȯ�� �̵������� ���Ѵ�.
*
*	\param X ��ü��ȯ ��ü.
*
*	\return ��ü��ȯ X�� �̵�����
*/
inline
GVector3 part_trans(const GTransf &X)
{
	return X.t;
}

/*!
*	\brief ��ü��ȯ�� ȸ�������� ���Ѵ�.
*
*	\param X ��ü��ȯ ��ü.
*
*	\return ��ü��ȯ X�� ȸ������
*/
inline
GQuater part_rot(const GTransf &X)
{
	return X.q;
}

/*!
*	\brief ��ü��ȯ X�� �ش��ϴ� 4�� 4���� ������ǥ��ȯ��ķ� ��ȯ�Ѵ�.
*	\param X ��ü��ȯ
*	\param isGL ���� true �̸� ��Ŀ� OpenGL ���·� �����Ѵ�.
*
*	\return ��ü��ȯ�� ǥ���ϴ� ���
*/
inline
GMatrix cast_mat(const GTransf &X, bool isGL)
{
	GMatrix R;
	R = cast_mat(X.q, isGL);
	if (isGL)
	{
		R[3][0] = X.t[0];
		R[3][1] = X.t[1];
		R[3][2] = X.t[2];
	}
	else
	{
		R[0][3] = X.t[0];
        R[1][3] = X.t[1];
        R[2][3] = X.t[2];
	}
	return R;
}

/*!
*	\brief ��ȯ X�� ũ�⸦ ���Ѵ�.
*
*	\param X ��ü��ȯ
*
*	\return ��ȯ X�� ũ��
*/
inline
double norm(const GTransf &X)
{
	return norm(X.t) + norm(log(X.q));
}

/*!
*	\brief �� GTransf ���� �Ÿ��� ���Ѵ�
*
*	\param X1 ù��° GTransf
*	\param X2 �ι�° GTransf
*
*	\return \a X1 ���� \a X2 ��ȯ������ �Ÿ�
*/
inline
double dist(const GTransf &X1, const GTransf &X2)
{
	return norm(inv(X1) * X2);
}

/*!
*	\brief ��ȯ���� �ð��� ���� ���������Ѵ�.
*
*	\param X ������ ��ȯ�� ����� ����
*	\param X1 ù��° ��ȯ
*	\param X2 �ι�° ��ȯ
*	\param t �ð� (0 <= t <= 1)
*
*	\return ������ ��ȯ�� ��ȯ�ȴ�.
*/
inline
void lerp(GTransf &X, const GTransf &X1, const GTransf &X2, const double t)
{
	GTransf XX1(X1), XX2(X2);

	if (quat_dot(XX1.q, XX2.q) < 0.0)
		XX2.q.SetNegate();

	X.t = (1.0 - t) * XX1.t + t * XX2.t;
	X.q = slerp(XX1.q, XX2.q, t);
}

/*!
*	\brief \a X ��ǥ�迡�� ǥ���� ��ȯ \a dX�� \a Y ��ǥ�迡�� ǥ���� ��ȯ���� ����Ѵ�.
*
*	\param X ������ǥ��
*	\param dX ������ǥ�� \a X���� ǥ���� ��ȯ
*	\param Y �����ǥ��
*
*	\return ������ǥ�� \a X���� ǥ���� ��ȯ \a dX�� �����ǥ�� \a Y���� ǥ������ ��ȯ
*/
inline
GTransf ad(const GTransf &X, const GTransf &dX, const GTransf &Y)
{
	GTransf YX = inv(Y) * X;
	GVector3 p = part_trans(YX * dX) - part_trans(YX);

	GVector3 axis;
	double angle;
	part_rot(dX).GetAngleAxis(axis, angle, true);

	GQuater q;
	q.SetFromAngleAxis(angle, YX * axis, true);

	return GTransf(p, q);
}

/*!
*	\class RBFinterp
*	\brief Radial basis�� ����� �����Լ��� ǥ���ϴ� Ŭ����
*
*	\author ������ (shyun@dongguk.edu) 
*	\date 03 Dec. 2008	
*/
enum RBFType
{ 
	RBF_GAUSSIAN = 0, 
	RBF_BIHARMONIC, 
	RBF_TRIHARMONIC, 
	RBF_MULTIQUADRIC, 
	RBF_THINPLATE,
};

class RBFinterp
{
public:
	// ������ �� �Ҹ���
	RBFinterp(const int &_m = 0, const int &_n = 0, RBFType _type = RBF_GAUSSIAN);
	RBFinterp(const RBFinterp &cpy);
	virtual ~RBFinterp();

	// ���Կ�����
	RBFinterp &operator =(const RBFinterp &rhs);
	
	// ����Լ�
	int GetDimParam() const;
	int GetDimValue() const;
	RBFType GetBasisType() const;
	GMatrix GetCoeffMatrix() const;
	vector<GVector> &GetParamList();
	vector<GVector> &GetValueList();
	void SetDimParam(const int &_m);
	void SetDimValue(const int &_n);
	void SetBasisType(RBFType _type);
	void AddData(const GVector &Param, const GVector &Value);
	void Interpolate();
	GVector Eval(double x0, ...);
	GVector Eval(const GVector &X);

protected:
	// ���λ��
	double basis(double r2);

protected:
	// ������ ���
	int m;					// �Ķ���� ������ ����
	int n;					// ������ ������ ����
	RBFType type;			// Radial basis �Լ��� Ÿ��
	GMatrix C;				// �������� ����� ����� ���
	vector<GVector> Params;	// �����Ķ������ m-���� ����
	vector<GVector> Values;	// ������������ n-���� ����
};

/*!
*	\brief ����Ʈ ������
*
*	\param _m �Ķ���Ͱ����� ����
*	\param _n ������ ������ ����
*	\param _type Radial basis�� Ÿ��
*/
inline
RBFinterp::RBFinterp(const int &_m, const int &_n, RBFType _type)
{
	m = _m;
	n = _n;
	type = _type;
}

/*!
*	\brief ���� ������
*
*	\param copy ����� ��ü
*/
inline
RBFinterp::RBFinterp(const RBFinterp &cpy)
: Params(cpy.Params), Values(cpy.Values)
{
	m = cpy.m;
	n = cpy.n;
	type = cpy.type;
	C = cpy.C;
}

/*!
*	\brief �Ҹ���
*/
inline
RBFinterp::~RBFinterp()
{
}

/*!
*	\brief ���Կ�����
*
*	\param rhs ���Ե� ��ü
*
*	\return ���Ե� �ڽ��� ��ȯ
*/
inline
RBFinterp &RBFinterp::operator =(const RBFinterp &rhs)
{
	m = rhs.m;
	n = rhs.n;
	type = rhs.type;
	Params = rhs.Params;
	Values = rhs.Values;
	C = rhs.C;

	return *this;
}

/*!
*	\brief �Ķ���� ������ ������ ��ȯ�Ѵ�.
*
*	\return �Ķ���� ������ ����
*/
inline
int RBFinterp::GetDimParam() const
{
	return m;
}

/*!
*	\brief ������ ������ ������ ��ȯ�Ѵ�.
*
*	\return ������ ������ ����
*/
inline
int RBFinterp::GetDimValue() const
{
	return n;
}

/*!
*	\brief Radial basis�� Ÿ���� ��ȯ�Ѵ�.
*
*	\return Radial basis�� Ÿ��
*/
inline
RBFType RBFinterp::GetBasisType() const
{
	return type;
}

/*!
*	\brief �����Լ��� �������� ��ȯ�Ѵ�.
*
*	\return �����Լ��� ������
*/
inline
GMatrix RBFinterp::GetCoeffMatrix() const
{
	return C;
}

/*!
*	\brief ���� �Ķ���� ����Ʈ�� ��ȯ�Ѵ�.
*
*	\return ���� �Ķ���� ����Ʈ
*/
inline
vector<GVector> &RBFinterp::GetParamList()
{
	return Params;
}

/*!
*	\brief ���� ������ ����Ʈ�� ��ȯ�Ѵ�.
*
*	\return ���� ������ ����Ʈ
*/
inline
vector<GVector> &RBFinterp::GetValueList()
{
	return Values;
}

/*!
*	\brief �Ķ���Ͱ����� ������ �����Ѵ�.
*
*	\param _m �Ķ���Ͱ����� ����
*/
inline
void RBFinterp::SetDimParam(const int &_m)
{
	m = _m;
}

/*!
*	\brief �����Ͱ����� ������ �����Ѵ�.
*
*	\param _n �����Ͱ����� ����
*/
inline
void RBFinterp::SetDimValue(const int &_n)
{
	n = _n;
}

/*!
*	\brief Basis Ÿ���� �����Ѵ�.
*
*	\param _type Basis Ÿ��
*/
inline
void RBFinterp::SetBasisType(RBFType _type)
{
	type = _type;
}

/*!
*	\brief ������ �Ķ���Ϳ� ���� �߰��Ѵ�.
*
*	\param Param �����Ķ����
*	\param Value \a Param�� �����ϴ� ������ ��
*/
inline
void RBFinterp::AddData(const GVector &Param, const GVector &Value)
{
	assert(m == Param.N && n == Value.N);
	vector<GVector>::iterator it1, it2;

	it1 = Params.begin();
	it2 = Values.begin();
	while (it1 != Params.end())
	{
		if (Param == *it1)
		{
			it1 = Params.erase(it1);
			it2 = Values.erase(it2);
			continue;
		}

		it1++;
		it2++;
	}
	Params.push_back(Param);
	Values.push_back(Value);
}

/*!
*	\brief �־��� �����͸� �����ϴ� �Լ��� �����Ѵ�.
*/
inline
void RBFinterp::Interpolate()
{
	int l = (int)Values.size();
	GMatrix A(l + m + 1, l + m + 1);
	GMatrix B(l + m + 1, n);
	C = GMatrix(n, l + m + 1);
	
	// A ����� ���� ����
	for (int i = 0; i < l; ++i)
	{
		for (int j = i; j < l; ++j)
		{
			double r2 = (Params[i] - Params[j]) * (Params[i] - Params[j]);
			A[i][j] = A[j][i] = basis(r2);
		}

		A[i][l] = A[l][i] = 1.0;
		for (int j = 0; j < m; ++j)
			A[i][l + j + 1] = A[l + j + 1][i] = Params[i][j];
	}

	// B ����� ���� ����
	for (int i = 0; i < l; ++i)
		for (int j = 0; j < n; ++j)
			B[i][j] = Values[i][j];

	if (l >= m)
	{
		// A ����� LU�� �����Ѵ�.
		GVector Idx(l + m + 1);
		double d;
		dcmp_lu(A, Idx, d);

		// ������������ Ǭ��.
		GVector x(l + m + 1), b(l + m + 1);
		for (int i = 0; i < n; ++i)
		{
			b = B.GetColVec(i);
			lin_sys_lu(A, x, b, Idx);
			C.SetRowVec(i, x);
		}
	}
	else
	{
		// A ����� USV�� �����Ѵ�.
		GMatrix U(l + m + 1, l + m + 1), V(l + m + 1, l + m + 1);
		GVector S(l + m + 1);
		dcmp_sv(A, U, S, V);

		// ������������ Ǭ��.
		GVector x(l + m + 1), b(l + m + 1);
		for (int i = 0; i < n; ++i)
		{
			b = B.GetColVec(i);
			lin_sys_sv(U, S, V, x, b);
			C.SetRowVec(i, x);
		}
	}
}

/*!
*	\brief Radial basis �Լ��� ���� ��ȯ�Ѵ�.
*
*	\param r2 �������κ����� �Ÿ��� ����
*/
inline
double RBFinterp::basis(double r2)
{
	double ret = 0.0;
	switch(type)
	{
	case RBF_GAUSSIAN:
		ret = exp(-0.03 * r2);	// exp(-beta * r^2)
		break;
	case RBF_MULTIQUADRIC:
		ret = sqrt(r2 + 10);	// sqrt(r^2 + beta^2)
		break;
	case RBF_BIHARMONIC:
		ret = sqrt(r2);
		break;
	case RBF_TRIHARMONIC:
		ret = r2 * sqrt(r2);
		break;
	case RBF_THINPLATE:
		ret = r2 * log(sqrt(r2));
		break;
	}

	return ret;
}

/*!
*	\brief ������ �Ķ���� ������ �Լ����� ����Ѵ�.
*
*	\param x0, ... �Լ����� ���� �Ķ���� ��
*
*	\return ���� �Լ���
*/
inline
GVector RBFinterp::Eval(double x0, ...)
{
	static double tmp[100];		// �ִ� 100���� ���� ����....

	tmp[0] = x0;
	va_list ap;
	va_start(ap, x0);
	for (int i = 1; i != m; ++i)
		tmp[i] = va_arg(ap, double);
	va_end(ap);
	
	int l = (int)Params.size();
	GVector X(m), PI(l + m + 1);
	X.Set(tmp);

	for (int i = 0; i < l; ++i)
	{
		double r2 = (X - Params[i]) * (X - Params[i]);
		PI.V[i] = basis(r2);
	}

	PI.V[l] = 1.0;
	for (int i = 0; i < m; ++i)
		PI.V[l + i + 1] = X.V[i];
	
	return C * PI;
}


/*!
*	\brief ������ �Ķ���� ������ �Լ����� ����Ѵ�.
*
*	\param X �Լ����� ���� �Ķ���� ��
*	\return ���� �Լ���
*/
inline
GVector RBFinterp::Eval(const GVector &X)
{
	assert(X.N == m);
	int l = (int)Params.size();
	GVector PI(l + m + 1);
	
	for (int i = 0; i < l; ++i)
	{
		double r2 = (X - Params[i]) * (X - Params[i]);
		PI.V[i] = basis(r2);
	}

	PI.V[l] = 1.0;
	for (int i = 0; i < m; ++i)
		PI.V[l + i + 1] = X.V[i];

	return C * PI;
}

#endif // _GMATH_H