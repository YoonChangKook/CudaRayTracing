#ifndef _COLOR_H_
#define _COLOR_H_

class Color
{
public:
	// Constructors
	__host__ __device__ Color();
	__host__ __device__ Color(unsigned char r, unsigned char g, unsigned char b);
	__host__ __device__ Color(const unsigned char color[3]);
	__host__ __device__ Color(const Color& cpy);
	// Destructors
	__host__ __device__ ~Color();

private:
	// Members
	unsigned char color[3];	// r, g, b

public:
	// Methods
	__host__ __device__ unsigned char& operator [](const int& idx);
	__host__ __device__ const unsigned char& operator [](const int& idx) const;
	__host__ __device__ Color operator +(const Color& other) const;
	__host__ __device__ Color operator -(const Color& other) const;
	__host__ __device__ Color operator *(const double& v) const;
	__host__ __device__ Color& operator =(const Color& other);
	__host__ __device__ Color& operator +=(const Color& other);
	__host__ __device__ Color& operator -=(const Color& other);
	__host__ __device__ Color& operator *=(const double& v);
};

Color::Color()
	: color{ 0, 0, 0 }
{}
Color::Color(unsigned char r, unsigned char g, unsigned char b)
	: color{ r, g, b }
{}
Color::Color(const unsigned char color[3])
	: color{ color[0], color[1], color[2] }
{}
Color::Color(const Color& cpy)
	: color{ cpy[0], cpy[1], cpy[2] }
{}

Color::~Color()
{}

unsigned char& Color::operator [](const int& idx)
{
	return this->color[idx];
}

const unsigned char& Color::operator [](const int& idx) const
{
	return this->color[idx];
}

Color Color::operator +(const Color& other) const
{
	Color temp(this->color);
	for (int i = 0; i < 3; i++)
	{
		if (temp[i] + other[i] < 255)
			temp[i] += other[i];
		else
			temp[i] = (unsigned char)255;
	}

	return temp;
}

Color Color::operator -(const Color& other) const
{
	Color temp(this->color);
	for (int i = 0; i < 3; i++)
	{
		if (temp[i] - other[i] > 0)
			temp[i] -= other[i];
		else
			temp[i] = (unsigned char)0;
	}

	return temp;
}

Color Color::operator *(const double& v) const
{
	Color temp(this->color);
	for (int i = 0; i < 3; i++)
	{
		if (temp[i] * v < 255)
			temp[i] *= v;
		else
			temp[i] = (unsigned char)255;
	}

	return temp;
}

Color& Color::operator=(const Color& other)
{
	for (int i = 0; i < 3; i++)
		this->color[i] = other[i];

	return *this;
}

Color& Color::operator +=(const Color& other)
{
	for (int i = 0; i < 3; i++)
	{
		if (this->color[i] + other[i] < 255)
			this->color[i] += other[i];
		else
			this->color[i] = (unsigned char)255;
	}
	return *this;
}

Color& Color::operator -=(const Color& other)
{
	for (int i = 0; i < 3; i++)
	{
		if (this->color[i] - other[i] > 0)
			this->color[i] -= other[i];
		else
			this->color[i] = (unsigned char)0;
	}
	return *this;
}

Color& Color::operator *=(const double& v)
{
	for (int i = 0; i < 3; i++)
	{
		if (this->color[i] * v < 255)
			this->color[i] *= v;
		else
			this->color[i] = (unsigned char)255;
	}
	return *this;
}

#endif