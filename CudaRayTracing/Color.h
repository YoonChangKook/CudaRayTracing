#pragma once

#ifndef _COLOR_H_
#define _COLOR_H_

#include <cassert>

class Color
{
public:
	// Constructors
	Color();
	Color(unsigned char r, unsigned char g, unsigned char b);
	Color(const unsigned char color[3]);
	Color(const Color& cpy);
	// Destructors
	virtual ~Color();

private:
	// Members
	unsigned char color[3];	// r, g, b

public:
	// Methods
	unsigned char& operator [](const int& idx);
	const unsigned char& operator [](const int& idx) const;
	Color operator +(const Color& other) const;
	Color operator -(const Color& other) const;
	Color operator *(const double& v) const;
	Color& operator =(const Color& other);
	Color& operator +=(const Color& other);
	Color& operator -=(const Color& other);
	Color& operator *=(const double& v);
};

#endif