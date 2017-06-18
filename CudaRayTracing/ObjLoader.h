#pragma once

#ifndef _OBJ_LOADER_H_
#define _OBJ_LOADER_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
#include <math.h>
#include "KMath.h"
#include "StringTokenizer.h"

using namespace std;

enum OBJ_OPTION
{
	V,
	VT,
	VN,
	F,
};

class CStringTokenizer;
class ObjData
{
public:
	ObjData() : is_loaded(false)
	{}
	ObjData(const char* pFileName) { Load(pFileName); }

public:
	bool IsLoaded() const;
	bool Load(const char* pFileName);
	vector<KPoint3> m_vertices;
	vector<KPoint3> m_vertices_texture;
	vector<KVector3> m_normals;
	vector<int> m_faces;

private:
	bool is_loaded;
	void ProcessData(OBJ_OPTION option, string& line);
};

const string split_sign = " ";
const string split_sign1 = "/";
const string split_sign2 = "//";

void DeleteOption(string& line) { line = &line[2]; }

template <typename T>
T ProcessData_3D(CStringTokenizer& token)
{
	float v[3] = { 0, };
	for (int i = 0; i<3; ++i)
		v[i] = atof(token.nextToken().c_str());

	return T(v[0], v[1], v[2]);
}

/*
*	v//vn or
*	v1/vt/vn
*/
void ProcessFace(CStringTokenizer& token, int out_faces[3]);

bool ObjData::IsLoaded() const
{
	return this->is_loaded;
}

/*
*	line[0] == 'v' && line[1] == 't'
*	this code is better than the code line.substr(0, 2) == "v "
*	because of there is no line or just one character,
*	the second way is throwing the error.
*	but the first way will check left, then right.
*	so, fine if there is not exist line[1].
*/
bool ObjData::Load(const char* pFileName)
{
	ifstream readStream(pFileName, std::ios::in);
	if (readStream.is_open() == false)
		return false;

	readStream.seekg(0, std::ios::end);
	streamoff filesize = readStream.tellg();
	readStream.seekg(0, std::ios::beg);
	streamoff readData = 0;

	string line;
	while (getline(readStream, line))
	{
		if (line.empty())
			continue;

		if (line[0] == 'v' && line[1] == ' ')
			ProcessData(OBJ_OPTION::V, line);
		if (line[0] == 'v' && line[1] == 't')
			ProcessData(OBJ_OPTION::VT, line);
		if (line[0] == 'v' && line[1] == 'n')
			ProcessData(OBJ_OPTION::VN, line);
		if (line[0] == 'f' && line[1] == ' ')
			ProcessData(OBJ_OPTION::F, line);

		readData = readStream.tellg();
		if (readData > 0)
		{
			printf("\r");
			cout << "per : " << ((float)readData / (float)filesize) * 100.f;
		}
	}

	readStream.close();
	is_loaded = true;
	return true;
}

void ObjData::ProcessData(OBJ_OPTION option, string& line)
{
	DeleteOption(line);
	CStringTokenizer token(line, split_sign);

	switch (option)
	{
	case OBJ_OPTION::V:
		m_vertices.push_back(ProcessData_3D<KPoint3>(token));
		break;
	case OBJ_OPTION::VT:
		m_vertices_texture.push_back(ProcessData_3D<KPoint3>(token));
		break;
	case OBJ_OPTION::VN:
		m_normals.push_back(ProcessData_3D<KVector3>(token));
		break;
	case OBJ_OPTION::F:
		int temp[3];
		ProcessFace(token, temp);
		for (int i = 0; i < 3; i++)
			m_faces.push_back(temp[i]);
		break;
	}
}

void ProcessFace(CStringTokenizer& token, int out_faces[3])
{
	int vertex_idx[3] = { 0, };
	int texcrd_idx[3] = { 0, };
	int normal_idx[3] = { 0, };
	string s[3];

	for (int i = 0; i<3; ++i)
		s[i] = token.nextToken();

	CStringTokenizer t[3];
	for (int i = 0; i<3; ++i)
		t[i].Init(s[i], split_sign1);

	for (int i = 0; i<3; ++i)
	{
		if (s[i].find(split_sign2, 0) == string::npos)
		{
			vertex_idx[i] = atoi(t[i].nextToken().c_str()) - 1;
			texcrd_idx[i] = atoi(t[i].nextToken().c_str()) - 1;
			normal_idx[i] = atoi(t[i].nextToken().c_str()) - 1;
		}
		else // If the string seperated by "//".
		{
			vertex_idx[i] = atoi(t[i].nextToken().c_str()) - 1;
			normal_idx[i] = atoi(t[i].nextToken().c_str()) - 1;
		}
	}

	for (int i = 0; i < 3; i++)
		out_faces[i] = vertex_idx[i];
}

#endif