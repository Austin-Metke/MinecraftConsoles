// GLES3_Maths.h – lightweight maths helpers for the OpenGL ES 3.0 platform
// build.
//
// Other platform builds use DirectXMath or their SDK equivalents.  For
// the GLES3 target we provide a minimal set of the constructs that the
// common game code actually needs.

#pragma once

#include <cmath>

// ---------------------------------------------------------------------------
// AUTO_VAR – already defined in stdafx.h for __GLES3__
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// XMFLOAT3 / XMFLOAT4 minimal stand-ins
// (Only the fields are needed; no operator overloads required.)
// ---------------------------------------------------------------------------

struct XMFLOAT2
{
	float x, y;
	XMFLOAT2()             : x(0), y(0) {}
	XMFLOAT2(float x, float y) : x(x), y(y) {}
};

struct XMFLOAT3
{
	float x, y, z;
	XMFLOAT3()                       : x(0), y(0), z(0) {}
	XMFLOAT3(float x, float y, float z) : x(x), y(y), z(z) {}
};

struct XMFLOAT4
{
	float x, y, z, w;
	XMFLOAT4()                               : x(0), y(0), z(0), w(0) {}
	XMFLOAT4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
};

// ---------------------------------------------------------------------------
// Radian / degree helpers
// ---------------------------------------------------------------------------

inline float DegToRad(float deg) { return deg * (3.14159265358979323846f / 180.0f); }
inline float RadToDeg(float rad) { return rad * (180.0f / 3.14159265358979323846f); }
