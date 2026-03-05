// GLES3_Types.h – portable type shims for the OpenGL ES 3.0 platform build.
//
// On Windows these are provided by <windows.h>; on Linux/Android we define
// them here so the rest of the codebase compiles unchanged under __GLES3__.

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <wchar.h>

// ---------------------------------------------------------------------------
// Basic Windows-style typedefs
// ---------------------------------------------------------------------------

typedef unsigned char      BYTE;
typedef unsigned short     WORD;
typedef unsigned int       DWORD;
typedef unsigned long long QWORD;
typedef int                BOOL;
typedef long               LONG;
typedef unsigned long      ULONG;
typedef void*              HANDLE;
typedef wchar_t            WCHAR;
typedef const wchar_t*     LPCWSTR;
typedef wchar_t*           LPWSTR;
typedef long               HRESULT;

#ifndef TRUE
#  define TRUE  1
#endif
#ifndef FALSE
#  define FALSE 0
#endif

// ---------------------------------------------------------------------------
// HRESULT helpers
// ---------------------------------------------------------------------------

#ifndef S_OK
#  define S_OK    ((HRESULT)0L)
#endif
#ifndef E_FAIL
#  define E_FAIL  ((HRESULT)0x80004005L)
#endif

#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)
#define FAILED(hr)    (((HRESULT)(hr)) < 0)

// ---------------------------------------------------------------------------
// Standard string type used throughout the game
// ---------------------------------------------------------------------------

#include <string>
typedef std::wstring wstring;
typedef std::string  string;

// ---------------------------------------------------------------------------
// PI constant (used by glWrapper.cpp, etc.)
// ---------------------------------------------------------------------------

#ifndef PI
#  define PI 3.14159265358979323846f
#endif

// ---------------------------------------------------------------------------
// NULL / nullptr safety
// ---------------------------------------------------------------------------

#ifndef NULL
#  define NULL nullptr
#endif

// ---------------------------------------------------------------------------
// byte typedef used in Tesselator and other code
// ---------------------------------------------------------------------------

typedef unsigned char byte;
