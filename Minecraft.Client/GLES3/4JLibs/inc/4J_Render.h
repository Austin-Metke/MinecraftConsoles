#pragma once

// OpenGL ES 3.0 render backend for C4JRender.
// This header is used when building with __GLES3__ defined.

#include <GLES3/gl3.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shared helper types (mirrors what other platform headers provide)
// ---------------------------------------------------------------------------

class ImageFileBuffer
{
public:
	enum EImageType
	{
		e_typePNG,
		e_typeJPG
	};

	EImageType m_type;
	void*      m_pBuffer;
	int        m_bufferSize;

	int   GetType()           { return m_type; }
	void* GetBufferPointer()  { return m_pBuffer; }
	int   GetBufferSize()     { return m_bufferSize; }
	void  Release()           { free(m_pBuffer); m_pBuffer = NULL; }
	bool  Allocated()         { return m_pBuffer != NULL; }
};

typedef struct
{
	int Width;
	int Height;
} D3DXIMAGE_INFO;

typedef struct _XSOCIAL_PREVIEWIMAGE
{
	uint8_t* pBytes;
	uint32_t Pitch;
	uint32_t Width;
	uint32_t Height;
} XSOCIAL_PREVIEWIMAGE, *PXSOCIAL_PREVIEWIMAGE;

// ---------------------------------------------------------------------------
// C4JRender – OpenGL ES 3.0 implementation
// ---------------------------------------------------------------------------

class C4JRender
{
public:
	// -----------------------------------------------------------------------
	// Lifecycle
	// -----------------------------------------------------------------------
	void Tick();
	void UpdateGamma(unsigned short usGamma);

	// -----------------------------------------------------------------------
	// Matrix stack
	// -----------------------------------------------------------------------
	void         MatrixMode(int type);
	void         MatrixSetIdentity();
	void         MatrixTranslate(float x, float y, float z);
	void         MatrixRotate(float angle, float x, float y, float z);
	void         MatrixScale(float x, float y, float z);
	void         MatrixPerspective(float fovy, float aspect, float zNear, float zFar);
	void         MatrixOrthogonal(float left, float right, float bottom, float top, float zNear, float zFar);
	void         MatrixPop();
	void         MatrixPush();
	void         MatrixMult(float* mat);
	const float* MatrixGet(int type);
	void         Set_matrixDirty();

	// -----------------------------------------------------------------------
	// Core
	// -----------------------------------------------------------------------
	void Initialise();
	void InitialiseContext();
	void StartFrame(bool actualFrameStart = true);
	void Present();
	void Clear(int flags);
	void SetClearColour(const float colourRGBA[4]);
	bool IsWidescreen();
	bool IsHiDef();
	void InternalScreenCapture();
	void CaptureThumbnail(ImageFileBuffer* pngOut, ImageFileBuffer* saveGamePngOut);
	void CaptureScreen(ImageFileBuffer* jpgOut, XSOCIAL_PREVIEWIMAGE* previewOut);
	void BeginConditionalSurvey(int identifier);
	void EndConditionalSurvey();
	void BeginConditionalRendering(int identifier);
	void EndConditionalRendering();

	// -----------------------------------------------------------------------
	// Vertex types
	// -----------------------------------------------------------------------
	typedef enum
	{
		VERTEX_TYPE_PF3_TF2_CB4_NB4_XW1,          // 32 bytes: pos3f, uv2f, col4b, nrm4b, pad4b
		VERTEX_TYPE_COMPRESSED,                     // 16 bytes: pos3s, col1s, uv2s, lightUV2s
		VERTEX_TYPE_PF3_TF2_CB4_NB4_XW1_LIT,       // Same layout as PF3… but lighting pre-applied
		VERTEX_TYPE_PF3_TF2_CB4_NB4_XW1_TEXGEN,    // Same layout as PF3… but uses tex-gen UVs
		VERTEX_TYPE_COUNT
	} eVertexType;

	// -----------------------------------------------------------------------
	// Pixel shader types
	// -----------------------------------------------------------------------
	typedef enum
	{
		PIXEL_SHADER_TYPE_STANDARD,
		PIXEL_SHADER_TYPE_PROJECTION,
		PIXEL_SHADER_TYPE_FORCELOD,
		PIXEL_SHADER_COUNT
	} ePixelShaderType;

	// -----------------------------------------------------------------------
	// Viewport types
	// -----------------------------------------------------------------------
	typedef enum
	{
		VIEWPORT_TYPE_FULLSCREEN,
		VIEWPORT_TYPE_SPLIT_TOP,
		VIEWPORT_TYPE_SPLIT_BOTTOM,
		VIEWPORT_TYPE_SPLIT_LEFT,
		VIEWPORT_TYPE_SPLIT_RIGHT,
		VIEWPORT_TYPE_QUADRANT_TOP_LEFT,
		VIEWPORT_TYPE_QUADRANT_TOP_RIGHT,
		VIEWPORT_TYPE_QUADRANT_BOTTOM_LEFT,
		VIEWPORT_TYPE_QUADRANT_BOTTOM_RIGHT,
	} eViewportType;

	// -----------------------------------------------------------------------
	// Primitive types
	// -----------------------------------------------------------------------
	typedef enum
	{
		PRIMITIVE_TYPE_TRIANGLE_LIST,
		PRIMITIVE_TYPE_TRIANGLE_STRIP,
		PRIMITIVE_TYPE_TRIANGLE_FAN,
		PRIMITIVE_TYPE_QUAD_LIST,
		PRIMITIVE_TYPE_LINE_LIST,
		PRIMITIVE_TYPE_LINE_STRIP,
		PRIMITIVE_TYPE_COUNT
	} ePrimitiveType;

	// -----------------------------------------------------------------------
	// Drawing
	// -----------------------------------------------------------------------
	void DrawVertices(ePrimitiveType primitiveType, int count, void* dataIn,
	                  eVertexType vType, ePixelShaderType psType);

	// -----------------------------------------------------------------------
	// Command buffers (display-list equivalent)
	// -----------------------------------------------------------------------
	void CBuffLockStaticCreations();
	int  CBuffCreate(int count);
	void CBuffDelete(int first, int count);
	void CBuffStart(int index, bool full = false);
	void CBuffClear(int index);
	int  CBuffSize(int index);
	void CBuffEnd();
	bool CBuffCall(int index, bool full = true);
	void CBuffTick();
	void CBuffDeferredModeStart();
	void CBuffDeferredModeEnd();

	// -----------------------------------------------------------------------
	// Texture formats
	// -----------------------------------------------------------------------
	typedef enum
	{
		TEXTURE_FORMAT_RxGyBzAw,        // RGBA8
		TEXTURE_FORMAT_RxGyBzAw5551,    // RGBA5551
		MAX_TEXTURE_FORMATS
	} eTextureFormat;

	// -----------------------------------------------------------------------
	// Texture management
	// -----------------------------------------------------------------------
	int    TextureCreate();
	void   TextureFree(int idx);
	void   TextureBind(int idx);
	void   TextureBindVertex(int idx);
	void   TextureSetTextureLevels(int levels);
	int    TextureGetTextureLevels();
	void   TextureData(int width, int height, void* data, int level,
	                   eTextureFormat format = TEXTURE_FORMAT_RxGyBzAw);
	void   TextureDataUpdate(int xoffset, int yoffset, int width, int height,
	                         void* data, int level);
	void   TextureSetParam(int param, int value);
	void   TextureDynamicUpdateStart();
	void   TextureDynamicUpdateEnd();
	int    LoadTextureData(const char* szFilename, D3DXIMAGE_INFO* pSrcInfo, int** ppDataOut);
	int    LoadTextureData(uint8_t* pbData, uint32_t dwBytes, D3DXIMAGE_INFO* pSrcInfo, int** ppDataOut);
	int    SaveTextureData(const char* szFilename, D3DXIMAGE_INFO* pSrcInfo, int* ppDataOut);
	void   TextureGetStats();
	GLuint TextureGetTexture(int idx);

	// -----------------------------------------------------------------------
	// State control
	// -----------------------------------------------------------------------
	void StateSetColour(float r, float g, float b, float a);
	void StateSetDepthMask(bool enable);
	void StateSetBlendEnable(bool enable);
	void StateSetBlendFunc(int src, int dst);
	void StateSetBlendFactor(unsigned int colour);
	void StateSetAlphaFunc(int func, float param);
	void StateSetDepthFunc(int func);
	void StateSetFaceCull(bool enable);
	void StateSetFaceCullCW(bool enable);
	void StateSetLineWidth(float width);
	void StateSetWriteEnable(bool red, bool green, bool blue, bool alpha);
	void StateSetDepthTestEnable(bool enable);
	void StateSetAlphaTestEnable(bool enable);
	void StateSetDepthSlopeAndBias(float slope, float bias);
	void StateSetFogEnable(bool enable);
	void StateSetFogMode(int mode);
	void StateSetFogNearDistance(float dist);
	void StateSetFogFarDistance(float dist);
	void StateSetFogDensity(float density);
	void StateSetFogColour(float red, float green, float blue);
	void StateSetLightingEnable(bool enable);
	void StateSetVertexTextureUV(float u, float v);
	void StateSetLightColour(int light, float red, float green, float blue);
	void StateSetLightAmbientColour(float red, float green, float blue);
	void StateSetLightDirection(int light, float x, float y, float z);
	void StateSetLightEnable(int light, bool enable);
	void StateSetViewport(eViewportType viewportType);
	void StateSetEnableViewportClipPlanes(bool enable);
	void StateSetTexGenCol(int col, float x, float y, float z, float w, bool eyeSpace);
	void StateSetStencil(int Function, uint8_t stencilRef,
	                     uint8_t stencilFuncMask, uint8_t stencilWriteMask);
	void StateSetForceLOD(int LOD);

	// -----------------------------------------------------------------------
	// Event tracking (profiling markers – no-ops on GLES3)
	// -----------------------------------------------------------------------
	void BeginEvent(const wchar_t* eventName);
	void EndEvent();

	// -----------------------------------------------------------------------
	// Memory helpers (thin wrappers over malloc/free on GLES3)
	// -----------------------------------------------------------------------
	void* MemoryAllocateGPUMem(uint32_t alignment, uint32_t size);
	void* MemoryAllocateCPUMem(uint32_t alignment, uint32_t size);
	void  MemoryFreeGPUMem(void* data);
	void  MemoryFreeCPUMem(void* data);
};

// ---------------------------------------------------------------------------
// Matrix-mode constants (same values as classic OpenGL)
// ---------------------------------------------------------------------------
const int GL_MODELVIEW_MATRIX  = 0;
const int GL_PROJECTION_MATRIX = 1;
const int GL_MODELVIEW         = 0;
const int GL_PROJECTION        = 1;
const int GL_TEXTURE           = 2;

// TexGen constants
const int GL_S = 0;
const int GL_T = 1;
const int GL_R = 2;
const int GL_Q = 3;

const int GL_TEXTURE_GEN_S    = 0;
const int GL_TEXTURE_GEN_T    = 1;
const int GL_TEXTURE_GEN_Q    = 2;
const int GL_TEXTURE_GEN_R    = 3;
const int GL_TEXTURE_GEN_MODE = 0;
const int GL_OBJECT_LINEAR    = 0;
const int GL_EYE_LINEAR       = 1;
const int GL_OBJECT_PLANE     = 0;
const int GL_EYE_PLANE        = 1;

// glEnable/glDisable tokens (must be non-zero and unique)
const int GL_TEXTURE_2D  = 1;
const int GL_BLEND       = 2;
const int GL_CULL_FACE   = 3;
const int GL_ALPHA_TEST  = 4;
const int GL_DEPTH_TEST  = 5;
const int GL_FOG         = 6;
const int GL_LIGHTING    = 7;
const int GL_LIGHT0      = 8;
const int GL_LIGHT1      = 9;

// Clear flags
const int CLEAR_DEPTH_FLAG  = 1;
const int CLEAR_COLOUR_FLAG = 2;
const int GL_DEPTH_BUFFER_BIT = CLEAR_DEPTH_FLAG;
const int GL_COLOR_BUFFER_BIT = CLEAR_COLOUR_FLAG;

// Blend factors – native GLES3 enum values
const int GL_SRC_ALPHA                = 0x0302;
const int GL_ONE_MINUS_SRC_ALPHA      = 0x0303;
const int GL_ONE                      = 1;
const int GL_ZERO                     = 0;
const int GL_DST_ALPHA                = 0x0304;
const int GL_SRC_COLOR                = 0x0300;
const int GL_DST_COLOR                = 0x0306;
const int GL_ONE_MINUS_DST_COLOR      = 0x0307;
const int GL_ONE_MINUS_SRC_COLOR      = 0x0301;
const int GL_CONSTANT_ALPHA           = 0x8003;
const int GL_ONE_MINUS_CONSTANT_ALPHA = 0x8004;

// Depth/alpha compare functions – native GLES3 enum values
const int GL_GREATER = 0x0204;
const int GL_EQUAL   = 0x0202;
const int GL_LEQUAL  = 0x0203;
const int GL_GEQUAL  = 0x0206;
const int GL_ALWAYS  = 0x0207;

// Texture parameter tokens
const int GL_TEXTURE_MIN_FILTER     = 0x2801;
const int GL_TEXTURE_MAG_FILTER     = 0x2800;
const int GL_TEXTURE_WRAP_S         = 0x2802;
const int GL_TEXTURE_WRAP_T         = 0x2803;
const int GL_NEAREST                = 0x2600;
const int GL_LINEAR                 = 0x2601;
const int GL_EXP                    = 2;
const int GL_NEAREST_MIPMAP_LINEAR  = 0x2702;
const int GL_CLAMP                  = 0x2900;
const int GL_REPEAT                 = 0x2901;

// Fog parameter tokens
const int GL_FOG_START   = 1;
const int GL_FOG_END     = 2;
const int GL_FOG_MODE    = 3;
const int GL_FOG_DENSITY = 4;
const int GL_FOG_COLOR   = 5;

// Light parameter tokens
const int GL_POSITION       = 1;
const int GL_AMBIENT        = 2;
const int GL_DIFFUSE        = 3;
const int GL_SPECULAR       = 4;
const int GL_LIGHT_MODEL_AMBIENT = 1;

// Primitive type aliases (mapped to C4JRender enum values)
const int GL_LINES          = C4JRender::PRIMITIVE_TYPE_LINE_LIST;
const int GL_LINE_STRIP     = C4JRender::PRIMITIVE_TYPE_LINE_STRIP;
const int GL_QUADS          = C4JRender::PRIMITIVE_TYPE_QUAD_LIST;
const int GL_TRIANGLE_FAN   = C4JRender::PRIMITIVE_TYPE_TRIANGLE_FAN;
const int GL_TRIANGLE_STRIP = C4JRender::PRIMITIVE_TYPE_TRIANGLE_STRIP;

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------
extern C4JRender RenderManager;
