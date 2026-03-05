// GLES3_Render.cpp – C4JRender implementation using OpenGL ES 3.0 directly.
//
// Vertex format reference (VERTEX_TYPE_PF3_TF2_CB4_NB4_XW1, 32 bytes):
//   float[3]  position   (x, y, z)
//   float[2]  texcoord   (u, v)  – u > 1 signals "no mipmap"
//   uint8[4]  colour     (r, g, b, a)  packed big-endian: r<<24|g<<16|b<<8|a
//   uint8[4]  normal     (nx, ny, nz, nw)
//   int16[2]  lightUV    (u2, v2)  – 0xFE00/0xFE00 = "use global"
//
// Vertex format reference (VERTEX_TYPE_COMPRESSED, 16 bytes):
//   int16[3]  position   x=raw/1024, y=raw/1024, z=raw/1024
//   int16     colour     RGB565 stored as (value - 0x8000) signed
//   int16[2]  texcoord   u=raw/8192, v=raw/8192
//   int16[2]  lightUV    u2, v2  (sentinel 0xFE00 = use global)

#include <GLES3/gl3.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "GLES3/4JLibs/inc/4J_Render.h"

// ---------------------------------------------------------------------------
// Compile-time settings
// ---------------------------------------------------------------------------

static const int  kMaxTextures    = 4096;
static const int  kMaxCBuffers    = 8192;
static const int  kMatrixDepth    = 32;
static const int  kMaxLights      = 2;

// Sentinel value written into the lightmap UV field when Tesselator has no per-vertex
// lightmap UV set.  Written as 0xFE00 (= -512 as signed int16) in both channels.
// The GLSL shaders test for < -400.0 (post int16→float conversion) to detect this.
static const int kLightUVUseGlobal = -512;  // 0xFE00 as signed int16

// ---------------------------------------------------------------------------
// 4×4 column-major matrix helpers
// ---------------------------------------------------------------------------

static void mat4_identity(float m[16])
{
	memset(m, 0, 64);
	m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat4_copy(float dst[16], const float src[16])
{
	memcpy(dst, src, 64);
}

// result = a * b  (column-major)
static void mat4_mul(float result[16], const float a[16], const float b[16])
{
	float tmp[16];
	for (int col = 0; col < 4; ++col)
	{
		for (int row = 0; row < 4; ++row)
		{
			float s = 0.0f;
			for (int k = 0; k < 4; ++k)
				s += a[k * 4 + row] * b[col * 4 + k];
			tmp[col * 4 + row] = s;
		}
	}
	mat4_copy(result, tmp);
}

// Pre-multiply: m = translate * m
static void mat4_translate(float m[16], float x, float y, float z)
{
	float t[16]; mat4_identity(t);
	t[12] = x; t[13] = y; t[14] = z;
	float tmp[16]; mat4_copy(tmp, m);
	mat4_mul(m, tmp, t);
}

// Pre-multiply: m = rotate * m
static void mat4_rotate(float m[16], float angle, float x, float y, float z)
{
	// Normalise axis
	float len = sqrtf(x * x + y * y + z * z);
	if (len < 1e-6f) return;
	x /= len; y /= len; z /= len;

	float c = cosf(angle), s = sinf(angle), ic = 1.0f - c;
	float r[16]; mat4_identity(r);
	r[0]  = x * x * ic + c;     r[1]  = y * x * ic + z * s; r[2]  = x * z * ic - y * s;
	r[4]  = x * y * ic - z * s; r[5]  = y * y * ic + c;     r[6]  = y * z * ic + x * s;
	r[8]  = x * z * ic + y * s; r[9]  = y * z * ic - x * s; r[10] = z * z * ic + c;

	float tmp[16]; mat4_copy(tmp, m);
	mat4_mul(m, tmp, r);
}

// Pre-multiply: m = scale * m
static void mat4_scale(float m[16], float x, float y, float z)
{
	float s[16]; mat4_identity(s);
	s[0] = x; s[5] = y; s[10] = z;
	float tmp[16]; mat4_copy(tmp, m);
	mat4_mul(m, tmp, s);
}

static void mat4_perspective(float m[16], float fovyRad, float aspect,
                              float zNear, float zFar)
{
	float f = 1.0f / tanf(fovyRad * 0.5f);
	float nf = 1.0f / (zNear - zFar);
	mat4_identity(m);
	m[0]  = f / aspect;
	m[5]  = f;
	m[10] = (zFar + zNear) * nf;
	m[11] = -1.0f;
	m[14] = 2.0f * zFar * zNear * nf;
	m[15] = 0.0f;
}

static void mat4_ortho(float m[16], float left, float right, float bottom,
                        float top, float zNear, float zFar)
{
	mat4_identity(m);
	m[0]  = 2.0f / (right - left);
	m[5]  = 2.0f / (top - bottom);
	m[10] = -2.0f / (zFar - zNear);
	m[12] = -(right + left) / (right - left);
	m[13] = -(top + bottom) / (top - bottom);
	m[14] = -(zFar + zNear) / (zFar - zNear);
}

// ---------------------------------------------------------------------------
// Render state
// ---------------------------------------------------------------------------

struct MatrixStack
{
	float stack[kMatrixDepth][16];
	int   top;

	MatrixStack() : top(0) { mat4_identity(stack[0]); }

	float*       current()       { return stack[top]; }
	const float* current() const { return stack[top]; }

	void push()
	{
		if (top + 1 < kMatrixDepth)
		{
			mat4_copy(stack[top + 1], stack[top]);
			++top;
		}
	}

	void pop()
	{
		if (top > 0) --top;
	}
};

struct LightState
{
	bool  enabled;
	float colour[3];
	float direction[3];
	LightState() : enabled(false)
	{
		colour[0] = colour[1] = colour[2] = 1.0f;
		direction[0] = 0.0f; direction[1] = 1.0f; direction[2] = 0.0f;
	}
};

struct TexGenCol
{
	float plane[4];
	bool  eyeSpace;
	TexGenCol() : eyeSpace(false) { memset(plane, 0, sizeof(plane)); }
};

struct GlesState
{
	// Matrix stacks
	MatrixStack  mvStack;
	MatrixStack  projStack;
	MatrixStack  texStack;
	int          matrixMode;   // 0=MV, 1=Proj, 2=Tex

	// Current colour
	float        colour[4];

	// Global lightmap UV
	float        vertexTexU, vertexTexV;

	// Fog
	bool         fogEnabled;
	int          fogMode;    // GL_EXP=2, linear uses near/far
	float        fogNear, fogFar, fogDensity;
	float        fogColour[3];

	// Lighting
	bool         lightingEnabled;
	float        lightAmbient[3];
	LightState   lights[kMaxLights];

	// Blend
	bool         blendEnabled;
	int          blendSrc, blendDst;
	unsigned int blendFactor;

	// Depth
	bool         depthTestEnabled;
	bool         depthMask;
	int          depthFunc;
	float        depthBiasSlope, depthBiasConst;

	// Stencil
	int          stencilFunc;
	uint8_t      stencilRef, stencilFuncMask, stencilWriteMask;

	// Cull
	bool         cullEnabled;
	bool         cullCW;

	// Write masks
	bool         writeR, writeG, writeB, writeA;

	// Alpha test
	bool         alphaTestEnabled;
	int          alphaFunc;
	float        alphaFuncParam;

	// TexGen
	TexGenCol    texGenCols[4];

	// Force LOD
	int          forceLOD;

	// Textures (bound IDs in our table)
	int          boundTex;
	int          boundTexVertex;

	// Viewport
	int          vpX, vpY, vpW, vpH;
	int          screenW, screenH;

	// Clear colour
	float        clearColour[4];

	GlesState()
		: matrixMode(0)
		, fogEnabled(false), fogMode(GL_EXP), fogNear(1.0f), fogFar(100.0f), fogDensity(1.0f)
		, lightingEnabled(false)
		, blendEnabled(false), blendSrc(GL_SRC_ALPHA), blendDst(GL_ONE_MINUS_SRC_ALPHA), blendFactor(0)
		, depthTestEnabled(true), depthMask(true), depthFunc(GL_LEQUAL)
		, depthBiasSlope(0.0f), depthBiasConst(0.0f)
		, stencilFunc(GL_ALWAYS), stencilRef(0), stencilFuncMask(0xFF), stencilWriteMask(0xFF)
		, cullEnabled(false), cullCW(false)
		, writeR(true), writeG(true), writeB(true), writeA(true)
		, alphaTestEnabled(false), alphaFunc(GL_ALWAYS), alphaFuncParam(0.0f)
		, forceLOD(-1)
		, boundTex(-1), boundTexVertex(-1)
		, vpX(0), vpY(0), vpW(0), vpH(0)
		, screenW(0), screenH(0)
		, vertexTexU(0.0f), vertexTexV(0.0f)
	{
		colour[0] = colour[1] = colour[2] = colour[3] = 1.0f;
		fogColour[0] = fogColour[1] = fogColour[2] = 0.0f;
		lightAmbient[0] = lightAmbient[1] = lightAmbient[2] = 0.0f;
		clearColour[0] = clearColour[1] = clearColour[2] = 0.0f;
		clearColour[3] = 1.0f;
	}
};

// ---------------------------------------------------------------------------
// A single recorded draw call (for CBuffer playback)
// ---------------------------------------------------------------------------

struct DrawCmd
{
	// Full state snapshot at time of draw
	float        mvp[16];        // pre-multiplied projection * modelview
	float        mv[16];         // modelview only (for fog distance)
	int          boundTex;
	int          boundTexVertex;
	float        colour[4];
	float        vertexTexU, vertexTexV;

	bool         fogEnabled;
	int          fogMode;
	float        fogNear, fogFar, fogDensity;
	float        fogColour[3];

	bool         lightingEnabled;
	bool         lights[kMaxLights];
	float        lightColour[kMaxLights][3];
	float        lightDir[kMaxLights][3];
	float        lightAmbient[3];

	bool         blendEnabled;
	int          blendSrc, blendDst;
	unsigned int blendFactor;

	bool         depthTestEnabled;
	bool         depthMask;
	int          depthFunc;
	float        depthBiasSlope, depthBiasConst;

	bool         cullEnabled;
	bool         cullCW;

	bool         writeR, writeG, writeB, writeA;

	bool         alphaTestEnabled;
	int          alphaFunc;
	float        alphaFuncParam;

	TexGenCol    texGenCols[4];

	int          forceLOD;
	int          stencilFunc;
	uint8_t      stencilRef, stencilFuncMask, stencilWriteMask;

	// Draw parameters
	C4JRender::ePrimitiveType  primitiveType;
	int                        vertexCount;
	C4JRender::eVertexType     vType;
	C4JRender::ePixelShaderType psType;
	std::vector<uint8_t>       vertexData;
};

struct CBuffer
{
	std::vector<DrawCmd> cmds;
};

// ---------------------------------------------------------------------------
// Shader sources – GLSL ES 3.00
// ---------------------------------------------------------------------------

// Vertex shader for VERTEX_TYPE_PF3_TF2_CB4_NB4_XW1 (and _LIT, _TEXGEN)
static const char* kVSStandard = R"(#version 300 es
precision highp float;

// vertex attributes (layout matches the 32-byte C4JRender format)
layout(location = 0) in vec3  a_pos;       // float[3]  position
layout(location = 1) in vec2  a_uv;        // float[2]  texcoord  (u>1 => no mipmap)
layout(location = 2) in vec4  a_colour;    // ubyte[4]  RGBA / 255
layout(location = 3) in vec4  a_normal;    // ubyte[4]  normal / 255
layout(location = 4) in vec2  a_lightUV;  // int16[2]  lightmap UV (raw, scale /256)

// uniforms
uniform mat4  u_mvp;
uniform mat4  u_mv;
uniform vec4  u_colour;
uniform vec2  u_globalLightUV;

uniform bool  u_lightingEnabled;
uniform bool  u_light0Enabled;
uniform bool  u_light1Enabled;
uniform vec3  u_lightDir[2];
uniform vec3  u_lightColour[2];
uniform vec3  u_lightAmbient;
uniform bool  u_preLit;          // true for VERTEX_TYPE_…_LIT (skip lighting)

uniform bool  u_fogEnabled;
uniform int   u_fogMode;         // GL_EXP=2, otherwise linear
uniform float u_fogNear;
uniform float u_fogFar;
uniform float u_fogDensity;

// texgen uniforms (used when psType == PROJECTION)
uniform bool  u_useTexGen;
uniform vec4  u_texGenS;
uniform vec4  u_texGenT;
uniform bool  u_texGenSEye;
uniform bool  u_texGenTEye;

out vec2  v_uv;
out vec2  v_lightUV;
out vec4  v_colour;
out float v_fog;
out vec4  v_eyePos;   // eye-space position for texgen

void main()
{
    vec4 worldPos  = vec4(a_pos, 1.0);
    vec4 eyePos    = u_mv  * worldPos;
    gl_Position    = u_mvp * worldPos;
    v_eyePos       = eyePos;

    // ---- UV ----
    if (u_useTexGen)
    {
        // Projective tex-gen: compute UVs from eye-space position
        vec4 ep = u_texGenSEye ? eyePos : worldPos;
        vec4 tp = u_texGenTEye ? eyePos : worldPos;
        v_uv = vec2(dot(u_texGenS, ep), dot(u_texGenT, tp));
    }
    else
    {
        v_uv = a_uv;
    }

    // ---- Lightmap UV ----
    // The sentinel value 0xFE00 = -512 as int16 means "use global"
    if (a_lightUV.x < -400.0)
        v_lightUV = u_globalLightUV;
    else
        v_lightUV = a_lightUV / 256.0;

    // ---- Colour + lighting ----
    vec4 vcol = a_colour * u_colour;

    if (u_lightingEnabled && !u_preLit)
    {
        // Decode normal: bytes / 255 maps 0..1, centre at 0.5 -> [-0.5, 0.5]
        vec3 N = normalize(a_normal.xyz - 0.5);

        vec3 diffuse = u_lightAmbient;
        if (u_light0Enabled)
            diffuse += u_lightColour[0] * max(dot(N, normalize(u_lightDir[0])), 0.0);
        if (u_light1Enabled)
            diffuse += u_lightColour[1] * max(dot(N, normalize(u_lightDir[1])), 0.0);

        vcol.rgb *= clamp(diffuse, vec3(0.0), vec3(1.0));
    }
    v_colour = vcol;

    // ---- Fog ----
    if (u_fogEnabled)
    {
        float dist = length(eyePos.xyz);
        if (u_fogMode == 2)   // GL_EXP
            v_fog = clamp(exp(-u_fogDensity * dist), 0.0, 1.0);
        else                  // linear
            v_fog = clamp((u_fogFar - dist) / (u_fogFar - u_fogNear), 0.0, 1.0);
    }
    else
    {
        v_fog = 1.0;
    }
}
)";

// Vertex shader for VERTEX_TYPE_COMPRESSED (16 bytes per vertex)
static const char* kVSCompressed = R"(#version 300 es
precision highp float;

// Each vertex is 8 x int16 (16 bytes)
// Passed as ivec4 pairs so we can access them from GLES3
layout(location = 0) in ivec4 a_data0;  // [x, y, z, packedColour]
layout(location = 1) in ivec4 a_data1;  // [u, v, lightU, lightV]

uniform mat4  u_mvp;
uniform mat4  u_mv;
uniform vec4  u_colour;
uniform vec2  u_globalLightUV;

uniform bool  u_fogEnabled;
uniform int   u_fogMode;
uniform float u_fogNear;
uniform float u_fogFar;
uniform float u_fogDensity;

// Texgen (reused from standard)
uniform bool  u_useTexGen;
uniform vec4  u_texGenS;
uniform vec4  u_texGenT;
uniform bool  u_texGenSEye;
uniform bool  u_texGenTEye;

out vec2  v_uv;
out vec2  v_lightUV;
out vec4  v_colour;
out float v_fog;
out vec4  v_eyePos;

void main()
{
    // Decode position: raw / 1024
    vec3 pos = vec3(float(a_data0.x), float(a_data0.y), float(a_data0.z)) / 1024.0;

    // Decode colour: stored as RGB565 - 0x8000 (signed int16)
    // Reconstruct unsigned: add 0x8000 (32768)
    int raw = a_data0.w + 32768;
    float r = float((raw >> 11) & 31)  / 31.0;
    float g = float((raw >>  5) & 63)  / 63.0;
    float b = float( raw        & 31)  / 31.0;

    // Decode UV: raw / 8192
    float u = float(a_data1.x) / 8192.0;
    float v = float(a_data1.y) / 8192.0;

    vec4 worldPos = vec4(pos, 1.0);
    vec4 eyePos   = u_mv  * worldPos;
    gl_Position   = u_mvp * worldPos;
    v_eyePos      = eyePos;

    if (u_useTexGen)
    {
        vec4 ep = u_texGenSEye ? eyePos : worldPos;
        vec4 tp = u_texGenTEye ? eyePos : worldPos;
        v_uv = vec2(dot(u_texGenS, ep), dot(u_texGenT, tp));
    }
    else
    {
        v_uv = vec2(u, v);
    }

    // Lightmap UV
    if (a_data1.z < -400)
        v_lightUV = u_globalLightUV;
    else
        v_lightUV = vec2(float(a_data1.z), float(a_data1.w)) / 256.0;

    v_colour = vec4(r, g, b, 1.0) * u_colour;

    // Fog
    if (u_fogEnabled)
    {
        float dist = length(eyePos.xyz);
        if (u_fogMode == 2)
            v_fog = clamp(exp(-u_fogDensity * dist), 0.0, 1.0);
        else
            v_fog = clamp((u_fogFar - dist) / (u_fogFar - u_fogNear), 0.0, 1.0);
    }
    else
    {
        v_fog = 1.0;
    }
}
)";

// Fragment shader – shared by both vertex programs
static const char* kFS = R"(#version 300 es
precision mediump float;

uniform sampler2D u_sampler;        // main texture  (unit 0)
uniform sampler2D u_samplerLight;   // lightmap       (unit 1)
uniform bool      u_hasTexture;
uniform bool      u_hasLightmap;

uniform bool      u_alphaTestEnabled;
uniform int       u_alphaFunc;      // GL_ALWAYS=0x207, GL_GREATER=0x204, GL_GEQUAL=0x206
uniform float     u_alphaFuncParam;

uniform bool      u_fogEnabled;
uniform vec3      u_fogColour;

uniform bool      u_forceLOD;       // when true, sample at LOD 0 (mipmap disabled)
uniform int       u_forceLODLevel;

in vec2  v_uv;
in vec2  v_lightUV;
in vec4  v_colour;
in float v_fog;
in vec4  v_eyePos;

out vec4 fragColor;

void main()
{
    vec4 col = v_colour;

    if (u_hasTexture)
    {
        vec2 sampleUV = v_uv;
        vec4 texel;
        if (u_forceLOD)
            texel = textureLod(u_sampler, sampleUV, float(u_forceLODLevel));
        else
            texel = texture(u_sampler, sampleUV);
        col *= texel;
    }

    if (u_hasLightmap)
    {
        col.rgb *= texture(u_samplerLight, v_lightUV).rgb;
    }

    // Alpha test (emulated – GLES3 has no fixed-function alpha test)
    if (u_alphaTestEnabled)
    {
        bool pass;
        if      (u_alphaFunc == 0x0204) pass = col.a >  u_alphaFuncParam;  // GL_GREATER
        else if (u_alphaFunc == 0x0206) pass = col.a >= u_alphaFuncParam;  // GL_GEQUAL
        else if (u_alphaFunc == 0x0203) pass = col.a <= u_alphaFuncParam;  // GL_LEQUAL
        else if (u_alphaFunc == 0x0202) pass = col.a == u_alphaFuncParam;  // GL_EQUAL
        else                            pass = true;                        // GL_ALWAYS
        if (!pass) discard;
    }

    // Fog
    if (u_fogEnabled)
        col.rgb = mix(u_fogColour, col.rgb, v_fog);

    fragColor = col;
}
)";

// ---------------------------------------------------------------------------
// Internal globals
// ---------------------------------------------------------------------------

static GlesState g_state;

// Texture pool
static GLuint g_texObjects[kMaxTextures];
static bool   g_texValid[kMaxTextures];
static int    g_texLevels[kMaxTextures];          // pending mip level count
static int    g_texParamMin[kMaxTextures];
static int    g_texParamMag[kMaxTextures];
static int    g_texParamWrapS[kMaxTextures];
static int    g_texParamWrapT[kMaxTextures];
static int    g_texPendingLevels;                 // for next TextureData call

// Shader programs
static GLuint g_progStandard;     // standard vertex format
static GLuint g_progCompressed;   // compressed vertex format

// Uniform locations – standard program
struct ProgUniforms
{
	GLint mvp, mv, colour, globalLightUV;
	GLint lightingEnabled, light0Enabled, light1Enabled;
	GLint lightDir, lightColour, lightAmbient, preLit;
	GLint fogEnabled, fogMode, fogNear, fogFar, fogDensity;
	GLint useTexGen, texGenS, texGenT, texGenSEye, texGenTEye;
	GLint sampler, samplerLight;
	GLint hasTexture, hasLightmap;
	GLint alphaTestEnabled, alphaFunc, alphaFuncParam;
	GLint fogColour;
	GLint forceLOD, forceLODLevel;
};

static ProgUniforms g_uStd;
static ProgUniforms g_uComp;

// Command buffer pool
static CBuffer* g_cbufs[kMaxCBuffers];
static bool     g_cbufValid[kMaxCBuffers];

// Recording state
static bool g_recording      = false;
static int  g_recordingIndex = -1;

// Deferred-mode recording
static bool g_deferredMode = false;

// Transient VBO/VAO used for immediate draws
static GLuint g_transientVBO = 0;
static GLuint g_transientVAO_std  = 0;
static GLuint g_transientVAO_comp = 0;

// Singleton
C4JRender RenderManager;

// ---------------------------------------------------------------------------
// Shader compilation helpers
// ---------------------------------------------------------------------------

static GLuint compileShader(GLenum type, const char* src)
{
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, nullptr);
	glCompileShader(s);

	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok)
	{
		char buf[2048];
		glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
		fprintf(stderr, "C4JRender: shader compile error:\n%s\n", buf);
		glDeleteShader(s);
		return 0;
	}
	return s;
}

static GLuint linkProgram(GLuint vs, GLuint fs)
{
	GLuint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);

	GLint ok = 0;
	glGetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok)
	{
		char buf[2048];
		glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
		fprintf(stderr, "C4JRender: program link error:\n%s\n", buf);
		glDeleteProgram(prog);
		return 0;
	}
	return prog;
}

static void bindUniforms(ProgUniforms& u, GLuint prog)
{
#define LOC(name) u.name = glGetUniformLocation(prog, #name)
	LOC(mvp); LOC(mv); LOC(colour); LOC(globalLightUV);
	LOC(lightingEnabled); LOC(light0Enabled); LOC(light1Enabled);
	LOC(lightDir); LOC(lightColour); LOC(lightAmbient); LOC(preLit);
	LOC(fogEnabled); LOC(fogMode); LOC(fogNear); LOC(fogFar); LOC(fogDensity);
	LOC(useTexGen); LOC(texGenS); LOC(texGenT); LOC(texGenSEye); LOC(texGenTEye);
	LOC(sampler); LOC(samplerLight);
	LOC(hasTexture); LOC(hasLightmap);
	LOC(alphaTestEnabled); LOC(alphaFunc); LOC(alphaFuncParam);
	LOC(fogColour);
	LOC(forceLOD); LOC(forceLODLevel);
#undef LOC
}

// ---------------------------------------------------------------------------
// VAO / VBO setup helpers
// ---------------------------------------------------------------------------

// Standard format: 32 bytes per vertex
// layout: pos(3f)@0  uv(2f)@12  col(4ub)@20  nrm(4ub)@24  lightUV(2s)@28
static void setupVAO_standard(GLuint vao, GLuint vbo)
{
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	const GLsizei stride = 32;

	// location 0: position – 3 floats
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);

	// location 1: texcoord – 2 floats
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)12);

	// location 2: colour – 4 unsigned bytes (normalized to 0..1)
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride, (void*)20);

	// location 3: normal – 4 unsigned bytes (normalized to 0..1)
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride, (void*)24);

	// location 4: lightmap UV – 2 signed int16 (NOT normalised – raw values used in shader)
	glEnableVertexAttribArray(4);
	glVertexAttribIPointer(4, 2, GL_SHORT, stride, (void*)28);

	glBindVertexArray(0);
}

// Compressed format: 16 bytes per vertex
// layout: data0(4 x int16)@0  data1(4 x int16)@8
static void setupVAO_compressed(GLuint vao, GLuint vbo)
{
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	const GLsizei stride = 16;

	// location 0: a_data0 – ivec4 (pos xyz + colour)
	glEnableVertexAttribArray(0);
	glVertexAttribIPointer(0, 4, GL_SHORT, stride, (void*)0);

	// location 1: a_data1 – ivec4 (uv + lightUV)
	glEnableVertexAttribArray(1);
	glVertexAttribIPointer(1, 4, GL_SHORT, stride, (void*)8);

	glBindVertexArray(0);
}

// ---------------------------------------------------------------------------
// Map C4JRender primitive type to GL primitive type.
// QUAD_LIST is emulated by expanding quads to two triangles at upload time.
// ---------------------------------------------------------------------------

static GLenum primTypeToGL(C4JRender::ePrimitiveType t)
{
	switch (t)
	{
	case C4JRender::PRIMITIVE_TYPE_TRIANGLE_LIST:  return GL_TRIANGLES;
	case C4JRender::PRIMITIVE_TYPE_TRIANGLE_STRIP: return GL_TRIANGLE_STRIP;
	case C4JRender::PRIMITIVE_TYPE_TRIANGLE_FAN:   return GL_TRIANGLE_FAN;
	case C4JRender::PRIMITIVE_TYPE_QUAD_LIST:      return GL_TRIANGLES; // expanded below
	case C4JRender::PRIMITIVE_TYPE_LINE_LIST:      return GL_LINES;
	case C4JRender::PRIMITIVE_TYPE_LINE_STRIP:     return GL_LINE_STRIP;
	default:                                        return GL_TRIANGLES;
	}
}

// Expand quad vertex data to triangle list data.
// Each group of 4 verts becomes 2 triangles (6 verts): 0,1,2, 0,2,3
static std::vector<uint8_t> expandQuads(const void* data, int quadCount, int bytesPerVert)
{
	const uint8_t* src = static_cast<const uint8_t*>(data);
	std::vector<uint8_t> out;
	out.resize(static_cast<size_t>(quadCount) * 6 * bytesPerVert);
	uint8_t* dst = out.data();

	for (int q = 0; q < quadCount; ++q)
	{
		const uint8_t* v0 = src + static_cast<ptrdiff_t>(q) * 4 * bytesPerVert;
		const uint8_t* v1 = v0 + bytesPerVert;
		const uint8_t* v2 = v0 + 2 * bytesPerVert;
		const uint8_t* v3 = v0 + 3 * bytesPerVert;

		memcpy(dst, v0, bytesPerVert); dst += bytesPerVert;
		memcpy(dst, v1, bytesPerVert); dst += bytesPerVert;
		memcpy(dst, v2, bytesPerVert); dst += bytesPerVert;
		memcpy(dst, v0, bytesPerVert); dst += bytesPerVert;
		memcpy(dst, v2, bytesPerVert); dst += bytesPerVert;
		memcpy(dst, v3, bytesPerVert); dst += bytesPerVert;
	}
	return out;
}

// ---------------------------------------------------------------------------
// Apply uniform state to the currently-bound shader program
// ---------------------------------------------------------------------------

static void applyUniforms(const ProgUniforms& u, const DrawCmd& cmd)
{
	glUniformMatrix4fv(u.mvp, 1, GL_FALSE, cmd.mvp);
	glUniformMatrix4fv(u.mv,  1, GL_FALSE, cmd.mv);
	glUniform4fv(u.colour, 1, cmd.colour);
	glUniform2f(u.globalLightUV, cmd.vertexTexU, cmd.vertexTexV);

	glUniform1i(u.lightingEnabled, cmd.lightingEnabled ? 1 : 0);
	glUniform1i(u.light0Enabled,   cmd.lights[0] ? 1 : 0);
	glUniform1i(u.light1Enabled,   cmd.lights[1] ? 1 : 0);
	glUniform3fv(u.lightDir,    2, cmd.lightDir[0]);
	glUniform3fv(u.lightColour, 2, cmd.lightColour[0]);
	glUniform3fv(u.lightAmbient, 1, cmd.lightAmbient);
	glUniform1i(u.preLit, (cmd.vType == C4JRender::VERTEX_TYPE_PF3_TF2_CB4_NB4_XW1_LIT) ? 1 : 0);

	glUniform1i(u.fogEnabled,  cmd.fogEnabled ? 1 : 0);
	glUniform1i(u.fogMode,     cmd.fogMode);
	glUniform1f(u.fogNear,     cmd.fogNear);
	glUniform1f(u.fogFar,      cmd.fogFar);
	glUniform1f(u.fogDensity,  cmd.fogDensity);
	glUniform3fv(u.fogColour,  1, cmd.fogColour);

	bool useTexGen = (cmd.vType == C4JRender::VERTEX_TYPE_PF3_TF2_CB4_NB4_XW1_TEXGEN ||
	                  cmd.psType == C4JRender::PIXEL_SHADER_TYPE_PROJECTION);
	glUniform1i(u.useTexGen,    useTexGen ? 1 : 0);
	glUniform4fv(u.texGenS,     1, cmd.texGenCols[0].plane);
	glUniform4fv(u.texGenT,     1, cmd.texGenCols[1].plane);
	glUniform1i(u.texGenSEye,   cmd.texGenCols[0].eyeSpace ? 1 : 0);
	glUniform1i(u.texGenTEye,   cmd.texGenCols[1].eyeSpace ? 1 : 0);

	// Texture units
	glUniform1i(u.sampler,      0);
	glUniform1i(u.samplerLight, 1);

	bool hasTex   = (cmd.boundTex >= 0 && g_texValid[cmd.boundTex]);
	bool hasLight = (cmd.boundTexVertex >= 0 && g_texValid[cmd.boundTexVertex]);
	glUniform1i(u.hasTexture,  hasTex ? 1 : 0);
	glUniform1i(u.hasLightmap, hasLight ? 1 : 0);

	glUniform1i(u.alphaTestEnabled, cmd.alphaTestEnabled ? 1 : 0);
	glUniform1i(u.alphaFunc,        cmd.alphaFunc);
	glUniform1f(u.alphaFuncParam,   cmd.alphaFuncParam);

	bool forceLOD = (cmd.forceLOD >= 0 &&
	                 cmd.psType == C4JRender::PIXEL_SHADER_TYPE_FORCELOD);
	glUniform1i(u.forceLOD,      forceLOD ? 1 : 0);
	glUniform1i(u.forceLODLevel, cmd.forceLOD >= 0 ? cmd.forceLOD : 0);
}

// ---------------------------------------------------------------------------
// Apply GL hardware state from a DrawCmd snapshot
// ---------------------------------------------------------------------------

static void applyHWState(const DrawCmd& cmd)
{
	// Blend
	if (cmd.blendEnabled)
	{
		glEnable(GL_BLEND);
		glBlendFunc((GLenum)cmd.blendSrc, (GLenum)cmd.blendDst);
	}
	else
	{
		glDisable(GL_BLEND);
	}

	// Depth
	if (cmd.depthTestEnabled)
		glEnable(GL_DEPTH_TEST);
	else
		glDisable(GL_DEPTH_TEST);

	glDepthMask(cmd.depthMask ? GL_TRUE : GL_FALSE);
	glDepthFunc((GLenum)cmd.depthFunc);

	if (cmd.depthBiasSlope != 0.0f || cmd.depthBiasConst != 0.0f)
	{
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(cmd.depthBiasSlope, cmd.depthBiasConst);
	}
	else
	{
		glDisable(GL_POLYGON_OFFSET_FILL);
	}

	// Culling
	if (cmd.cullEnabled)
	{
		glEnable(GL_CULL_FACE);
		glFrontFace(cmd.cullCW ? GL_CW : GL_CCW);
		glCullFace(GL_BACK);
	}
	else
	{
		glDisable(GL_CULL_FACE);
	}

	// Write masks
	glColorMask(cmd.writeR ? GL_TRUE : GL_FALSE,
	             cmd.writeG ? GL_TRUE : GL_FALSE,
	             cmd.writeB ? GL_TRUE : GL_FALSE,
	             cmd.writeA ? GL_TRUE : GL_FALSE);

	// Stencil
	glStencilFunc((GLenum)cmd.stencilFunc, (GLint)cmd.stencilRef, (GLuint)cmd.stencilFuncMask);
	glStencilMask((GLuint)cmd.stencilWriteMask);

	// Textures
	glActiveTexture(GL_TEXTURE0);
	if (cmd.boundTex >= 0 && g_texValid[cmd.boundTex])
		glBindTexture(GL_TEXTURE_2D, g_texObjects[cmd.boundTex]);
	else
		glBindTexture(GL_TEXTURE_2D, 0);

	glActiveTexture(GL_TEXTURE1);
	if (cmd.boundTexVertex >= 0 && g_texValid[cmd.boundTexVertex])
		glBindTexture(GL_TEXTURE_2D, g_texObjects[cmd.boundTexVertex]);
	else
		glBindTexture(GL_TEXTURE_2D, 0);

	glActiveTexture(GL_TEXTURE0);
}

// ---------------------------------------------------------------------------
// Execute a single DrawCmd (called both for immediate draws and CBuffer replay)
// ---------------------------------------------------------------------------

static void executeDrawCmd(const DrawCmd& cmd)
{
	applyHWState(cmd);

	bool isCompressed = (cmd.vType == C4JRender::VERTEX_TYPE_COMPRESSED);
	GLuint prog = isCompressed ? g_progCompressed : g_progStandard;
	glUseProgram(prog);

	const ProgUniforms& u = isCompressed ? g_uComp : g_uStd;
	applyUniforms(u, cmd);

	// Build draw data (expand quads if needed)
	const void*  drawData    = cmd.vertexData.data();
	int          drawCount   = cmd.vertexCount;
	GLenum       glPrim      = primTypeToGL(cmd.primitiveType);
	int          bytesPerVert = isCompressed ? 16 : 32;

	std::vector<uint8_t> expanded;
	if (cmd.primitiveType == C4JRender::PRIMITIVE_TYPE_QUAD_LIST)
	{
		int quadCount = cmd.vertexCount / 4;
		expanded = expandQuads(drawData, quadCount, bytesPerVert);
		drawData  = expanded.data();
		drawCount = quadCount * 6;
	}

	// Upload to transient VBO
	GLuint vbo = g_transientVBO;
	GLuint vao = isCompressed ? g_transientVAO_comp : g_transientVAO_std;

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER,
	             static_cast<GLsizeiptr>(drawCount * bytesPerVert),
	             drawData, GL_STREAM_DRAW);

	// Re-bind VAO to pick up the new VBO data
	if (isCompressed)
		setupVAO_compressed(vao, vbo);
	else
		setupVAO_standard(vao, vbo);

	glBindVertexArray(vao);
	glDrawArrays(glPrim, 0, drawCount);
	glBindVertexArray(0);
}

// ---------------------------------------------------------------------------
// Capture current state into a DrawCmd
// ---------------------------------------------------------------------------

static void captureState(DrawCmd& cmd)
{
	// Compute MVP = projection * modelview
	mat4_mul(cmd.mvp, g_state.projStack.current(), g_state.mvStack.current());
	mat4_copy(cmd.mv, g_state.mvStack.current());

	cmd.boundTex       = g_state.boundTex;
	cmd.boundTexVertex = g_state.boundTexVertex;
	memcpy(cmd.colour, g_state.colour, 16);
	cmd.vertexTexU     = g_state.vertexTexU;
	cmd.vertexTexV     = g_state.vertexTexV;

	cmd.fogEnabled     = g_state.fogEnabled;
	cmd.fogMode        = g_state.fogMode;
	cmd.fogNear        = g_state.fogNear;
	cmd.fogFar         = g_state.fogFar;
	cmd.fogDensity     = g_state.fogDensity;
	memcpy(cmd.fogColour, g_state.fogColour, 12);

	cmd.lightingEnabled = g_state.lightingEnabled;
	for (int i = 0; i < kMaxLights; ++i)
	{
		cmd.lights[i] = g_state.lights[i].enabled;
		memcpy(cmd.lightColour[i], g_state.lights[i].colour, 12);
		memcpy(cmd.lightDir[i],    g_state.lights[i].direction, 12);
	}
	memcpy(cmd.lightAmbient, g_state.lightAmbient, 12);

	cmd.blendEnabled   = g_state.blendEnabled;
	cmd.blendSrc       = g_state.blendSrc;
	cmd.blendDst       = g_state.blendDst;
	cmd.blendFactor    = g_state.blendFactor;

	cmd.depthTestEnabled = g_state.depthTestEnabled;
	cmd.depthMask        = g_state.depthMask;
	cmd.depthFunc        = g_state.depthFunc;
	cmd.depthBiasSlope   = g_state.depthBiasSlope;
	cmd.depthBiasConst   = g_state.depthBiasConst;

	cmd.cullEnabled = g_state.cullEnabled;
	cmd.cullCW      = g_state.cullCW;

	cmd.writeR = g_state.writeR;
	cmd.writeG = g_state.writeG;
	cmd.writeB = g_state.writeB;
	cmd.writeA = g_state.writeA;

	cmd.alphaTestEnabled = g_state.alphaTestEnabled;
	cmd.alphaFunc        = g_state.alphaFunc;
	cmd.alphaFuncParam   = g_state.alphaFuncParam;

	for (int i = 0; i < 4; ++i)
		cmd.texGenCols[i] = g_state.texGenCols[i];

	cmd.forceLOD = g_state.forceLOD;

	cmd.stencilFunc      = g_state.stencilFunc;
	cmd.stencilRef       = g_state.stencilRef;
	cmd.stencilFuncMask  = g_state.stencilFuncMask;
	cmd.stencilWriteMask = g_state.stencilWriteMask;
}

// ---------------------------------------------------------------------------
// MatrixStack helpers
// ---------------------------------------------------------------------------

static MatrixStack& currentStack()
{
	if      (g_state.matrixMode == GL_PROJECTION) return g_state.projStack;
	else if (g_state.matrixMode == GL_TEXTURE)    return g_state.texStack;
	else                                           return g_state.mvStack;
}

// ---------------------------------------------------------------------------
// C4JRender method implementations
// ---------------------------------------------------------------------------

void C4JRender::Tick()
{
	// Nothing per-tick required on GLES3
}

void C4JRender::UpdateGamma(unsigned short /*usGamma*/)
{
	// GLES3 has no built-in gamma correction; a post-process pass could be added
}

// ---- Matrix stack ----

void C4JRender::MatrixMode(int type)
{
	g_state.matrixMode = type;
}

void C4JRender::MatrixSetIdentity()
{
	mat4_identity(currentStack().current());
}

void C4JRender::MatrixTranslate(float x, float y, float z)
{
	mat4_translate(currentStack().current(), x, y, z);
}

void C4JRender::MatrixRotate(float angle, float x, float y, float z)
{
	mat4_rotate(currentStack().current(), angle, x, y, z);
}

void C4JRender::MatrixScale(float x, float y, float z)
{
	mat4_scale(currentStack().current(), x, y, z);
}

void C4JRender::MatrixPerspective(float fovy, float aspect, float zNear, float zFar)
{
	float p[16];
	mat4_perspective(p, fovy, aspect, zNear, zFar);
	mat4_mul(currentStack().current(), currentStack().current(), p);
}

void C4JRender::MatrixOrthogonal(float left, float right, float bottom, float top,
                                  float zNear, float zFar)
{
	float o[16];
	mat4_ortho(o, left, right, bottom, top, zNear, zFar);
	mat4_mul(currentStack().current(), currentStack().current(), o);
}

void C4JRender::MatrixPop()
{
	currentStack().pop();
}

void C4JRender::MatrixPush()
{
	currentStack().push();
}

void C4JRender::MatrixMult(float* mat)
{
	float tmp[16];
	mat4_copy(tmp, currentStack().current());
	mat4_mul(currentStack().current(), tmp, mat);
}

const float* C4JRender::MatrixGet(int type)
{
	if      (type == GL_PROJECTION) return g_state.projStack.current();
	else if (type == GL_TEXTURE)    return g_state.texStack.current();
	else                            return g_state.mvStack.current();
}

void C4JRender::Set_matrixDirty()
{
	// Nothing needed – we always recompute MVP per draw call
}

// ---- Core ----

void C4JRender::Initialise()
{
	// Initialise texture pool
	memset(g_texValid,    0, sizeof(g_texValid));
	memset(g_texObjects,  0, sizeof(g_texObjects));
	memset(g_cbufValid,   0, sizeof(g_cbufValid));
	memset(g_cbufs,       0, sizeof(g_cbufs));
	g_texPendingLevels = 1;

	for (int i = 0; i < kMaxTextures; ++i)
	{
		g_texParamMin[i]   = GL_NEAREST_MIPMAP_LINEAR;
		g_texParamMag[i]   = GL_LINEAR;
		g_texParamWrapS[i] = GL_REPEAT;
		g_texParamWrapT[i] = GL_REPEAT;
		g_texLevels[i]     = 1;
	}

	// Compile shaders
	GLuint vsStd  = compileShader(GL_VERTEX_SHADER,   kVSStandard);
	GLuint vsComp = compileShader(GL_VERTEX_SHADER,   kVSCompressed);
	GLuint fs     = compileShader(GL_FRAGMENT_SHADER, kFS);

	g_progStandard   = linkProgram(vsStd,  fs);
	g_progCompressed = linkProgram(vsComp, fs);

	glDeleteShader(vsStd);
	glDeleteShader(vsComp);
	glDeleteShader(fs);

	bindUniforms(g_uStd,  g_progStandard);
	bindUniforms(g_uComp, g_progCompressed);

	// Create transient VBO / VAOs
	glGenBuffers(1, &g_transientVBO);
	glGenVertexArrays(1, &g_transientVAO_std);
	glGenVertexArrays(1, &g_transientVAO_comp);

	// Default GL state
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(GL_TRUE);

	// Verify the sentinel value matches what Tesselator writes (0xFE00 as signed int16)
	static_assert(kLightUVUseGlobal == -512, "kLightUVUseGlobal must equal -512 (0xFE00 as signed int16)");
}

void C4JRender::InitialiseContext()
{
	// Called after the GL context is current – forward to Initialise
	Initialise();
}

void C4JRender::StartFrame(bool /*actualFrameStart*/)
{
	// Nothing needed per frame on GLES3
}

void C4JRender::Present()
{
	// EGL swap-buffers is handled by the platform app layer
}

void C4JRender::Clear(int flags)
{
	GLbitfield mask = 0;
	if (flags & CLEAR_COLOUR_FLAG)
	{
		glClearColor(g_state.clearColour[0], g_state.clearColour[1],
		             g_state.clearColour[2], g_state.clearColour[3]);
		mask |= GL_COLOR_BUFFER_BIT;
	}
	if (flags & CLEAR_DEPTH_FLAG)
		mask |= GL_DEPTH_BUFFER_BIT;
	if (mask)
		glClear(mask);
}

void C4JRender::SetClearColour(const float colourRGBA[4])
{
	memcpy(g_state.clearColour, colourRGBA, 16);
}

bool C4JRender::IsWidescreen() { return true;  }
bool C4JRender::IsHiDef()      { return true;  }

void C4JRender::InternalScreenCapture()                                   {}
void C4JRender::CaptureThumbnail(ImageFileBuffer* /*p*/, ImageFileBuffer* /*s*/) {}
void C4JRender::CaptureScreen(ImageFileBuffer* /*j*/, XSOCIAL_PREVIEWIMAGE* /*p*/) {}

void C4JRender::BeginConditionalSurvey(int /*id*/)  {}
void C4JRender::EndConditionalSurvey()              {}
void C4JRender::BeginConditionalRendering(int /*id*/) {}
void C4JRender::EndConditionalRendering()           {}

// ---- Drawing ----

void C4JRender::DrawVertices(ePrimitiveType primitiveType, int count,
                              void* dataIn, eVertexType vType,
                              ePixelShaderType psType)
{
	if (count <= 0 || dataIn == nullptr) return;

	bool isCompressed = (vType == VERTEX_TYPE_COMPRESSED);
	int bytesPerVert  = isCompressed ? 16 : 32;

	DrawCmd cmd;
	captureState(cmd);
	cmd.primitiveType = primitiveType;
	cmd.vertexCount   = count;
	cmd.vType         = vType;
	cmd.psType        = psType;

	// Copy vertex data
	int totalBytes = count * bytesPerVert;
	cmd.vertexData.resize(static_cast<size_t>(totalBytes));
	memcpy(cmd.vertexData.data(), dataIn, static_cast<size_t>(totalBytes));

	if (g_recording && g_recordingIndex >= 0)
	{
		// Store for later replay
		g_cbufs[g_recordingIndex]->cmds.push_back(std::move(cmd));
	}
	else
	{
		// Immediate draw
		executeDrawCmd(cmd);
	}
}

// ---- Command buffers ----

void C4JRender::CBuffLockStaticCreations() {}

int C4JRender::CBuffCreate(int count)
{
	// Find 'count' consecutive free slots
	for (int i = 0; i <= kMaxCBuffers - count; ++i)
	{
		bool free = true;
		for (int j = 0; j < count; ++j)
		{
			if (g_cbufValid[i + j]) { free = false; break; }
		}
		if (free)
		{
			for (int j = 0; j < count; ++j)
			{
				g_cbufs[i + j]     = new CBuffer();
				g_cbufValid[i + j] = true;
			}
			return i;
		}
	}
	return -1;
}

void C4JRender::CBuffDelete(int first, int count)
{
	for (int i = first; i < first + count; ++i)
	{
		if (i >= 0 && i < kMaxCBuffers && g_cbufValid[i])
		{
			delete g_cbufs[i];
			g_cbufs[i]     = nullptr;
			g_cbufValid[i] = false;
		}
	}
}

void C4JRender::CBuffStart(int index, bool /*full*/)
{
	if (index < 0 || index >= kMaxCBuffers) return;
	if (!g_cbufValid[index]) return;

	g_cbufs[index]->cmds.clear();
	g_recording      = true;
	g_recordingIndex = index;
}

void C4JRender::CBuffClear(int index)
{
	if (index >= 0 && index < kMaxCBuffers && g_cbufValid[index])
		g_cbufs[index]->cmds.clear();
}

int C4JRender::CBuffSize(int index)
{
	if (index < 0 || index >= kMaxCBuffers || !g_cbufValid[index]) return 0;
	return static_cast<int>(g_cbufs[index]->cmds.size());
}

void C4JRender::CBuffEnd()
{
	g_recording      = false;
	g_recordingIndex = -1;
}

bool C4JRender::CBuffCall(int index, bool /*full*/)
{
	if (index < 0 || index >= kMaxCBuffers || !g_cbufValid[index]) return false;
	for (const DrawCmd& cmd : g_cbufs[index]->cmds)
		executeDrawCmd(cmd);
	return true;
}

void C4JRender::CBuffTick()          {}
void C4JRender::CBuffDeferredModeStart() { g_deferredMode = true;  }
void C4JRender::CBuffDeferredModeEnd()   { g_deferredMode = false; }

// ---- Texture management ----

int C4JRender::TextureCreate()
{
	for (int i = 0; i < kMaxTextures; ++i)
	{
		if (!g_texValid[i])
		{
			glGenTextures(1, &g_texObjects[i]);
			g_texValid[i]     = true;
			g_texLevels[i]    = 1;
			g_texParamMin[i]  = GL_NEAREST_MIPMAP_LINEAR;
			g_texParamMag[i]  = GL_LINEAR;
			g_texParamWrapS[i]= GL_REPEAT;
			g_texParamWrapT[i]= GL_REPEAT;
			return i;
		}
	}
	return -1;
}

void C4JRender::TextureFree(int idx)
{
	if (idx < 0 || idx >= kMaxTextures || !g_texValid[idx]) return;
	glDeleteTextures(1, &g_texObjects[idx]);
	g_texValid[idx]   = false;
	g_texObjects[idx] = 0;
}

static void applyTexParams(int idx)
{
	// Filter mode tokens are defined as actual GLES3 enum values in 4J_Render.h,
	// so they can be passed directly to glTexParameteri.
	GLenum minF = (GLenum)g_texParamMin[idx];
	GLenum magF = (GLenum)g_texParamMag[idx];

	// Wrap mode tokens are C4JRender-local values (GL_CLAMP=0, GL_REPEAT=1).
	// Map them to the corresponding GLES3 enums.
	GLenum wrapS = (g_texParamWrapS[idx] == GL_CLAMP) ? GL_CLAMP_TO_EDGE : GL_REPEAT;
	GLenum wrapT = (g_texParamWrapT[idx] == GL_CLAMP) ? GL_CLAMP_TO_EDGE : GL_REPEAT;

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)minF);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)magF);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     (GLint)wrapS);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     (GLint)wrapT);
}

void C4JRender::TextureBind(int idx)
{
	g_state.boundTex = idx;
	glActiveTexture(GL_TEXTURE0);
	if (idx >= 0 && idx < kMaxTextures && g_texValid[idx])
	{
		glBindTexture(GL_TEXTURE_2D, g_texObjects[idx]);
		applyTexParams(idx);
	}
	else
	{
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

void C4JRender::TextureBindVertex(int idx)
{
	g_state.boundTexVertex = idx;
	glActiveTexture(GL_TEXTURE1);
	if (idx >= 0 && idx < kMaxTextures && g_texValid[idx])
	{
		glBindTexture(GL_TEXTURE_2D, g_texObjects[idx]);
		applyTexParams(idx);
	}
	else
	{
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	glActiveTexture(GL_TEXTURE0);
}

void C4JRender::TextureSetTextureLevels(int levels)
{
	g_texPendingLevels = levels;
	if (g_state.boundTex >= 0 && g_state.boundTex < kMaxTextures)
		g_texLevels[g_state.boundTex] = levels;
}

int C4JRender::TextureGetTextureLevels()
{
	if (g_state.boundTex >= 0 && g_state.boundTex < kMaxTextures)
		return g_texLevels[g_state.boundTex];
	return 1;
}

void C4JRender::TextureData(int width, int height, void* data, int level,
                             eTextureFormat format)
{
	if (g_state.boundTex < 0 || !g_texValid[g_state.boundTex]) return;

	GLenum internalFmt, pixelFmt, pixelType;
	if (format == TEXTURE_FORMAT_RxGyBzAw5551)
	{
		internalFmt = GL_RGB5_A1;
		pixelFmt    = GL_RGBA;
		pixelType   = GL_UNSIGNED_SHORT_5_5_5_1;
	}
	else
	{
		internalFmt = GL_RGBA;
		pixelFmt    = GL_RGBA;
		pixelType   = GL_UNSIGNED_BYTE;
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_texObjects[g_state.boundTex]);
	glTexImage2D(GL_TEXTURE_2D, level, (GLint)internalFmt,
	             width, height, 0, pixelFmt, pixelType, data);
}

void C4JRender::TextureDataUpdate(int xoffset, int yoffset, int width, int height,
                                   void* data, int level)
{
	if (g_state.boundTex < 0 || !g_texValid[g_state.boundTex]) return;
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_texObjects[g_state.boundTex]);
	glTexSubImage2D(GL_TEXTURE_2D, level, xoffset, yoffset, width, height,
	                GL_RGBA, GL_UNSIGNED_BYTE, data);
}

void C4JRender::TextureSetParam(int param, int value)
{
	int idx = g_state.boundTex;
	if (idx < 0 || idx >= kMaxTextures || !g_texValid[idx]) return;

	switch (param)
	{
	case GL_TEXTURE_MIN_FILTER: g_texParamMin[idx]   = value; break;
	case GL_TEXTURE_MAG_FILTER: g_texParamMag[idx]   = value; break;
	case GL_TEXTURE_WRAP_S:     g_texParamWrapS[idx] = value; break;
	case GL_TEXTURE_WRAP_T:     g_texParamWrapT[idx] = value; break;
	}
}

void C4JRender::TextureDynamicUpdateStart() {}
void C4JRender::TextureDynamicUpdateEnd()   {}

int C4JRender::LoadTextureData(const char* /*szFilename*/, D3DXIMAGE_INFO* /*pSrcInfo*/,
                                int** /*ppDataOut*/)
{
	return -1; // Not implemented – handled by platform image loading
}

int C4JRender::LoadTextureData(uint8_t* /*pbData*/, uint32_t /*dwBytes*/,
                                D3DXIMAGE_INFO* /*pSrcInfo*/, int** /*ppDataOut*/)
{
	return -1;
}

int C4JRender::SaveTextureData(const char* /*szFilename*/, D3DXIMAGE_INFO* /*pSrcInfo*/,
                                int* /*ppDataOut*/)
{
	return -1;
}

void C4JRender::TextureGetStats() {}

GLuint C4JRender::TextureGetTexture(int idx)
{
	if (idx >= 0 && idx < kMaxTextures && g_texValid[idx])
		return g_texObjects[idx];
	return 0;
}

// ---- State control ----

void C4JRender::StateSetColour(float r, float g, float b, float a)
{
	g_state.colour[0] = r;
	g_state.colour[1] = g;
	g_state.colour[2] = b;
	g_state.colour[3] = a;
}

void C4JRender::StateSetDepthMask(bool enable)
{
	g_state.depthMask = enable;
	glDepthMask(enable ? GL_TRUE : GL_FALSE);
}

void C4JRender::StateSetBlendEnable(bool enable)
{
	g_state.blendEnabled = enable;
	if (enable) glEnable(GL_BLEND);
	else        glDisable(GL_BLEND);
}

void C4JRender::StateSetBlendFunc(int src, int dst)
{
	g_state.blendSrc = src;
	g_state.blendDst = dst;
	glBlendFunc((GLenum)src, (GLenum)dst);
}

void C4JRender::StateSetBlendFactor(unsigned int colour)
{
	g_state.blendFactor = colour;
	float r = ((colour >> 24) & 0xFF) / 255.0f;
	float g = ((colour >> 16) & 0xFF) / 255.0f;
	float b = ((colour >>  8) & 0xFF) / 255.0f;
	float a = ((colour)       & 0xFF) / 255.0f;
	glBlendColor(r, g, b, a);
}

void C4JRender::StateSetAlphaFunc(int func, float param)
{
	g_state.alphaTestEnabled = true;
	g_state.alphaFunc        = func;
	g_state.alphaFuncParam   = param;
}

void C4JRender::StateSetDepthFunc(int func)
{
	g_state.depthFunc = func;
	glDepthFunc((GLenum)func);
}

void C4JRender::StateSetFaceCull(bool enable)
{
	g_state.cullEnabled = enable;
	if (enable) glEnable(GL_CULL_FACE);
	else        glDisable(GL_CULL_FACE);
}

void C4JRender::StateSetFaceCullCW(bool cw)
{
	g_state.cullCW = cw;
	glFrontFace(cw ? GL_CW : GL_CCW);
}

void C4JRender::StateSetLineWidth(float width)
{
	glLineWidth(width);
}

void C4JRender::StateSetWriteEnable(bool r, bool g, bool b, bool a)
{
	g_state.writeR = r;
	g_state.writeG = g;
	g_state.writeB = b;
	g_state.writeA = a;
	glColorMask(r ? GL_TRUE : GL_FALSE,
	             g ? GL_TRUE : GL_FALSE,
	             b ? GL_TRUE : GL_FALSE,
	             a ? GL_TRUE : GL_FALSE);
}

void C4JRender::StateSetDepthTestEnable(bool enable)
{
	g_state.depthTestEnabled = enable;
	if (enable) glEnable(GL_DEPTH_TEST);
	else        glDisable(GL_DEPTH_TEST);
}

void C4JRender::StateSetAlphaTestEnable(bool enable)
{
	g_state.alphaTestEnabled = enable;
	// Actual effect applied via uniform in fragment shader
}

void C4JRender::StateSetDepthSlopeAndBias(float slope, float bias)
{
	g_state.depthBiasSlope = slope;
	g_state.depthBiasConst = bias;
	if (slope != 0.0f || bias != 0.0f)
	{
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(slope, bias);
	}
	else
	{
		glDisable(GL_POLYGON_OFFSET_FILL);
	}
}

void C4JRender::StateSetFogEnable(bool enable)   { g_state.fogEnabled  = enable; }
void C4JRender::StateSetFogMode(int mode)         { g_state.fogMode     = mode;   }
void C4JRender::StateSetFogNearDistance(float d)  { g_state.fogNear     = d;      }
void C4JRender::StateSetFogFarDistance(float d)   { g_state.fogFar      = d;      }
void C4JRender::StateSetFogDensity(float d)        { g_state.fogDensity  = d;      }

void C4JRender::StateSetFogColour(float r, float g, float b)
{
	g_state.fogColour[0] = r;
	g_state.fogColour[1] = g;
	g_state.fogColour[2] = b;
}

void C4JRender::StateSetLightingEnable(bool enable)
{
	g_state.lightingEnabled = enable;
}

void C4JRender::StateSetVertexTextureUV(float u, float v)
{
	g_state.vertexTexU = u;
	g_state.vertexTexV = v;
}

void C4JRender::StateSetLightColour(int light, float r, float g, float b)
{
	if (light < 0 || light >= kMaxLights) return;
	g_state.lights[light].colour[0] = r;
	g_state.lights[light].colour[1] = g;
	g_state.lights[light].colour[2] = b;
}

void C4JRender::StateSetLightAmbientColour(float r, float g, float b)
{
	g_state.lightAmbient[0] = r;
	g_state.lightAmbient[1] = g;
	g_state.lightAmbient[2] = b;
}

void C4JRender::StateSetLightDirection(int light, float x, float y, float z)
{
	if (light < 0 || light >= kMaxLights) return;
	g_state.lights[light].direction[0] = x;
	g_state.lights[light].direction[1] = y;
	g_state.lights[light].direction[2] = z;
}

void C4JRender::StateSetLightEnable(int light, bool enable)
{
	if (light < 0 || light >= kMaxLights) return;
	g_state.lights[light].enabled = enable;
}

void C4JRender::StateSetViewport(eViewportType viewportType)
{
	int w = g_state.screenW;
	int h = g_state.screenH;
	int hw = w / 2, hh = h / 2;

	int x = 0, y = 0, vw = w, vh = h;
	switch (viewportType)
	{
	case VIEWPORT_TYPE_SPLIT_TOP:              x = 0;  y = hh; vw = w;  vh = hh; break;
	case VIEWPORT_TYPE_SPLIT_BOTTOM:           x = 0;  y = 0;  vw = w;  vh = hh; break;
	case VIEWPORT_TYPE_SPLIT_LEFT:             x = 0;  y = 0;  vw = hw; vh = h;  break;
	case VIEWPORT_TYPE_SPLIT_RIGHT:            x = hw; y = 0;  vw = hw; vh = h;  break;
	case VIEWPORT_TYPE_QUADRANT_TOP_LEFT:      x = 0;  y = hh; vw = hw; vh = hh; break;
	case VIEWPORT_TYPE_QUADRANT_TOP_RIGHT:     x = hw; y = hh; vw = hw; vh = hh; break;
	case VIEWPORT_TYPE_QUADRANT_BOTTOM_LEFT:   x = 0;  y = 0;  vw = hw; vh = hh; break;
	case VIEWPORT_TYPE_QUADRANT_BOTTOM_RIGHT:  x = hw; y = 0;  vw = hw; vh = hh; break;
	default:                                   break;
	}
	g_state.vpX = x; g_state.vpY = y;
	g_state.vpW = vw; g_state.vpH = vh;
	glViewport(x, y, vw, vh);
}

void C4JRender::StateSetEnableViewportClipPlanes(bool /*enable*/)
{
	// GLES3 supports 1 user clip plane via gl_ClipDistance but not commonly needed
}

void C4JRender::StateSetTexGenCol(int col, float x, float y, float z, float w,
                                   bool eyeSpace)
{
	if (col < 0 || col >= 4) return;
	g_state.texGenCols[col].plane[0] = x;
	g_state.texGenCols[col].plane[1] = y;
	g_state.texGenCols[col].plane[2] = z;
	g_state.texGenCols[col].plane[3] = w;
	g_state.texGenCols[col].eyeSpace  = eyeSpace;
}

void C4JRender::StateSetStencil(int Function, uint8_t stencilRef,
                                 uint8_t stencilFuncMask, uint8_t stencilWriteMask)
{
	g_state.stencilFunc      = Function;
	g_state.stencilRef       = stencilRef;
	g_state.stencilFuncMask  = stencilFuncMask;
	g_state.stencilWriteMask = stencilWriteMask;
	glStencilFunc((GLenum)Function, stencilRef, stencilFuncMask);
	glStencilMask(stencilWriteMask);
}

void C4JRender::StateSetForceLOD(int LOD)
{
	g_state.forceLOD = LOD;
}

// ---- Event tracking ----

void C4JRender::BeginEvent(const wchar_t* /*eventName*/) {}
void C4JRender::EndEvent()                               {}

// ---- Memory helpers ----

void* C4JRender::MemoryAllocateGPUMem(uint32_t alignment, uint32_t size)
{
	if (alignment <= 1) return malloc(size);
#if defined(_MSC_VER)
	return _aligned_malloc(size, alignment);
#else
	void* ptr = nullptr;
	posix_memalign(&ptr, alignment, size);
	return ptr;
#endif
}

void* C4JRender::MemoryAllocateCPUMem(uint32_t alignment, uint32_t size)
{
	return MemoryAllocateGPUMem(alignment, size);
}

void C4JRender::MemoryFreeGPUMem(void* data)
{
#if defined(_MSC_VER)
	_aligned_free(data);
#else
	free(data);
#endif
}

void C4JRender::MemoryFreeCPUMem(void* data)
{
	MemoryFreeGPUMem(data);
}
