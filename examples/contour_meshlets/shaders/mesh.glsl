#include "cpu_gpu_shared_config.h"

//////////////////////////////////////////////////////////////////////
// Meshlet data passed from the host side:
#if !USE_REDIRECTED_GPU_DATA
struct meshlet
{
	uint    mVertices[64];
	uint8_t mIndices[372]; // 126 triangles * 3 indices
	uint8_t mVertexCount;
	uint8_t mTriangleCount;
};
#else
struct meshlet
{
	uint mDataOffset;
	uint8_t mVertexCount;
	uint8_t mTriangleCount;
};
#endif

struct extended_meshlet
{
	mat4 mTransformationMatrix;
	uint mMaterialIndex;
	uint mTexelBufferIndex;
	uint mModelIndex;
	
	bool mAnimated;

	vec3 coneAxis;
	float coneCutoff;
	float radius;

	meshlet mGeometry;
};