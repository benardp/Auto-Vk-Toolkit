#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage  : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#include "glsl_helpers.glsl"
#include "mesh.glsl"

layout (location = 0) in vec3 inPosition; 
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec3 inNormal;

layout(push_constant) uniform PushConstants {
	mat4 mModelMatrix;
	int mMaterialIndex;
} pushConstants;

layout(set = 0, binding = 1) uniform CameraTransform
{
	mat4 mViewProjMatrix;
	vec3 mCameraCenter;
} ubo;

layout(set = 1, binding = 1) buffer InstanceTransforms
{
	mat4 mat[];
} instanceMatrices;

layout(set = 2, binding = 0) buffer BoneMatrices 
{
	mat4 mat[]; // length of #bones
} boneMatrices[]; // length of #models

layout(set = 4, binding = 0) buffer MeshletsBuffer { extended_meshlet meshletsBuffer[]; } ;


layout (location = 0) out PerVertexData
{
	vec3 positionWS;
	vec3 normalWS;
	vec2 texCoord;
	flat int materialIndex;
	vec3 color;
} v_out;

void main() {
	vec4 posWS = pushConstants.mModelMatrix * vec4(inPosition.xyz, 1.0);
	v_out.positionWS = posWS.xyz;
    v_out.texCoord = inTexCoord;
	v_out.normalWS = mat3(pushConstants.mModelMatrix) * inNormal;
	v_out.materialIndex = pushConstants.mMaterialIndex;
	v_out.color = vec3(0.5, 0.5, 0.5);
    gl_Position = ubo.mViewProjMatrix * posWS;
}
