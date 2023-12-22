#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform sampler2D textures[];

struct MaterialGpuData
{
	vec4 mDiffuseReflectivity;
	vec4 mAmbientReflectivity;
	vec4 mSpecularReflectivity;
	vec4 mEmissiveColor;
	vec4 mTransparentColor;
	vec4 mReflectiveColor;
	vec4 mAlbedo;

	float mOpacity;
	float mBumpScaling;
	float mShininess;
	float mShininessStrength;
	
	float mRefractionIndex;
	float mReflectivity;
	float mMetallic;
	float mSmoothness;
	
	float mSheen;
	float mThickness;
	float mRoughness;
	float mAnisotropy;
	
	vec4 mAnisotropyRotation;
	vec4 mCustomData;
	
	int mDiffuseTexIndex;
	int mSpecularTexIndex;
	int mAmbientTexIndex;
	int mEmissiveTexIndex;
	int mHeightTexIndex;
	int mNormalsTexIndex;
	int mShininessTexIndex;
	int mOpacityTexIndex;
	int mDisplacementTexIndex;
	int mReflectionTexIndex;
	int mLightmapTexIndex;
	int mExtraTexIndex;
	
	vec4 mDiffuseTexOffsetTiling;
	vec4 mSpecularTexOffsetTiling;
	vec4 mAmbientTexOffsetTiling;
	vec4 mEmissiveTexOffsetTiling;
	vec4 mHeightTexOffsetTiling;
	vec4 mNormalsTexOffsetTiling;
	vec4 mShininessTexOffsetTiling;
	vec4 mOpacityTexOffsetTiling;
	vec4 mDisplacementTexOffsetTiling;
	vec4 mReflectionTexOffsetTiling;
	vec4 mLightmapTexOffsetTiling;
	vec4 mExtraTexOffsetTiling;
};

layout(set = 1, binding = 0) buffer Material 
{
	MaterialGpuData materials[];
} matSsbo;

layout (location = 0) in PerVertexData
{
	vec3 positionWS;
	vec3 normalWS;
	vec2 texCoord;
	flat int materialIndex;
	vec3 color;
} v_in;

layout(push_constant) uniform PushConstants {
	bool mHighlightMeshlets;
	int  mVisibleMeshletIndexFrom;
	int  mVisibleMeshletIndexTo;  
} pushConstants;

layout (location = 0) out vec4 fs_out;

void main() 
{
	int matIndex = v_in.materialIndex;

	int diffuseTexIndex = matSsbo.materials[matIndex].mDiffuseTexIndex;
    vec3 color = texture(textures[diffuseTexIndex], v_in.texCoord).rgb;
	
	float ambient = 0.2;
	vec3 diffuse = matSsbo.materials[matIndex].mDiffuseReflectivity.rgb;
	vec3 toLight = normalize(vec3(-0.5, -1.5, 1.0));
	vec3 illum = vec3(ambient) + 0.75 * diffuse * max(0.0, dot(normalize(v_in.normalWS), toLight));
	vec3 color2 = color * illum;

	toLight = normalize(vec3(0.5, 0.5, 0.0));
	illum = diffuse * max(0.0, dot(normalize(v_in.normalWS), toLight));
	color2 += 0.5 * color * illum;
	
	if(pushConstants.mHighlightMeshlets) {
		color2 = mix(color2, v_in.color, 0.75);
	}

	fs_out = vec4(color2, 1.0);
}
