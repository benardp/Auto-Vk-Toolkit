
vec4 bone_transform(mat4 BM0, mat4 BM1, mat4 BM2, mat4 BM3, vec4 weights, vec4 positionToTransform)
{
	weights.w = 1.0 - dot(weights.xyz, vec3(1.0, 1.0, 1.0));
	vec4 tr0 = BM0 * positionToTransform;
	vec4 tr1 = BM1 * positionToTransform;
	vec4 tr2 = BM2 * positionToTransform;
	vec4 tr3 = BM3 * positionToTransform;
	return weights[0] * tr0 + weights[1] * tr1 + weights[2] * tr2 + weights[3] * tr3;
}

vec3 bone_transform(mat4 BM0, mat4 BM1, mat4 BM2, mat4 BM3, vec4 weights, vec3 normalToTransform)
{
	weights.w = 1.0 - dot(weights.xyz, vec3(1.0, 1.0, 1.0));
	vec3 tr0 = mat3(BM0) * normalToTransform;
	vec3 tr1 = mat3(BM1) * normalToTransform;
	vec3 tr2 = mat3(BM2) * normalToTransform;
	vec3 tr3 = mat3(BM3) * normalToTransform;
	return weights[0] * tr0 + weights[1] * tr1 + weights[2] * tr2 + weights[3] * tr3;
}

#define M_PI_2 1.5707963268

bool coneCull(vec3 center, float radius, vec3 cone_axis, float cone_cutoff, vec3 camera_position)
{
	vec3 ViewSpine = center - camera_position;
	//return dot(-ViewSpine, cone_axis) < 0;
	//return abs(dot(ViewSpine, cone_axis)) >= cone_cutoff * length(ViewSpine) + radius;
	//return dot(ViewSpine, cone_axis) >= sin( acos(cone_cutoff) + asin(radius/length(ViewSpine)) );

	float viewDistance = max(length(ViewSpine), 0.001);
	if(radius >= viewDistance)
		return false;

	float ViewOverture = asin(radius / viewDistance);
		
	float absNSdotVS = abs(dot(normalize(ViewSpine), cone_axis));
	float NormalOverture = cone_cutoff;

	float sumAngles = NormalOverture + ViewOverture;


	if(sumAngles + 0.001 >= M_PI_2) {
		return false;
	} else if ( absNSdotVS - 0.001 > sin(sumAngles) ) {
		return true;
	} else {
		return false;
	}

}