#include "auto_vk_toolkit.hpp"
#include "imgui.h"

#include "configure_and_compose.hpp"
#include "imgui_manager.hpp"
#include "invokee.hpp"
#include "material_image_helpers.hpp"
#include "meshlet_helpers.hpp"
#include "model.hpp"
#include "serializer.hpp"
#include "orbit_camera.hpp"
#include "quake_camera.hpp"
#include "sequential_invoker.hpp"
#include "meshoptimizer.h"
#include "interface/cmcinterface.h"

/**
 *	Please note: This example can provide the geometry data in two different formats:
 *	 - USE_REDIRECTED_GPU_DATA 0 ...
 *	 - USE_REDIRECTED_GPU_DATA 1 ...
 *	Change the mode for both, C++ and GLSL, in cpu_gpu_shared_config.h
 */
#include "../shaders/cpu_gpu_shared_config.h"
#include "vk_convenience_functions.hpp"
#include <corecrt_math_defines.h>

#define USE_CACHE 0

static constexpr size_t sNumVertices = 64;
static constexpr size_t sNumIndices = 126 * 4;
static constexpr uint32_t cConcurrentFrames = 3u;

class skinned_meshlets_app : public avk::invokee
{
	struct alignas(16) push_constants
	{
		vk::Bool32 mHighlightMeshlets;
		vk::Bool32 mCull;
		vk::Bool32 mContours;
		int32_t    mVisibleMeshletIndexFrom;
		int32_t    mVisibleMeshletIndexTo;
		int32_t    mNbInstances;
	};

	struct view_info
	{
		glm::mat4 mViewProjMatrix;
		glm::vec3 mCameraCenter;
	};

	/** Contains the necessary buffers for drawing everything */
	struct data_for_draw_call
	{
		avk::buffer mPositionsBuffer;
		avk::buffer mTexCoordsBuffer;
		avk::buffer mNormalsBuffer;
		avk::buffer mBoneIndicesBuffer;
		avk::buffer mBoneWeightsBuffer;
#if USE_REDIRECTED_GPU_DATA
		avk::buffer mIndicesDataBuffer;
#endif

		glm::mat4 mModelMatrix;

		uint32_t mMaterialIndex;
		uint32_t mModelIndex;
	};

	/** Contains the data for each draw call */
	struct loaded_data_for_draw_call
	{
		std::vector<glm::vec3> mPositions;
		std::vector<glm::vec2> mTexCoords;
		std::vector<glm::vec3> mNormals;
		std::vector<uint32_t> mIndices;
		std::vector<glm::uvec4> mBoneIndices;
		std::vector<glm::vec4> mBoneWeights;
#if USE_REDIRECTED_GPU_DATA
		std::vector<uint32_t> mIndicesData;
#endif

		glm::mat4 mModelMatrix;

		uint32_t mMaterialIndex;
		uint32_t mModelIndex;
	};

	struct additional_animated_model_data
	{
		std::vector<glm::mat4> mBoneMatricesAni;
	};

	/** Helper struct for the animations. */
	struct animated_model_data
	{
		animated_model_data() = default;
		animated_model_data(animated_model_data&&) = default;
		animated_model_data& operator=(animated_model_data&&) = default;
		~animated_model_data() = default;
		// prevent copying of the data:
		animated_model_data(const animated_model_data&) = delete;
		animated_model_data& operator=(const animated_model_data&) = delete;

		std::string mModelName;
		avk::animation_clip_data mClip;
		uint32_t mNumBoneMatrices;
		size_t mBoneMatricesBufferIndex;
		avk::animation mAnimation;

		[[nodiscard]] double start_sec() const { return mClip.mStartTicks / mClip.mTicksPerSecond; }
		[[nodiscard]] double end_sec() const { return mClip.mEndTicks / mClip.mTicksPerSecond; }
		[[nodiscard]] double duration_sec() const { return end_sec() - start_sec(); }
		[[nodiscard]] double duration_ticks() const { return mClip.mEndTicks - mClip.mStartTicks; }
	};

	/** The meshlet we upload to the gpu with its additional data. */
	struct alignas(16) meshlet
	{
		glm::mat4 mTransformationMatrix;
		uint32_t mMaterialIndex;
		uint32_t mTexelBufferIndex;
		uint32_t mModelIndex;
		
		bool mAnimated;

		glm::vec3 center;
		float radius;
		glm::vec3 coneAxis;
		float coneCutoff;

#if !USE_REDIRECTED_GPU_DATA
		avk::meshlet_gpu_data<sNumVertices, sNumIndices> mGeometry;
#else
		avk::meshlet_redirected_gpu_data mGeometry;
#endif
	};

public: // v== avk::invokee overrides which will be invoked by the framework ==v
	skinned_meshlets_app(avk::queue& aQueue)
		: mQueue{ &aQueue }
	{}

	/** Creates buffers for all the drawcalls.
	 *  Called after everything has been loaded and split into meshlets properly.
	 *  @param dataForDrawCall		The loaded data for the drawcalls.
	 *	@param drawCallsTarget		The target vector for the draw call data.
	 */
	void add_draw_calls(std::vector<loaded_data_for_draw_call>& dataForDrawCall, std::vector<data_for_draw_call>& drawCallsTarget) {
		for (auto& drawCallData : dataForDrawCall) {
			auto& drawCall = drawCallsTarget.emplace_back();
			drawCall.mModelMatrix = drawCallData.mModelMatrix;
			drawCall.mMaterialIndex = drawCallData.mMaterialIndex;

			drawCall.mPositionsBuffer = avk::context().create_buffer(avk::memory_usage::device, {},
				avk::vertex_buffer_meta::create_from_data(drawCallData.mPositions).describe_only_member(drawCallData.mPositions[0], avk::content_description::position),
				avk::storage_buffer_meta::create_from_data(drawCallData.mPositions),
				avk::uniform_texel_buffer_meta::create_from_data(drawCallData.mPositions).describe_only_member(drawCallData.mPositions[0]) // just take the vec3 as it is
			);

			drawCall.mNormalsBuffer = avk::context().create_buffer(avk::memory_usage::device, {},
				avk::vertex_buffer_meta::create_from_data(drawCallData.mNormals),
				avk::storage_buffer_meta::create_from_data(drawCallData.mNormals),
				avk::uniform_texel_buffer_meta::create_from_data(drawCallData.mNormals).describe_only_member(drawCallData.mNormals[0]) // just take the vec3 as it is
			);

			drawCall.mTexCoordsBuffer = avk::context().create_buffer(avk::memory_usage::device, {},
				avk::vertex_buffer_meta::create_from_data(drawCallData.mTexCoords),
				avk::storage_buffer_meta::create_from_data(drawCallData.mTexCoords),
				avk::uniform_texel_buffer_meta::create_from_data(drawCallData.mTexCoords).describe_only_member(drawCallData.mTexCoords[0]) // just take the vec2 as it is   
			);

#if USE_REDIRECTED_GPU_DATA
			drawCall.mIndicesDataBuffer = avk::context().create_buffer(avk::memory_usage::device, {},
				avk::vertex_buffer_meta::create_from_data(drawCallData.mIndicesData),
				avk::storage_buffer_meta::create_from_data(drawCallData.mIndicesData)
			);
#endif

			drawCall.mBoneIndicesBuffer = avk::context().create_buffer(avk::memory_usage::device, {},
				avk::vertex_buffer_meta::create_from_data(drawCallData.mBoneIndices),
				avk::storage_buffer_meta::create_from_data(drawCallData.mBoneIndices),
				avk::uniform_texel_buffer_meta::create_from_data(drawCallData.mBoneIndices).describe_only_member(drawCallData.mBoneIndices[0]) // just take the uvec4 as it is 
			);

			drawCall.mBoneWeightsBuffer = avk::context().create_buffer(avk::memory_usage::device, {},
				avk::vertex_buffer_meta::create_from_data(drawCallData.mBoneWeights),
				avk::storage_buffer_meta::create_from_data(drawCallData.mBoneWeights),
				avk::uniform_texel_buffer_meta::create_from_data(drawCallData.mBoneWeights).describe_only_member(drawCallData.mBoneWeights[0]) // just take the vec4 as it is 
			);

			avk::context().record_and_submit_with_fence({
				drawCall.mPositionsBuffer->fill(drawCallData.mPositions.data(), 0),
				drawCall.mNormalsBuffer->fill(drawCallData.mNormals.data(), 0),
				drawCall.mTexCoordsBuffer->fill(drawCallData.mTexCoords.data(), 0),
				drawCall.mBoneIndicesBuffer->fill(drawCallData.mBoneIndices.data(), 0),
				drawCall.mBoneWeightsBuffer->fill(drawCallData.mBoneWeights.data(), 0)
#if USE_REDIRECTED_GPU_DATA
				, drawCall.mIndicesDataBuffer->fill(drawCallData.mIndicesData.data(), 0)
#endif
				}, *mQueue)->wait_until_signalled();

			// add them to the texel buffers
			mPositionBuffers.push_back(avk::context().create_buffer_view(drawCall.mPositionsBuffer));
			mNormalBuffers.push_back(avk::context().create_buffer_view(drawCall.mNormalsBuffer));
			mTexCoordsBuffers.push_back(avk::context().create_buffer_view(drawCall.mTexCoordsBuffer));
#if USE_REDIRECTED_GPU_DATA
			mIndicesDataBuffers.push_back(drawCall.mIndicesDataBuffer);
#endif
			mBoneIndicesBuffers.push_back(avk::context().create_buffer_view(drawCall.mBoneIndicesBuffer));
			mBoneWeightsBuffers.push_back(avk::context().create_buffer_view(drawCall.mBoneWeightsBuffer));
		}
	}

	void initialize() override
	{
		// use helper functions to create ImGui elements
		auto surfaceCap = avk::context().physical_device().getSurfaceCapabilitiesKHR(avk::context().main_window()->surface());

		// Create a descriptor cache that helps us to conveniently create descriptor sets:
		mDescriptorCache = avk::context().create_descriptor_cache();

		glm::mat4 globalTransform = glm::mat4(1);// glm::rotate(glm::radians(180.f), glm::vec3(0.f, 1.f, 0.f))* glm::scale(glm::vec3(1.f));
		std::vector<avk::model> loadedModels;
		// Load a model from file:
		//auto model = avk::model_t::load_from_file("assets/crab.fbx", aiProcess_Triangulate);
		//loadedModels.push_back(std::move(model));

		void** dataPassBuffers = (void**)malloc(sizeof(void*) * 3);
		CMCModuleStart(dataPassBuffers);
		float* dataPassBufferFloat = (float*)dataPassBuffers[0];
		int* dataPassBufferInt = (int*)dataPassBuffers[1];
		SceneData* sceneData = CMCInternal_GetSceneData();
		//char* filepath = (char*)"assets/MESH_dragon_vrip.cmcr";
		//char* filepath = (char*) "assets/MESH_Armadillo.cmcr";
		//char* filepath = (char*)"assets/MESH_bun_zipper.cmcr";
		//char* filepath = (char*)"assets/MESH_Env_RomanBath.cmcr";
		//char* filepath = (char*)"assets/MESH_Env_SpaceStationNoDebris.cmcr";
		//char* filepath = (char*)"assets/MESH_Env_BookFantasy.cmcr";
		//char* filepath = (char*)"assets/MESH_Env_Vigilant.cmcr";
		//char* filepath = (char*)"assets/Chr_Pigman.cmcr";
		//char* filepath = (char*)"assets/Chr_Tuba.cmcr";
		char* filepath = (char*)"assets/GawainFull.cmcr";
		MeshletData* mData = CMCInternal_LoadNewMeshletDataFromDisk(filepath);
		ProxyMesh* pMesh = CMCInternal_LoadProxyMeshFromBasePath(mData, (mData->proxyMeshes + LODMETHOD_GPU_MESHSHADER_REDUCEDSPHERE), LODMETHOD_GPU_MESHSHADER_REDUCEDSPHERE, filepath);



		auto model = avk::model_t::load_from_file("assets/stanford_bunny3.obj", aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_PreTransformVertices);
		loadedModels.push_back(std::move(model));

		std::vector<avk::material_config> allMatConfigs; // <-- Gather the material config from all models to be loaded
		std::vector<loaded_data_for_draw_call> dataForDrawCall;
		std::vector<meshlet> meshletsGeometry;
		std::vector<animated_model_data> animatedModels;		

		// Crab-specific animation config: (Needs to be adapted for other models)
		const uint32_t cAnimationIndex = 0;
		const uint32_t cStartTimeTicks = 0;
		const uint32_t cEndTimeTicks   = 58;
		const uint32_t cTicksPerSecond = 34;

		// Generate the meshlets for each loaded model.
		for (size_t i = 0; i < loadedModels.size(); ++i) {
			auto curModel = std::move(loadedModels[i]);

			if (curModel->has_animations()) {

				// load the animation
				auto curClip = curModel->load_animation_clip(cAnimationIndex, cStartTimeTicks, cEndTimeTicks);
				curClip.mTicksPerSecond = cTicksPerSecond;
				auto& curEntry = animatedModels.emplace_back();
				curEntry.mModelName = curModel->path();
				curEntry.mClip = curClip;

				// get all the meshlet indices of the model
				const auto meshIndicesInOrder = curModel->select_all_meshes();

				curEntry.mNumBoneMatrices = curModel->num_bone_matrices(meshIndicesInOrder);

				// Store offset into the vector of buffers that store the bone matrices
				curEntry.mBoneMatricesBufferIndex = i;

				auto distinctMaterials = curModel->distinct_material_configs();
				const auto matOffset = allMatConfigs.size();
				// add all the materials of the model
				for (auto& pair : distinctMaterials) {
					allMatConfigs.push_back(pair.first);
				}

				// prepare the animation for the current entry
				curEntry.mAnimation = curModel->prepare_animation(curEntry.mClip.mAnimationIndex, meshIndicesInOrder);

				// Generate meshlets for each submesh of the current loaded model. Load all it's data into the drawcall for later use.
				for (size_t mpos = 0; mpos < meshIndicesInOrder.size(); mpos++) {
					auto meshIndex = meshIndicesInOrder[mpos];
					std::string meshname = curModel->name_of_mesh(mpos);

					auto texelBufferIndex = dataForDrawCall.size();
					auto& drawCallData = dataForDrawCall.emplace_back();

					drawCallData.mMaterialIndex = static_cast<int32_t>(matOffset);
					drawCallData.mModelMatrix = globalTransform;
					drawCallData.mModelIndex = static_cast<uint32_t>(curEntry.mBoneMatricesBufferIndex);
					// Find and assign the correct material (in the ~"global" allMatConfigs vector!)
					for (auto pair : distinctMaterials) {
						if (std::end(pair.second) != std::find(std::begin(pair.second), std::end(pair.second), meshIndex)) {
							break;
						}

						drawCallData.mMaterialIndex++;
					}

					auto selection = avk::make_model_references_and_mesh_indices_selection(curModel, meshIndex);
					std::vector<avk::mesh_index_t> meshIndices;
					meshIndices.push_back(meshIndex);
					// Build meshlets:
					std::tie(drawCallData.mPositions, drawCallData.mIndices) = avk::get_vertices_and_indices(selection);
					drawCallData.mNormals = avk::get_normals(selection);
					drawCallData.mTexCoords = avk::get_2d_texture_coordinates(selection, 0);
					// Get bone indices and weights too
					drawCallData.mBoneIndices = avk::get_bone_indices_for_single_target_buffer(selection, meshIndicesInOrder);
					drawCallData.mBoneWeights = avk::get_bone_weights(selection);

					// create selection for the meshlets
					auto meshletSelection = avk::make_models_and_mesh_indices_selection(curModel, meshIndex);

					auto cpuMeshlets = avk::divide_into_meshlets(meshletSelection);
#if !USE_REDIRECTED_GPU_DATA
#if USE_CACHE
					avk::serializer serializer("direct_meshlets-" + meshname + "-" + std::to_string(mpos) + ".cache");
					auto [gpuMeshlets, _] = avk::convert_for_gpu_usage_cached<avk::meshlet_gpu_data<sNumVertices, sNumIndices>>(serializer, cpuMeshlets);
#else
					auto [gpuMeshlets, _] = avk::convert_for_gpu_usage<avk::meshlet_gpu_data<sNumVertices, sNumIndices>, sNumVertices, sNumIndices>(cpuMeshlets);
#endif
#else
#if USE_CACHE
					avk::serializer serializer("redirected_meshlets-" + meshname + "-" + std::to_string(mpos) + ".cache");
					auto [gpuMeshlets, gpuIndicesData] = avk::convert_for_gpu_usage_cached<avk::meshlet_redirected_gpu_data, sNumVertices, sNumIndices>(serializer, cpuMeshlets);
#else
					auto [gpuMeshlets, gpuIndicesData] = avk::convert_for_gpu_usage<avk::meshlet_redirected_gpu_data, sNumVertices, sNumIndices>(cpuMeshlets);
#endif
					drawCallData.mIndicesData = std::move(gpuIndicesData.value());
#endif

					// fill our own meshlets with the loaded/generated data
					for (size_t mshltidx = 0; mshltidx < gpuMeshlets.size(); ++mshltidx) {
						auto& genMeshlet = gpuMeshlets[mshltidx];

						auto& ml = meshletsGeometry.emplace_back(meshlet{});

#pragma region start to assemble meshlet struct
						ml.mTransformationMatrix = drawCallData.mModelMatrix;
						ml.mMaterialIndex = drawCallData.mMaterialIndex;
						ml.mTexelBufferIndex = static_cast<uint32_t>(texelBufferIndex);
						ml.mModelIndex = static_cast<uint32_t>(curEntry.mBoneMatricesBufferIndex);

						ml.mGeometry = genMeshlet;
						ml.mAnimated = true;
#pragma endregion 
					}
				}
			}
			else // without animation
			{ 

				// get all the meshlet indices of the model
				const auto meshIndicesInOrder = curModel->select_all_meshes();

				auto distinctMaterials = curModel->distinct_material_configs();
				const auto matOffset = allMatConfigs.size();
				// add all the materials of the model
				for (auto& pair : distinctMaterials) {
					allMatConfigs.push_back(pair.first);
				}

				// Generate meshlets for each submesh of the current loaded model. Load all it's data into the draw call for later use.
				for (size_t mpos = 0; mpos < meshIndicesInOrder.size(); mpos++) {
					auto meshIndex = meshIndicesInOrder[mpos];
					std::string meshname = curModel->name_of_mesh(mpos);

					std::map<avk::model_t::Edge, avk::model_t::Neighbors, avk::model_t::CompareEdges> adjacency;
					std::map<avk::model_t::Edge, bool, avk::model_t::CompareEdges> visitedEdges;
					curModel->findAdjacencies(meshIndex, adjacency);
					LOG_INFO(std::format("nb edges: {}", adjacency.size()));

					auto texelBufferIndex = dataForDrawCall.size();
					auto& drawCallData = dataForDrawCall.emplace_back();

					drawCallData.mMaterialIndex = static_cast<int32_t>(matOffset);
					drawCallData.mModelMatrix = globalTransform * curModel->transformation_matrix_for_mesh(meshIndex);
					// Find and assign the correct material in the allMatConfigs vector
					for (auto pair : distinctMaterials) {
						if (std::end(pair.second) != std::ranges::find(pair.second, meshIndex)) {
							break;
						}
						drawCallData.mMaterialIndex++;
					}

					std::vector<avk::meshlet> cpuMeshlets;

					if (mBuildMeshlets) {

						auto selection = avk::make_model_references_and_mesh_indices_selection(curModel, meshIndex);
						// Build meshlets:
						std::tie(drawCallData.mPositions, drawCallData.mIndices) = avk::get_vertices_and_indices(selection);
						for (const auto&p : drawCallData.mPositions) {
							mSceneBBox.extend(vec3(p.x, p.y, p.z));
						}
						drawCallData.mNormals = avk::get_normals(selection);
						drawCallData.mTexCoords = avk::get_2d_texture_coordinates(selection, 0);
						// Empty bone indices and weights too
						drawCallData.mBoneIndices.resize(drawCallData.mPositions.size());
						drawCallData.mBoneWeights.resize(drawCallData.mPositions.size());

						// create selection for the meshlets
						auto meshletSelection = avk::make_models_and_mesh_indices_selection(curModel, meshIndex);

						auto meshoptimizer_clustering = [&](const std::vector<glm::vec3>& tVertices, const std::vector<uint32_t>& aIndices,
							const avk::model_t& aModel, std::optional<avk::mesh_index_t> aMeshIndex,
							uint32_t aMaxVertices, uint32_t aMaxIndices) {

								// definitions
								size_t max_triangles = aMaxIndices / 3;
								const float cone_weight = 0.75f;

								// get the maximum number of meshlets that could be generated
								size_t max_meshlets = meshopt_buildMeshletsBound(aIndices.size(), aMaxVertices, max_triangles);
								std::vector<meshopt_Meshlet> meshlets(max_meshlets);
								std::vector<unsigned int> meshlet_vertices(max_meshlets * aMaxVertices);
								std::vector<unsigned char> meshlet_triangles(max_meshlets * max_triangles * 3);

								// let meshoptimizer build the meshlets for us
								size_t meshlet_count = meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
									aIndices.data(), aIndices.size(), &tVertices[0].x, tVertices.size(), sizeof(glm::vec3),
									aMaxVertices, max_triangles, cone_weight);

								// copy the data over to Auto-Vk-Toolkit's meshlet structure
								std::vector<avk::meshlet> generatedMeshlets(meshlet_count);
								generatedMeshlets.resize(meshlet_count);
								for (int k = 0; k < meshlet_count; k++) {
									auto& m = meshlets[k];
									auto& gm = generatedMeshlets[k];

									// compute bounds
									meshopt_Bounds bounds = meshopt_computeMeshletBounds(&(meshlet_vertices[m.vertex_offset]), &(meshlet_triangles[m.triangle_offset]), m.triangle_count, &tVertices[0].x, tVertices.size(), sizeof(glm::vec3));
									gm.center = glm::vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
									gm.radius = bounds.radius;
									gm.coneAxis = glm::vec3(bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2]);
									gm.coneCutoff = bounds.cone_cutoff > 0 ? acos(bounds.cone_cutoff) : M_PI;

									std::map<avk::model_t::Edge, avk::model_t::Neighbors, avk::model_t::CompareEdges> indexMap;
									std::vector< std::pair<avk::model_t::Edge, unsigned int> > boundaryEdges;
									for (unsigned int i = 0; i < m.triangle_count; ++i) {
										glm::ivec3 face;
										for (unsigned int f = 0; f < 3; ++f) {
											face[f] = meshlet_triangles[m.triangle_offset + i * 3 + f];
										}
										avk::model_t::Edge e1(face[0], face[1]);
										avk::model_t::Edge e2(face[1], face[2]);
										avk::model_t::Edge e3(face[2], face[0]);

										indexMap[e1].AddNeigbor(i);
										indexMap[e2].AddNeigbor(i);
										indexMap[e3].AddNeigbor(i);
									}

									for (const auto& e : indexMap) {
										// meshlet boundary edges
										if (e.second.n1 == (unsigned int)-1) {
											boundaryEdges.push_back(std::make_pair(e.first, e.second.n2));
											continue;
										}
										if (e.second.n2 == (unsigned int)-1) {
											boundaryEdges.push_back(std::make_pair(e.first, e.second.n1));
											continue;
										}
										uint32_t v_a = e.first.a;
										uint32_t v_b = e.first.b;
										uint32_t v_c, v_d;
										for (unsigned int k = 0; k < 3; k++) {
											uint32_t otherVertexIndex = meshlet_triangles[m.triangle_offset + 3 * e.second.n1 + k];
											if (otherVertexIndex != v_a && otherVertexIndex != v_b) {
												v_c = otherVertexIndex;
											}
											otherVertexIndex = meshlet_triangles[m.triangle_offset + 3 * e.second.n2 + k];
											if (otherVertexIndex != v_a && otherVertexIndex != v_b) {
												v_d = otherVertexIndex;
											}
										}
										gm.mIndices.push_back(v_a);
										gm.mIndices.push_back(v_b);
										gm.mIndices.push_back(v_c);
										gm.mIndices.push_back(v_d);

									}

									gm.mVertices.resize(m.vertex_count);
									std::ranges::copy(meshlet_vertices.begin() + m.vertex_offset,
										meshlet_vertices.begin() + m.vertex_offset + m.vertex_count,
										gm.mVertices.begin());

									for (const auto& pair : boundaryEdges) {
										avk::model_t::Edge mesh_edge = pair.first;
										uint32_t v_a = meshlet_vertices[m.vertex_offset + mesh_edge.a];
										uint32_t v_b = meshlet_vertices[m.vertex_offset + mesh_edge.b];
										uint32_t v_c = -1, lv_c;
										for (unsigned int k = 0; k < 3; k++) {
											lv_c = meshlet_triangles[m.triangle_offset + 3 * pair.second + k];
											v_c = meshlet_vertices[m.vertex_offset + lv_c];
											if (v_c != v_a && v_c != v_b) {
												break;
											}
										}
										assert(v_c != -1);
										avk::model_t::Edge e(v_a, v_b);
										assert(adjacency.find(e) != adjacency.end());
										if (visitedEdges.find(e) != visitedEdges.end()) {
											continue;
										}
										visitedEdges[e] = true;
										avk::model_t::Neighbors neighbours = adjacency[e];
										uint32_t n = -1;
										for (unsigned int k = 0; k < 3; k++) {
											if (neighbours.n1 != -1 && aIndices[neighbours.n1 * 3 + k] == v_c) {
												n = neighbours.n2;
												break;
											}
											if (neighbours.n2 != -1 && aIndices[neighbours.n2 * 3 + k] == v_c) {
												n = neighbours.n1;
												break;
											}
										}
										assert(n != -1);
										uint32_t v_d = -1;
										for (unsigned int k = 0; k < 3; k++) {
											v_d = aIndices[n * 3 + k];
											if (v_d != v_a && v_d != v_b) {
												break;
											}
										}
										assert(v_d != -1);
										gm.mIndices.push_back(mesh_edge.a);
										gm.mIndices.push_back(mesh_edge.b);
										gm.mIndices.push_back(lv_c);
										gm.mIndices.push_back(gm.mVertices.size());
										gm.mVertices.push_back(v_d);
									}

									gm.mIndexCount = gm.mIndices.size();
									gm.mVertexCount = gm.mVertices.size();

									//float radius = 0.0;
									//glm::vec3 center(0);
									//for (unsigned int v : gm.mVertices) {
									//	center += tVertices[v];
									//}
									//center /= gm.mVertexCount;

									//for (unsigned int v : gm.mVertices) {
									//	radius = std::max(radius, glm::length(tVertices[v] - center));
									//}

									//gm.center = center;
									//gm.radius = radius;

									assert(gm.mVertexCount <= 128);


									/*
									gm.mIndexCount = m.triangle_count * 3;
									gm.mVertexCount = m.vertex_count;
									gm.mVertices.resize(m.vertex_count);
									gm.mIndices.resize(gm.mIndexCount);
									std::ranges::copy(meshlet_vertices.begin() + m.vertex_offset,
										meshlet_vertices.begin() + m.vertex_offset + m.vertex_count,
										gm.mVertices.begin());
									std::ranges::copy(meshlet_triangles.begin() + m.triangle_offset,
										meshlet_triangles.begin() + m.triangle_offset + gm.mIndexCount,
										gm.mIndices.begin()); */
								}

								return generatedMeshlets;
							};

						/* auto meshcontours_LOD = [&](const std::vector<glm::vec3>& tVertices, const std::vector<uint32_t>& aIndices,
							const avk::model_t& aModel, std::optional<avk::mesh_index_t> aMeshIndex,
							uint32_t aMaxVertices, uint32_t aMaxIndices) {

								ComputerProfilingData computerInria;
								computerInria.computerID = 0;
								computerInria.CPU_CostPerEdgeTest = 3.96827596; // To redo
								computerInria.CPU_CostPerPatchTest = 5.408779793;
								computerInria.GPU_CostPerEdgeTest = 0.33799174979074; // in ns
								computerInria.GPU_CostPerPatchTest = 0.440296657093; // in ns, occup method
								computerInria.GPU_PatchOccupancy = 16;
								computerInria.GPU_EdgeOccupancy = 16;

								MeshletData mData;
								mData.nbVertices = tVertices.size();
								mData.nbFaces = aIndices.size() / 3;
								mData.nbEdges = adjacency.size();
								mData.nbBones = 0;

								mData.restPosition = (float*) &(tVertices[0][0]);
								mData.topoFaceVertex = (unsigned int *)aIndices.data();
								mData.cdVertexCoord = (float*)&(tVertices[0][0]);

								// Construct winged edge
								int* topoWingedEdgeVVVVFF = (int*)malloc(sizeof(int) * 6 * mData.nbEdges);
								int* topoWingedEdgeVVVV = (int*)malloc(sizeof(int) * 4 * mData.nbEdges);
								int* topoWingedEdgeVVFF = (int*)malloc(sizeof(int) * 4 * mData.nbEdges);
								int* topoWingedEdgeVVVVI = (int*)malloc(sizeof(int) * 5 * mData.nbEdges);
								size_t i = 0;
								for (const auto& [edge, neighboor] : adjacency) {
									topoWingedEdgeVVVVFF[6 * i] = edge.a;
									topoWingedEdgeVVVVFF[6 * i + 1] = edge.b;

									topoWingedEdgeVVVV[4 * i] = edge.a;
									topoWingedEdgeVVVV[4 * i + 1] = edge.b;

									topoWingedEdgeVVVVI[5 * i] = edge.a;
									topoWingedEdgeVVVVI[5 * i + 1] = edge.b;

									topoWingedEdgeVVFF[4 * i] = edge.a;
									topoWingedEdgeVVFF[4 * i + 1] = edge.a;

									for (int j = 0; j < 2; ++j) {
										int faceIndex = j == 0 ? neighboor.n1 : neighboor.n2;
										int vertexIndex = -1;
										int flipIndicator = 1;
										if (faceIndex >= 0) {
											vertexIndex = aIndices[3 * faceIndex];
											if (vertexIndex == edge.a || vertexIndex == edge.b) {
												vertexIndex = aIndices[3 * faceIndex + 1];
												if (vertexIndex == edge.a || vertexIndex == edge.b)
													vertexIndex = aIndices[3 * faceIndex + 2];
											}

											// Activate the flip indicator if the vertex order is reversed
											//  1 if the vertex order is ABD
											// -1 if the vertex order is BAD
											int vertexIndexA = edge.a;
											int vertexIndexB = edge.b;
											if ((aIndices[3 * faceIndex + 0] == vertexIndexB && aIndices[3 * faceIndex + 1] == vertexIndexA) ||
												(aIndices[3 * faceIndex + 1] == vertexIndexB && aIndices[3 * faceIndex + 2] == vertexIndexA) ||
												(aIndices[3 * faceIndex + 2] == vertexIndexB && aIndices[3 * faceIndex + 0] == vertexIndexA))
												flipIndicator = -1;
										}

										topoWingedEdgeVVVVFF[6 * i + 2 + j] = vertexIndex;
										topoWingedEdgeVVVVFF[6 * i + 4 + j] = faceIndex;

										topoWingedEdgeVVVV[4 * i + 2 + j] = vertexIndex;

										topoWingedEdgeVVVVI[5 * i + 2 + j] = vertexIndex;
										topoWingedEdgeVVVVI[5 * i + 4] = flipIndicator;

										topoWingedEdgeVVFF[4 * i + 2 + j] = faceIndex;
									}
									++i;
								}
								mData.topoWingedEdgeVVVVFF = topoWingedEdgeVVVVFF;
								mData.topoWingedEdgeVVVV = topoWingedEdgeVVVV;
								mData.topoWingedEdgeVVVVI = topoWingedEdgeVVVVI;
								mData.topoWingedEdgeVVFF = topoWingedEdgeVVFF;

								ProxyMesh pMesh;
								pMesh.methodID = LODMETHOD_PATCHFUSION_GPU_MEANSPHEREPLUS;
								pMesh.nbFreeEdges = 0;
								pMesh.nbPositiveEdges = 0;
								pMesh.freeEdgesIDs = (int*)malloc(sizeof(int) * mData.nbEdges);
								pMesh.positiveEdgesIDTypes = (int*)malloc(sizeof(int) * 2 * mData.nbEdges);

								SetComputerProfilingData(computerInria);

								ProxyMesh* pMesh2 = LODMake_PatchFusion(&mData, &pMesh, EEM_CPP_LODEXTRACT, BVMETHOD_MEAN_SPHERE_PLUS, NDFMETHOD_NORMAL_CONE);

								std::vector<avk::meshlet> generatedMeshlets(pMesh2->nbPatches);


								return generatedMeshlets;

							};*/

						cpuMeshlets = avk::divide_into_meshlets(meshletSelection, meshoptimizer_clustering, true, 32, 16 * 4 * 3);
					}
					else {

						for (size_t k = 0; k < mData->nbVertices; ++k) {
							mSceneBBox.extend(vec3(mData->restPosition[3 * k], mData->restPosition[3 * k + 1], mData->restPosition[3 * k + 2]));
						}

						drawCallData.mNormals.resize(mData->nbVertices, glm::vec3(0));
						for (size_t f = 0; f < mData->nbFaces; ++f) {
							glm::vec3 n = glm::vec3(mData->restNormals[3 * f], mData->restNormals[3 * f + 1], mData->restNormals[3 * f + 2]);
							drawCallData.mNormals[mData->topoFaceVertex[3 * f]] += n;
							drawCallData.mNormals[mData->topoFaceVertex[3 * f + 1]] += n;
							drawCallData.mNormals[mData->topoFaceVertex[3 * f + 2]] += n;
						}

						drawCallData.mPositions.resize(mData->nbVertices);
						for (size_t k = 0; k < mData->nbVertices; ++k) {
							/*vec3 p = (vec3(mData->restPosition[3 * k], mData->restPosition[3 * k + 1], mData->restPosition[3 * k + 2]) - mSceneBBox.center()) / mSceneBBox.diagonal().norm();
							drawCallData.mPositions[k] = glm::vec3(p.x(), p.y(), p.z());*/
							drawCallData.mPositions[k] = glm::vec3(mData->restPosition[3 * k], mData->restPosition[3 * k + 1], mData->restPosition[3 * k + 2]);

							drawCallData.mNormals[k] = glm::normalize(drawCallData.mNormals[k]);
						}
						drawCallData.mIndices = std::move(std::vector<uint32_t>(mData->topoFaceVertex, mData->topoFaceVertex + mData->nbFaces * 3));
						drawCallData.mTexCoords.resize(mData->nbVertices);
						drawCallData.mBoneIndices.resize(mData->nbVertices);
						drawCallData.mBoneWeights.resize(mData->nbVertices);

						cpuMeshlets.resize(pMesh->nbPatches);
						std::vector<avk::meshlet> extraMeshlets;
						curModel.enable_shared_ownership();
						for (size_t k = 0; k < pMesh->nbPatches; ++k) {
							avk::meshlet* cpuMeshlet = &cpuMeshlets[k];
							auto& patch = pMesh->patches[k];

							int ctrlIDA = patch.patchBasisVerticesID[0];
							int ctrlIDB = patch.patchBasisVerticesID[1];
							int ctrlIDC = patch.patchBasisVerticesID[2];
							vec3 patchBasisPosA = vec3(mData->restPosition[3 * ctrlIDA], mData->restPosition[3 * ctrlIDA + 1], mData->restPosition[3 * ctrlIDA + 2]);
							vec3 patchBasisPosB = vec3(mData->restPosition[3 * ctrlIDB], mData->restPosition[3 * ctrlIDB + 1], mData->restPosition[3 * ctrlIDB + 2]);
							vec3 patchBasisPosC = vec3(mData->restPosition[3 * ctrlIDC], mData->restPosition[3 * ctrlIDC + 1], mData->restPosition[3 * ctrlIDC + 2]);
							vec3 patchBasisX = (patchBasisPosB - patchBasisPosA).normalized();
							vec3 patchBasisY = (patchBasisPosC - patchBasisPosA);
							patchBasisY = (patchBasisY - patchBasisX * patchBasisX.dot(patchBasisY)).normalized();
							vec3 patchBasisZ = patchBasisY.cross(patchBasisX).normalized();
							vec3 sphereOrigin = patchBasisPosA + patch.boundingSphereOriginInPatchBasis.x() * patchBasisX + patch.boundingSphereOriginInPatchBasis.y() * patchBasisY + patch.boundingSphereOriginInPatchBasis.z() * patchBasisZ;
							vec3 normal = patch.ndfConeDirectionInPatchBasis.x() * patchBasisX + patch.ndfConeDirectionInPatchBasis.y() * patchBasisY + patch.ndfConeDirectionInPatchBasis.z() * patchBasisZ;

							Eigen::AlignedBox3d bbox;
							vec3 meanNormal;
							meanNormal.setZero();
							double cosAngle = 1;

							std::map<int, unsigned int> indicesMap;
							std::vector<vec3> positions;
							std::map<int, vec3> normals;
							for (size_t i = 0; i < patch.nbFreeEdges; ++i) {
								std::array<vec3, 4> vertices;
								for (size_t j = 0; j < 4; ++j) {
									int idx = patch.patchPRTOTopoBufferVVVV[4 * i + j];
									assert(idx < mData->nbVertices);
									if (idx == -1) {
										idx = 0; // boundary
									}
									vec3 p(mData->restPosition[3 * idx], mData->restPosition[3 * idx + 1], mData->restPosition[3 * idx + 2]);
									if (indicesMap.find(idx) == indicesMap.end()) {
										indicesMap[idx] = cpuMeshlet->mVertices.size();
										cpuMeshlet->mVertices.push_back(idx);
										positions.push_back(p);
									}
									cpuMeshlet->mIndices.push_back(indicesMap[idx]);
									vertices[j] = p;
								}
								bbox.extend(0.5 * (vertices[0] + vertices[1]));
								int f1 = mData->topoEdgeFace[2 * patch.freeEdgesIDs[i]];
								int f2 = mData->topoEdgeFace[2 * patch.freeEdgesIDs[i] + 1];
								vec3 n1(mData->restNormals[3 * f1], mData->restNormals[3 * f1 + 1], mData->restNormals[3 * f1 + 2]);
								vec3 n2(mData->restNormals[3 * f2], mData->restNormals[3 * f2 + 1], mData->restNormals[3 * f2 + 2]);
								normals[f1] = n1;
								normals[f2] = n2;
								meanNormal = meanNormal + n1 + n2;
								if (cpuMeshlet->mVertices.size() > 64 || cpuMeshlet->mIndices.size() / 4 == 126) {
									//LOG_WARNING(std::format("Patch too big {} vertices, {} faces", cpuMeshlet->mVertices.size(), cpuMeshlet->mIndices.size() / 4));
									cpuMeshlet->mVertexCount = cpuMeshlet->mVertices.size();
									cpuMeshlet->mIndexCount = cpuMeshlet->mIndices.size();

									cpuMeshlet->center = glm::vec3(bbox.center().x(), bbox.center().y(), bbox.center().z());
									cpuMeshlet->radius = 0;
									for (auto& p : positions) {
										cpuMeshlet->radius = std::max(cpuMeshlet->radius, float((p - bbox.center()).norm()));
									}
									meanNormal.normalize();
									for (auto& n : normals) {
										cosAngle = std::min(cosAngle, n.second.dot(meanNormal));
									}
									cpuMeshlet->coneAxis = glm::vec3(meanNormal.x(), meanNormal.y(), meanNormal.z());
									cpuMeshlet->coneCutoff = cosAngle > 0 ? std::acos(cosAngle) : M_PI;

									/*cpuMeshlet->center = glm::vec3(sphereOrigin.x(), sphereOrigin.y(), sphereOrigin.z());
									cpuMeshlet->radius = patch.boundingSphereRadius;
									cpuMeshlet->coneAxis = glm::vec3(normal.x(), normal.y(), normal.z());
									cpuMeshlet->coneCutoff = patch.ndfConeAngleRadians;*/
									//break;
									cpuMeshlet = &extraMeshlets.emplace_back();
									indicesMap.clear();
									bbox.setEmpty();
									meanNormal.setZero();
									normals.clear();
									positions.clear();
									cosAngle = 1;
								}
							}

							cpuMeshlet->mVertexCount = cpuMeshlet->mVertices.size();
							cpuMeshlet->mIndexCount = cpuMeshlet->mIndices.size();

							cpuMeshlet->center = glm::vec3(bbox.center().x(), bbox.center().y(), bbox.center().z());
							cpuMeshlet->radius = 0;
							for (auto& p : positions) {
								cpuMeshlet->radius = std::max(cpuMeshlet->radius, float((p - bbox.center()).norm()));
							}
							meanNormal.normalize();
							for (auto& n : normals) {
								cosAngle = std::min(cosAngle, n.second.dot(meanNormal));
							}
							cpuMeshlet->coneAxis = glm::vec3(meanNormal.x(), meanNormal.y(), meanNormal.z());
							cpuMeshlet->coneCutoff = cosAngle > 0 ? std::acos(cosAngle) : M_PI;

							cpuMeshlet->center = glm::vec3(sphereOrigin.x(), sphereOrigin.y(), sphereOrigin.z());
							cpuMeshlet->radius = patch.boundingSphereRadius;
							cpuMeshlet->coneAxis = glm::vec3(normal.x(), normal.y(), normal.z());
							cpuMeshlet->coneCutoff = patch.ndfConeAngleRadians;
						}

						avk::meshlet* cpuMeshlet;
						if(pMesh->nbFreeEdges > 0) cpuMeshlet = &extraMeshlets.emplace_back();
						std::map<int, unsigned int> indicesMap;
						for (size_t k = 0; k < pMesh->nbFreeEdges; k++) {
							int eId = pMesh->freeEdgesIDs[k];
							for (size_t j = 0; j < 4; ++j) {
								int idx = mData->topoWingedEdgeVVVV[4 * eId + j];
								if (idx == -1) {
									idx = 0; // boundary
								}
								if (indicesMap.find(idx) == indicesMap.end()) {
									indicesMap[idx] = cpuMeshlet->mVertices.size();
									cpuMeshlet->mVertices.push_back(idx);
								}
								cpuMeshlet->mIndices.push_back(indicesMap[idx]);
							}
							if (cpuMeshlet->mVertices.size() >= 60 || cpuMeshlet->mIndices.size() / 4 == 126) {
								//LOG_WARNING(std::format("Patch too big {} vertices, {} faces", cpuMeshlet->mVertices.size(), cpuMeshlet->mIndices.size() / 4));
								cpuMeshlet->mVertexCount = cpuMeshlet->mVertices.size();
								cpuMeshlet->mIndexCount = cpuMeshlet->mIndices.size();
								cpuMeshlet->coneCutoff = 3.14;
								cpuMeshlet->radius = 0;

								cpuMeshlet = &extraMeshlets.emplace_back();
								indicesMap.clear();
							}
						}
						if (extraMeshlets.size() > 0) {
							cpuMeshlet->mVertexCount = cpuMeshlet->mVertices.size();
							cpuMeshlet->mIndexCount = cpuMeshlet->mIndices.size();

							cpuMeshlet->coneCutoff = 3.14;
							cpuMeshlet->radius = 0;
							cpuMeshlets.insert(cpuMeshlets.end(), std::make_move_iterator(extraMeshlets.begin()), std::make_move_iterator(extraMeshlets.end()));
						}
					}

#if !USE_REDIRECTED_GPU_DATA
#if USE_CACHE
					avk::serializer serializer("direct_meshlets-" + meshname + "-" + std::to_string(mpos) + ".cache");
					auto [gpuMeshlets, _] = avk::convert_for_gpu_usage_cached<avk::meshlet_gpu_data<sNumVertices, sNumIndices>>(serializer, cpuMeshlets);
#else
					auto [gpuMeshlets, _] = avk::convert_for_gpu_usage<avk::meshlet_gpu_data<sNumVertices, sNumIndices>, sNumVertices, sNumIndices>(cpuMeshlets);
#endif
#else
#if USE_CACHE
					avk::serializer serializer("redirected_meshlets-" + meshname + "-" + std::to_string(mpos) + ".cache");
					auto [gpuMeshlets, gpuIndicesData] = avk::convert_for_gpu_usage_cached<avk::meshlet_redirected_gpu_data, sNumVertices, sNumIndices>(serializer, cpuMeshlets);
#else
					auto [gpuMeshlets, gpuIndicesData] = avk::convert_for_gpu_usage<avk::meshlet_redirected_gpu_data, sNumVertices, sNumIndices>(cpuMeshlets);
#endif
					drawCallData.mIndicesData = std::move(gpuIndicesData.value());
#endif

					// fill our own meshlets with the loaded/generated data
					for (size_t mshltidx = 0; mshltidx < gpuMeshlets.size(); ++mshltidx) {
						auto& genMeshlet = gpuMeshlets[mshltidx];
						auto& cpuMeshlet = cpuMeshlets[mshltidx];

						auto& ml = meshletsGeometry.emplace_back(meshlet{});

#pragma region start to assemble meshlet struct
						ml.mTransformationMatrix = drawCallData.mModelMatrix;
						ml.mMaterialIndex = drawCallData.mMaterialIndex;
						ml.mTexelBufferIndex = static_cast<uint32_t>(texelBufferIndex);

						ml.mGeometry = genMeshlet;
						assert(cpuMeshlet.mIndexCount % 4 == 0);
						ml.mGeometry.mPrimitiveCount = cpuMeshlet.mIndexCount / 4;
						assert(ml.mGeometry.mPrimitiveCount <= 126);
						assert(ml.mGeometry.mVertexCount <= 64);
						ml.mAnimated = false;

						ml.center = cpuMeshlet.center;
						ml.radius = cpuMeshlet.radius;
						ml.coneAxis = cpuMeshlet.coneAxis;
						ml.coneCutoff = cpuMeshlet.coneCutoff;
#pragma endregion 
					}
				}
			}
		} // for (size_t i = 0; i < loadedModels.size(); ++i)

		// create buffers for animation data
		for (size_t i = 0; i < animatedModels.size(); ++i) {
			auto& animModel = mAnimatedModels.emplace_back(std::move(animatedModels[i]), additional_animated_model_data{});

			// buffers for the animated bone matrices, will be populated before rendering
			std::get<additional_animated_model_data>(animModel).mBoneMatricesAni.resize(std::get<animated_model_data>(animModel).mNumBoneMatrices);
			for (size_t cfi = 0; cfi < cConcurrentFrames; ++cfi) {
				mBoneMatricesBuffersAni[cfi].push_back(avk::context().create_buffer(
					avk::memory_usage::host_coherent, {},
					avk::storage_buffer_meta::create_from_data(std::get<additional_animated_model_data>(animModel).mBoneMatricesAni)
				));
			}
		}
		std::vector<glm::mat4> identity;
		identity.push_back(glm::mat4());
		for (size_t i = 0; i < loadedModels.size() - animatedModels.size(); ++i) {
			for (size_t cfi = 0; cfi < cConcurrentFrames; ++cfi) {
				mBoneMatricesBuffersAni[cfi].push_back(avk::context().create_buffer(
					avk::memory_usage::host_coherent, {},
					avk::storage_buffer_meta::create_from_data(identity)
				));
			}
		}

		mInstanceMatrices.resize(num_instances2);
		for (size_t cfi = 0; cfi < cConcurrentFrames; ++cfi) {
			mInstanceMatricesBuffer[cfi] = avk::context().create_buffer(
				avk::memory_usage::host_coherent, {},
				avk::storage_buffer_meta::create_from_data(mInstanceMatrices)
			);
		}

		// create all the buffers for our drawcall data
		add_draw_calls(dataForDrawCall, mDrawCalls);

		// Put the meshlets that we have gathered into a buffer:
		mMeshletsBuffer = avk::context().create_buffer(
			avk::memory_usage::device, {},
			avk::storage_buffer_meta::create_from_data(meshletsGeometry)
		);
		mNumMeshlets = static_cast<uint32_t>(meshletsGeometry.size());
		mShowMeshletsTo = static_cast<int>(mNumMeshlets);

		// For all the different materials, transfer them in structs which are well
		// suited for GPU-usage (proper alignment, and containing only the relevant data),
		// also load all the referenced images from file and provide access to them
		// via samplers; It all happens in `ak::convert_for_gpu_usage`:
		auto [gpuMaterials, imageSamplers, matCommands] = avk::convert_for_gpu_usage<avk::material_gpu_data>(
			allMatConfigs, false, false,
			avk::image_usage::general_texture,
			avk::filter_mode::trilinear
		);

		avk::context().record_and_submit_with_fence({
			mMeshletsBuffer->fill(meshletsGeometry.data(), 0),
			matCommands
		}, *mQueue)->wait_until_signalled();

		view_info info;
		// One for each concurrent frame
		const auto concurrentFrames = avk::context().main_window()->number_of_frames_in_flight();
		for (int i = 0; i < concurrentFrames; ++i) {
			mViewProjBuffers.push_back(avk::context().create_buffer(
				avk::memory_usage::host_coherent, {},
				avk::uniform_buffer_meta::create_from_data(info)
			));
		}

		mMaterialBuffer = avk::context().create_buffer(
			avk::memory_usage::host_visible, {},
			avk::storage_buffer_meta::create_from_data(gpuMaterials)
		);
		auto emptyCommand = mMaterialBuffer->fill(gpuMaterials.data(), 0);

		mImageSamplers = std::move(imageSamplers);

		// Before creating a pipeline, let's query the VK_EXT_mesh_shader-specific device properties:
		// Also, just out of curiosity, query the subgroup properties too:
		vk::PhysicalDeviceMeshShaderPropertiesEXT meshShaderProps{};
		vk::PhysicalDeviceSubgroupProperties subgroupProps;
		vk::PhysicalDeviceProperties2 phProps2{};
		phProps2.pNext = &meshShaderProps;
		meshShaderProps.pNext = &subgroupProps;
		avk::context().physical_device().getProperties2(&phProps2);
		LOG_INFO(std::format("Max. preferred task threads is {}, mesh threads is {}, subgroup size is {}.",
			meshShaderProps.maxPreferredTaskWorkGroupInvocations,
			meshShaderProps.maxPreferredMeshWorkGroupInvocations,
			subgroupProps.subgroupSize));
		LOG_INFO(std::format("This device supports the following subgroup operations: {}", vk::to_string(subgroupProps.supportedOperations)));
		LOG_INFO(std::format("This device supports subgroup operations in the following stages: {}", vk::to_string(subgroupProps.supportedStages)));
		mTaskInvocationsExt = meshShaderProps.maxPreferredTaskWorkGroupInvocations;

		mIndirectDrawParam.emplace_back(vk::DrawMeshTasksIndirectCommandEXT(avk::div_ceil(mNumMeshlets* num_instances2, mTaskInvocationsExt), 1, 1));
		//mIndirectDrawParam.emplace_back(vk::DrawMeshTasksIndirectCommandEXT(avk::div_ceil(mNumMeshlets, mTaskInvocationsExt), 1, 1));
		mIndirectDrawParamBuffer = avk::context().create_buffer(
			avk::memory_usage::host_coherent, {},
			avk::indirect_buffer_meta::create_from_data(mIndirectDrawParam));
		auto emptyCommand2 = mIndirectDrawParamBuffer->fill(mIndirectDrawParam.data(), 0);

		// Create our graphics mesh pipeline with the required configuration:
		auto createGraphicsMeshPipeline = [this](auto taskShader, auto meshShader, uint32_t taskInvocations, uint32_t meshInvocations) {
			return avk::context().create_graphics_pipeline_for(
			    // Specify which shaders the pipeline consists of:
				avk::task_shader(taskShader)
					.set_specialization_constant(0, taskInvocations),
				avk::mesh_shader(meshShader)
					.set_specialization_constant(0, taskInvocations)
					.set_specialization_constant(1, meshInvocations),
			    avk::fragment_shader("shaders/diffuse_shading_fixed_lightsource.frag"),
			    // Some further settings:
			    avk::cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			    avk::cfg::viewport_depth_scissors_config::from_framebuffer(avk::context().main_window()->backbuffer_reference_at_index(0)),
				avk::cfg::polygon_drawing(avk::cfg::polygon_drawing::dynamic_for_lines()),
			    // We'll render to the back buffer, which has a color attachment always, and in our case additionally a depth
			    // attachment, which has been configured when creating the window. See main() function!
			    avk::context().create_renderpass({
				    avk::attachment::declare(avk::format_from_window_color_buffer(avk::context().main_window()), avk::on_load::clear.from_previous_layout(avk::layout::undefined), avk::usage::color(0)     , avk::on_store::store).set_clear_color({1.f, 1.f, 1.f, 0.0f}),
				    avk::attachment::declare(avk::format_from_window_depth_buffer(avk::context().main_window()), avk::on_load::clear.from_previous_layout(avk::layout::undefined), avk::usage::depth_stencil, avk::on_store::dont_care)
				    }, avk::context().main_window()->renderpass_reference().subpass_dependencies()),
			    // The following define additional data which we'll pass to the pipeline:
			    avk::push_constant_binding_data{ avk::shader_type::all, 0, sizeof(push_constants) },
			    avk::descriptor_binding(0, 0, avk::as_combined_image_samplers(mImageSamplers, avk::layout::shader_read_only_optimal)),
			    avk::descriptor_binding(0, 1, mViewProjBuffers[0]),
			    avk::descriptor_binding(1, 0, mMaterialBuffer),
				avk::descriptor_binding(1, 1, mInstanceMatricesBuffer[0]),
			    avk::descriptor_binding(2, 0, mBoneMatricesBuffersAni[0]),
			    // texel buffers
			    avk::descriptor_binding(3, 0, avk::as_uniform_texel_buffer_views(mPositionBuffers)),
			    avk::descriptor_binding(3, 2, avk::as_uniform_texel_buffer_views(mNormalBuffers)),
			    avk::descriptor_binding(3, 3, avk::as_uniform_texel_buffer_views(mTexCoordsBuffers)),
#if USE_REDIRECTED_GPU_DATA
			    avk::descriptor_binding(3, 4, avk::as_storage_buffers(mIndicesDataBuffers)),
#endif
			    avk::descriptor_binding(3, 5, avk::as_uniform_texel_buffer_views(mBoneIndicesBuffers)),
			    avk::descriptor_binding(3, 6, avk::as_uniform_texel_buffer_views(mBoneWeightsBuffers)),
			    avk::descriptor_binding(4, 0, mMeshletsBuffer)
		    );
		};

		mPipelineExt = createGraphicsMeshPipeline(
			"shaders/meshlet.task", "shaders/meshlet.mesh",
			meshShaderProps.maxPreferredTaskWorkGroupInvocations,
			meshShaderProps.maxPreferredMeshWorkGroupInvocations
		);
		// we want to use an updater, so create one:
		mUpdater.emplace();
		mUpdater->on(avk::shader_files_changed_event(mPipelineExt.as_reference())).update(mPipelineExt);


		mPipelineDebug = createGraphicsMeshPipeline(
			"shaders/meshlet.task", "shaders/debug.mesh", 
			meshShaderProps.maxPreferredTaskWorkGroupInvocations,
			meshShaderProps.maxPreferredMeshWorkGroupInvocations
		);

		mUpdater->on(avk::shader_files_changed_event(mPipelineDebug.as_reference())).update(mPipelineDebug);


		// Add the camera to the composition (and let it handle the updates)
		float sceneSize = mSceneBBox.diagonal().norm() * num_instances;
		glm::vec3 sceneCenter = glm::vec3(mSceneBBox.center().x(), mSceneBBox.center().y(), mSceneBBox.center().z());
		mZoom = 2.f * mSceneBBox.diagonal().norm();
		mOrbitCam.set_translation(sceneCenter - glm::vec3(mZoom, 0.0f, 0.0f ));
		mOrbitCam.look_at(sceneCenter);
		mOrbitCam.set_pivot_distance(sqrt(2.f)*sceneSize);
		mQuakeCam.set_translation({ 0.0f, sceneSize, 0.0f });
		mQuakeCam.look_at(sceneCenter);
		mOrbitCam.set_perspective_projection(glm::radians(60.0f), avk::context().main_window()->aspect_ratio(), 0.3f, 1000.0f);
		mQuakeCam.set_perspective_projection(glm::radians(60.0f), avk::context().main_window()->aspect_ratio(), 0.3f, 1000.0f);
		avk::current_composition()->add_element(mOrbitCam);
		avk::current_composition()->add_element(mQuakeCam);
		mQuakeCam.disable();

		auto imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if (nullptr != imguiManager) {
			imguiManager->add_callback([
				this, imguiManager,
				timestampPeriod = std::invoke([]() {
				// get timestamp period from physical device, could be different for other GPUs
				auto props = avk::context().physical_device().getProperties();
				return static_cast<double>(props.limits.timestampPeriod);
					}),
				lastFrameDurationMs = 0.0,
				lastDrawMeshTasksDurationMs = 0.0
			]() mutable {
				ImGui::Begin("Info & Settings");
				ImGui::SetWindowPos(ImVec2(1.0f, 1.0f), ImGuiCond_FirstUseEver);
				ImGui::Text("%.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
				ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
				ImGui::Separator();
				/*ImGui::TextColored(ImVec4(.5f, .3f, .4f, 1.f), "Timestamp Period: %.3f ns", timestampPeriod);
				lastFrameDurationMs = glm::mix(lastFrameDurationMs, mLastFrameDuration * 1e-6 * timestampPeriod, 0.05);
				lastDrawMeshTasksDurationMs = glm::mix(lastDrawMeshTasksDurationMs, mLastDrawMeshTasksDuration * 1e-6 * timestampPeriod, 0.05);
				ImGui::TextColored(ImVec4(.8f, .1f, .6f, 1.f), "Frame time (timer queries): %.3lf ms", lastFrameDurationMs);
				ImGui::TextColored(ImVec4(.8f, .1f, .6f, 1.f), "drawMeshTasks took        : %.3lf ms", lastDrawMeshTasksDurationMs);*/
				ImGui::Text("Fragment shader : %llu", mPipelineStats[0]);
				ImGui::Text("Task shader     : %llu", mPipelineStats[1]);
				ImGui::Text("Mesh shader     : %llu", mPipelineStats[2]);
				
				ImGui::Separator();
				bool quakeCamEnabled = mQuakeCam.is_enabled();
				if (ImGui::Checkbox("Enable Quake Camera", &quakeCamEnabled)) {
					if (quakeCamEnabled) { // => should be enabled
						mQuakeCam.set_matrix(mOrbitCam.matrix());
						mQuakeCam.enable();
						mOrbitCam.disable();
					}
				}
				if (quakeCamEnabled) {
					ImGui::TextColored(ImVec4(0.f, .6f, .8f, 1.f), "[F1] to exit Quake Camera navigation.");
					if (avk::input().key_pressed(avk::key_code::f1)) {
						mOrbitCam.set_matrix(mQuakeCam.matrix());
						mOrbitCam.enable();
						mQuakeCam.disable();
					}
				}
				if (imguiManager->begin_wanting_to_occupy_mouse() && mOrbitCam.is_enabled()) {
					mOrbitCam.disable();
				}
				if (imguiManager->end_wanting_to_occupy_mouse() && !mQuakeCam.is_enabled()) {
					mOrbitCam.enable();
				}
				ImGui::Separator();

				ImGui::Separator();
				int choice = mUseDebugPipeline ? 1 : 0;
				ImGui::Combo("Pipeline", &choice, "Contours\0Debug\0");
				mUseDebugPipeline = (choice == 1);
				ImGui::Separator();

				// Select the range of meshlets to be rendered:
				ImGui::Checkbox("Highlight meshlets", &mHighlightMeshlets);
				ImGui::Checkbox("Cull meshlets", &mCullMeshlets);
				ImGui::Checkbox("Extract contours", &mExtractContours);
				ImGui::Text("Select meshlets to be rendered:");
				ImGui::DragIntRange2("Visible range", &mShowMeshletsFrom, &mShowMeshletsTo, 1, 0, static_cast<int>(mNumMeshlets));
				ImGui::Checkbox("Animate", &mAnimate);

				ImGui::End();
			});
		}

		mTimestampPool = avk::context().create_query_pool_for_timestamp_queries(
			static_cast<uint32_t>(avk::context().main_window()->number_of_frames_in_flight()) * 2
		);

		mPipelineStatsPool = avk::context().create_query_pool_for_pipeline_statistics_queries(
			vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations | vk::QueryPipelineStatisticFlagBits::eMeshShaderInvocationsEXT | vk::QueryPipelineStatisticFlagBits::eTaskShaderInvocationsEXT,
			avk::context().main_window()->number_of_frames_in_flight()
		);
	}

	void update() override
	{
		//if (avk::input().key_pressed(avk::key_code::c)) {
		//	// Center the cursor:
		//	auto resolution = avk::context().main_window()->resolution();
		//	avk::context().main_window()->set_cursor_pos({ resolution[0] / 2.0, resolution[1] / 2.0 });
		//}
		if (avk::input().key_pressed(avk::key_code::escape)) {
			// Stop the current composition:
			avk::current_composition()->stop();
		}
		if (avk::input().key_pressed(avk::key_code::p)) {
			// activate the culling
			mCullMeshlets = !mCullMeshlets;
		}
		if (avk::input().key_pressed(avk::key_code::c)) {
			// activate contour extraction
			mExtractContours = !mExtractContours;
		}
	}

	void render() override
	{
		using namespace avk;

		auto mainWnd = context().main_window();
		auto inFlightIndex = mainWnd->current_in_flight_index();

		// Animate all the meshes
		for (auto& model : mAnimatedModels) {
			auto& animation = std::get<animated_model_data>(model).mAnimation;
			auto& clip = std::get<animated_model_data>(model).mClip;
			const auto doubleTime = fmod(time().absolute_time_dp(), std::get<animated_model_data>(model).duration_sec() * 2);
			auto time = glm::lerp(std::get<animated_model_data>(model).start_sec(), std::get<animated_model_data>(model).end_sec(), (doubleTime > std::get<animated_model_data>(model).duration_sec() ? doubleTime - std::get<animated_model_data>(model).duration_sec() : doubleTime) / std::get<animated_model_data>(model).duration_sec());
			auto targetMemory = std::get<additional_animated_model_data>(model).mBoneMatricesAni.data();

			// Use lambda option 1 that takes as parameters: mesh_bone_info, inverse mesh root matrix, global node/bone transform w.r.t. the animation, inverse bind-pose matrix
			animation.animate(clip, time, [this, &animation, targetMemory](mesh_bone_info aInfo, const glm::mat4& aInverseMeshRootMatrix, const glm::mat4& aTransformMatrix, const glm::mat4& aInverseBindPoseMatrix, const glm::mat4& aLocalTransformMatrix, size_t aAnimatedNodeIndex, size_t aBoneMeshTargetIndex, double aAnimationTimeInTicks) {
				// Construction of the bone matrix for this node:
				//   1. Bring vertex into bone space
				//   2. Apply transformaton in bone space => MODEL SPACE
				targetMemory[aInfo.mGlobalBoneIndexOffset + aInfo.mMeshLocalBoneIndex] = aTransformMatrix * aInverseBindPoseMatrix;
				});
		}

		float sceneDiag = mSceneBBox.diagonal().norm();
		glm::vec3 sceneCenter = glm::vec3(0, 0, 0); //glm::vec3(mSceneBBox.center().x(), mSceneBBox.center().y(), mSceneBBox.center().z());
		if (mAnimate) {
			mTime = avk::time().absolute_time_dp();
			if (mZoom > 50) mGrow = false;
			if (mZoom < 1.5) mGrow = true;
			float speed = 5.f * std::exp((mZoom - 1.5f) / 50.f);
			mZoom = mGrow ? mZoom + avk::time().delta_time() * speed : mZoom - avk::time().delta_time() * speed;
			mOrbitCam.set_translation({ 0.0f, mZoom * sceneDiag, 0.0f });
			mOrbitCam.look_at(sceneCenter);
		}
		// create buffer for instances
		for (int32_t y = -grid_size; y <= grid_size; y++) {
			for (int32_t x = -grid_size; x <= grid_size; x++) {
				glm::mat4 M(1.0f);
				M = glm::translate(M, 0.75f * glm::vec3(x * sceneDiag, 0.0f, y * sceneDiag));
				M = glm::rotate(M, mTime * 1.6f + x * 9774.37f, glm::vec3(1.0f, 0.0f, 0.0f));
				M = glm::rotate(M, mTime * 3.2f + y * 2715.53f, glm::vec3(0.0f, 0.0f, 1.0f));
				//M = glm::scale(M, glm::vec3(sin(time + (x ^ y) * 13.73f) * 0.2f + 0.8f));
				mInstanceMatrices[(y + grid_size) * num_instances + x + grid_size] = M;
			}
		}		

		view_info infos;
		infos.mViewProjMatrix = mQuakeCam.is_enabled()
			? mQuakeCam.projection_and_view_matrix()
			: mOrbitCam.projection_and_view_matrix();
		infos.mCameraCenter = mQuakeCam.is_enabled()
			? mQuakeCam.translation()
			: mOrbitCam.translation();
		auto emptyCmd = mViewProjBuffers[inFlightIndex]->fill(&infos, 0);

		// Get a command pool to allocate command buffers from:
		auto& commandPool = context().get_command_pool_for_single_use_command_buffers(*mQueue);

		// The swap chain provides us with an "image available semaphore" for the current frame.
		// Only after the swapchain image has become available, we may start rendering into it.
		auto imageAvailableSemaphore = mainWnd->consume_current_image_available_semaphore();

		// Create a command buffer and render into the *current* swap chain image:
		auto cmdBfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

		const auto firstQueryIndex = static_cast<uint32_t>(inFlightIndex) * 2;
		if (mainWnd->current_frame() > mainWnd->number_of_frames_in_flight()) // otherwise we will wait forever
		{
			auto timers = mTimestampPool->get_results<uint64_t, 2>(
				firstQueryIndex, 2, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait // => ensure that the results are available
			);
			mLastDrawMeshTasksDuration = timers[1] - timers[0];
			mLastFrameDuration = timers[1] - mLastTimestamp;
			mLastTimestamp = timers[1];

			mPipelineStats = mPipelineStatsPool->get_results<uint64_t, 3>(inFlightIndex, 1, vk::QueryResultFlagBits::e64);
		}

		auto& pipeline = mUseDebugPipeline ? mPipelineDebug : mPipelineExt;

		context().record(command::gather(
			mPipelineStatsPool->reset(inFlightIndex, 1),
			mPipelineStatsPool->begin_query(inFlightIndex),
			mTimestampPool->reset(firstQueryIndex, 2),     // reset the two values relevant for the current frame in flight
			mTimestampPool->write_timestamp(firstQueryIndex + 0, stage::all_commands), // measure before drawMeshTasks*

			command::custom_commands([](avk::command_buffer_t& cb) { vkCmdSetLineWidth(cb.handle(), 2.f); }),

			// Upload the updated bone matrices into the buffer for the current frame (considering that we have cConcurrentFrames-many concurrent frames):
			command::one_for_each(mAnimatedModels, [this, inFlightIndex](const std::tuple<animated_model_data, additional_animated_model_data>& tpl) {
				return mBoneMatricesBuffersAni[inFlightIndex][std::get<animated_model_data>(tpl).mBoneMatricesBufferIndex]->fill(std::get<additional_animated_model_data>(tpl).mBoneMatricesAni.data(), 0);
				}),

			command::conditional([this]() { return true; }, 
				[this, inFlightIndex]() {
					return mInstanceMatricesBuffer[inFlightIndex]->fill(mInstanceMatrices.data(), 0);
				}),

			command::render_pass(pipeline->renderpass_reference(), context().main_window()->current_backbuffer_reference(), {
				command::bind_pipeline(pipeline.as_reference()),
				command::bind_descriptors(pipeline->layout(), mDescriptorCache->get_or_create_descriptor_sets({
					descriptor_binding(0, 0, as_combined_image_samplers(mImageSamplers, layout::shader_read_only_optimal)),
					descriptor_binding(0, 1, mViewProjBuffers[inFlightIndex]),
					descriptor_binding(1, 0, mMaterialBuffer),
					descriptor_binding(1, 1, mInstanceMatricesBuffer[inFlightIndex]),
					descriptor_binding(2, 0, mBoneMatricesBuffersAni[inFlightIndex]),
					descriptor_binding(3, 0, as_uniform_texel_buffer_views(mPositionBuffers)),
					descriptor_binding(3, 2, as_uniform_texel_buffer_views(mNormalBuffers)),
					descriptor_binding(3, 3, as_uniform_texel_buffer_views(mTexCoordsBuffers)),
#if USE_REDIRECTED_GPU_DATA
						descriptor_binding(3, 4, avk::as_storage_buffers(mIndicesDataBuffers)),
#endif
						descriptor_binding(3, 5, as_uniform_texel_buffer_views(mBoneIndicesBuffers)),
						descriptor_binding(3, 6, as_uniform_texel_buffer_views(mBoneWeightsBuffers)),
						descriptor_binding(4, 0, mMeshletsBuffer)
					})),

					command::push_constants(pipeline->layout(), push_constants{
						mHighlightMeshlets,
						mCullMeshlets,
						mExtractContours,
						static_cast<int32_t>(mShowMeshletsFrom),
						static_cast<int32_t>(mShowMeshletsTo),
						static_cast<int32_t>(num_instances2)
					}),

				// Draw all the meshlets with just one single draw call:
				//command::draw_mesh_tasks_ext(div_ceil(mNumMeshlets * num_instances2, mTaskInvocationsExt), 1, 1),

				command::draw_mesh_tasks_indirect_ext(mIndirectDrawParamBuffer, 0, 1, sizeof(vk::DrawMeshTasksIndirectCommandEXT)),
				}),

				mTimestampPool->write_timestamp(firstQueryIndex + 1, stage::mesh_shader),
				mPipelineStatsPool->end_query(inFlightIndex)
			))
			.into_command_buffer(cmdBfr)
			.then_submit_to(*mQueue)
			// Do not start to render before the image has become available:
			.waiting_for(imageAvailableSemaphore >> stage::color_attachment_output)
			.submit();
					
		mainWnd->handle_lifetime(std::move(cmdBfr));
	}

private: // v== Member variables ==v

	avk::queue* mQueue;
	avk::descriptor_cache mDescriptorCache;

	std::vector<std::tuple<animated_model_data, additional_animated_model_data>> mAnimatedModels;
	Eigen::AlignedBox3d mSceneBBox;

	std::vector<avk::buffer> mViewProjBuffers;
	avk::buffer mMaterialBuffer;
	avk::buffer mMeshletsBuffer;
	std::array<std::vector<avk::buffer>, cConcurrentFrames> mBoneMatricesBuffersAni;
	std::vector<avk::image_sampler> mImageSamplers;

	// scene size
	int32_t grid_size = 1;
	uint32_t num_instances = grid_size * 2 + 1;
	uint32_t num_instances2 = num_instances * num_instances;
	std::vector<glm::mat4> mInstanceMatrices;
	std::array<avk::buffer, cConcurrentFrames> mInstanceMatricesBuffer;
	float mTime{ 0.f };
	float mZoom{ 5.f };
	bool mGrow{ true };

	std::vector<data_for_draw_call> mDrawCalls;
	avk::graphics_pipeline mPipelineExt;
	avk::graphics_pipeline mPipelineDebug;

	avk::orbit_camera mOrbitCam;
	avk::quake_camera mQuakeCam;

	std::vector<vk::DrawMeshTasksIndirectCommandEXT> mIndirectDrawParam;
	avk::buffer mIndirectDrawParamBuffer;

    uint32_t mNumMeshlets;
	uint32_t mTaskInvocationsExt;
	uint32_t mTaskInvocationsNv;

	std::vector<avk::buffer_view> mPositionBuffers;
	std::vector<avk::buffer_view> mTexCoordsBuffers;
	std::vector<avk::buffer_view> mNormalBuffers;
	std::vector<avk::buffer_view> mBoneWeightsBuffers;
	std::vector<avk::buffer_view> mBoneIndicesBuffers;
#if USE_REDIRECTED_GPU_DATA
	std::vector<avk::buffer> mIndicesDataBuffers;
#endif

	bool mBuildMeshlets = false;
	bool mHighlightMeshlets = false;
	bool mCullMeshlets = false;
	bool mExtractContours = false;
	int  mShowMeshletsFrom  = 0;
	int  mShowMeshletsTo    = 0;
	bool mUseDebugPipeline = false;
	bool mAnimate = false;

	avk::query_pool mTimestampPool;
	uint64_t mLastTimestamp = 0;
	uint64_t mLastDrawMeshTasksDuration = 0;
	uint64_t mLastFrameDuration = 0;

	avk::query_pool mPipelineStatsPool;
	std::array<uint64_t, 3> mPipelineStats;

}; // skinned_meshlets_app

int main() // <== Starting point ==
{
	int result = EXIT_FAILURE;
	try {
		// Create a window and open it
		auto mainWnd = avk::context().create_window("Edge Contour Meshlets");

		mainWnd->set_resolution({ 1920, 1080 });
		mainWnd->enable_resizing(true);
		mainWnd->set_additional_back_buffer_attachments({
			avk::attachment::declare(vk::Format::eD32Sfloat, avk::on_load::clear.from_previous_layout(avk::layout::undefined), avk::usage::depth_stencil, avk::on_store::dont_care)
		});
		mainWnd->set_presentaton_mode(avk::presentation_mode::mailbox);
		mainWnd->set_number_of_concurrent_frames(cConcurrentFrames);
		mainWnd->open();

		auto& singleQueue = avk::context().create_queue({}, avk::queue_selection_preference::versatile_queue, mainWnd);
		mainWnd->set_queue_family_ownership(singleQueue.family_index());
		mainWnd->set_present_queue(singleQueue);

		// Create an instance of our main avk::element which contains all the functionality:
		auto app = skinned_meshlets_app(singleQueue);
		// Create another element for drawing the UI with ImGui
		auto ui = avk::imgui_manager(singleQueue);

		// Compile all the configuration parameters and the invokees into a "composition":
		auto composition = configure_and_compose(
			avk::application_name("Edge Contour Meshlets"),
			// Gotta enable the mesh shader extension, ...
			avk::required_device_extensions(VK_EXT_MESH_SHADER_EXTENSION_NAME),
			avk::optional_device_extensions(VK_NV_MESH_SHADER_EXTENSION_NAME),
			// ... and enable the mesh shader features that we need:
			[](vk::PhysicalDeviceMeshShaderFeaturesEXT& meshShaderFeatures) {
				meshShaderFeatures.setMeshShader(VK_TRUE);
				meshShaderFeatures.setTaskShader(VK_TRUE);
				meshShaderFeatures.setMeshShaderQueries(VK_TRUE);
			},
			[](vk::PhysicalDeviceFeatures& features) {
				features.setPipelineStatisticsQuery(VK_TRUE);
				features.setWideLines(VK_TRUE);
			},
			[](vk::PhysicalDeviceVulkan12Features& features) {
				features.setUniformAndStorageBuffer8BitAccess(VK_TRUE);
				features.setStorageBuffer8BitAccess(VK_TRUE);
			},
			// Pass windows:
			mainWnd,
			// Pass invokees:
			app, ui
		);

		// Create an invoker object, which defines the way how invokees/elements are invoked
		// (In this case, just sequentially in their execution order):
		avk::sequential_invoker invoker;

		// With everything configured, let us start our render loop:
		composition.start_render_loop(
			// Callback in the case of update:
			[&invoker](const std::vector<avk::invokee*>& aToBeInvoked) {
				// Call all the update() callbacks:
				invoker.invoke_updates(aToBeInvoked);
			},
			// Callback in the case of render:
				[&invoker](const std::vector<avk::invokee*>& aToBeInvoked) {
				// Sync (wait for fences and so) per window BEFORE executing render callbacks
				avk::context().execute_for_each_window([](avk::window* wnd) {
					wnd->sync_before_render();
					});

				// Call all the render() callbacks:
				invoker.invoke_renders(aToBeInvoked);

				// Render per window:
				avk::context().execute_for_each_window([](avk::window* wnd) {
					wnd->render_frame();
				});
			}
			); // This is a blocking call, which loops until avk::current_composition()->stop(); has been called (see update())

		result = EXIT_SUCCESS;
	}
	catch (avk::logic_error&) {}
	catch (avk::runtime_error&) {}
	return result;
}
