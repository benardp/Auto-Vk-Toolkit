#include <gvk.hpp>
#include <imgui.h>

class multiple_queues_app : public gvk::invokee
{
	// Define a struct for our vertex input data:
	struct Vertex {
	    glm::vec3 pos;
	    glm::vec3 color;
	};

	// Vertex data for drawing a pyramid:
	const std::vector<Vertex> mVertexData = {
		// pyramid front
		{{ 0.0f, -0.5f, 0.5f},  {1.0f, 0.0f, 0.0f}},
		{{ 0.3f,  0.5f, 0.2f},  {0.5f, 0.5f, 0.5f}},
		{{-0.3f,  0.5f, 0.2f},  {0.5f, 0.5f, 0.5f}},
		// pyramid right
		{{ 0.0f, -0.5f, 0.5f},  {1.0f, 0.0f, 0.0f}},
		{{ 0.3f,  0.5f, 0.8f},  {0.6f, 0.6f, 0.6f}},
		{{ 0.3f,  0.5f, 0.2f},  {0.6f, 0.6f, 0.6f}},
		// pyramid back
		{{ 0.0f, -0.5f, 0.5f},  {1.0f, 0.0f, 0.0f}},
		{{-0.3f,  0.5f, 0.8f},  {0.5f, 0.5f, 0.5f}},
		{{ 0.3f,  0.5f, 0.8f},  {0.5f, 0.5f, 0.5f}},
		// pyramid left
		{{ 0.0f, -0.5f, 0.5f},  {1.0f, 0.0f, 0.0f}},
		{{-0.3f,  0.5f, 0.2f},  {0.4f, 0.4f, 0.4f}},
		{{-0.3f,  0.5f, 0.8f},  {0.4f, 0.4f, 0.4f}},
	};

	// Indices for the faces (triangles) of the pyramid:
	const std::vector<uint16_t> mIndices = {
		 0, 1, 2,  3, 4, 5,  6, 7, 8,  9, 10, 11
	};

public: // v== avk::invokee overrides which will be invoked by the framework ==v
	multiple_queues_app(avk::queue& aQueue)
		: mQueue{ &aQueue }
		, mAdditionalTranslationY{ 0.0f }
		, mRotationSpeed{ 1.0f }
	{ }

	void initialize() override
	{
		// Create graphics pipeline for rasterization with the required configuration:
		mPipeline = gvk::context().create_graphics_pipeline_for(
			avk::from_buffer_binding(0) -> stream_per_vertex(&Vertex::pos)   -> to_location(0),	// Describe the position vertex attribute
			avk::from_buffer_binding(0) -> stream_per_vertex(&Vertex::color) -> to_location(1), // Describe the color vertex attribute
			avk::vertex_shader("shaders/passthrough.vert"),										// Add a vertex shader
			avk::fragment_shader("shaders/color.frag"),											// Add a fragment shader
			avk::cfg::front_face::define_front_faces_to_be_clockwise(),							// Front faces are in clockwise order
			avk::cfg::viewport_depth_scissors_config::from_framebuffer(gvk::context().main_window()->backbuffer_reference_at_index(0)), // Align viewport with main window's resolution
			gvk::context().main_window()->renderpass()
		);

		// Create vertex buffers --- namely one for each frame in flight.
		// We create multiple vertex buffers because we'll update the data every frame and frames run concurrently.
		// However, do not upload vertices yet. we'll do that in the render() method.
		auto numFramesInFlight = gvk::context().main_window()->number_of_frames_in_flight();
		for (int i = 0; i < numFramesInFlight; ++i) {
			mVertexBuffers.emplace_back(
				gvk::context().create_buffer(
					avk::memory_usage::device, {},							// Create the buffer on the device, i.e. in GPU memory, (no additional usage flags).
					avk::vertex_buffer_meta::create_from_data(mVertexData)		// Infer meta data from the given buffer.
				)
			);
		}
		
		// Create index buffer. Upload data already since we won't change it ever
		// (hence the usage of avk::create_and_fill instead of just avk::create)
		mIndexBuffer = gvk::context().create_buffer(
			avk::memory_usage::device, {},										// Also this buffer should "live" in GPU memory
			avk::index_buffer_meta::create_from_data(mIndices)					// Pass/create meta data about the indices
		);

		// Fill it with data already here, in initialize(), because this buffer will stay constant forever:
		auto fence = gvk::context().record_and_submit_with_fence({ mIndexBuffer->fill(mIndices.data(), 0) }, mQueue);
		// Wait with a fence until the data transfer has completed:
		fence->wait_until_signalled();

		// Get hold of the "ImGui Manager" and add a callback that draws UI elements:
		auto imguiManager = gvk::current_composition()->element_by_type<gvk::imgui_manager>();
		if (nullptr != imguiManager) {
			imguiManager->add_callback([this](){
		        ImGui::Begin("Info & Settings");
				ImGui::SetWindowPos(ImVec2(1.0f, 1.0f), ImGuiCond_FirstUseEver);
				ImGui::Text("%.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
				ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
				ImGui::SliderFloat("Translation", &mAdditionalTranslationY, -1.0f, 1.0f);
				ImGui::InputFloat("Rotation Speed", &mRotationSpeed, 0.1f, 1.0f);
		        ImGui::End();
			});
		}
	}

	void update() override
	{
		// On C pressed,
		if (gvk::input().key_pressed(gvk::key_code::c)) {
			// center the cursor:
			auto resolution = gvk::context().main_window()->resolution();
			gvk::context().main_window()->set_cursor_pos({ resolution[0] / 2.0, resolution[1] / 2.0 });
		}

		// On Esc pressed,
		if (gvk::input().key_pressed(gvk::key_code::escape)) {
			// stop the current composition:
			gvk::current_composition()->stop();
		}
	}

	void render() override
	{
		// Modify our vertex data according to our rotation animation and upload this frame's vertex data:
		auto rotAngle = glm::radians(90.0f) * gvk::time().time_since_start() * mRotationSpeed;
		auto rotMatrix = glm::rotate(rotAngle, glm::vec3(0.f, 1.f, 0.f));
		auto translateZ = glm::translate(glm::vec3{ 0.0f, 0.0f, -0.5f });
		auto invTranslZ = glm::inverse(translateZ);
		auto translateY = glm::translate(glm::vec3{ 0.0f, -mAdditionalTranslationY, 0.0f });

		// Store modified vertices in vertexDataCurrentFrame:
		std::vector<Vertex> vertexDataCurrentFrame;
		for (const Vertex& vtx : mVertexData) {
			glm::vec4 origPos{ vtx.pos, 1.0f };
			vertexDataCurrentFrame.push_back({
				glm::vec3(translateY * invTranslZ * rotMatrix * translateZ * origPos),
				vtx.color
			});
		}

		// Note: Updating this data in update() would also be perfectly fine when using a varying_update_timer.
		//		 However, when using a fixed_update_timer --- where update() and render() might not be called
		//		 in sync --- it can make a difference.
		//		 Since the buffer-updates here are solely rendering-relevant and do not depend on other aspects
		//		 like e.g. input or physics simulation, it makes most sense to perform them in render(). This
		//		 will ensure correct and smooth rendering regardless of the timer used.

		// For the right vertex buffer, ...
		auto mainWnd = gvk::context().main_window();
		auto inFlightIndex = mainWnd->in_flight_index_for_frame();

		// ... update its vertex data, then get a semaphore which signals as soon as the operation has completed:
		auto vertexBufferFillSemaphore = gvk::context().record_and_submit_with_semaphore({
				mVertexBuffers[inFlightIndex]->fill(vertexDataCurrentFrame.data(), 0)
			}, 
			mQueue,
			avk::stage::auto_stage // Let the framework determine the (source) stage after which the semaphore can be signaled (will be stage::copy due to buffer_t::fill)
		);

		// Get a command pool to allocate command buffers from:
		auto& commandPool = gvk::context().get_command_pool_for_single_use_command_buffers(*mQueue);

		// The swap chain provides us with an "image available semaphore" for the current frame.
		// Only after the swapchain image has become available, we may start rendering into it.
		auto imageAvailableSemaphore = mainWnd->consume_current_image_available_semaphore();
		
		// Create a command buffer and render into the *current* swap chain image:
		auto cmdBfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		
		gvk::context().record({
				// Begin and end one renderpass:
				avk::command::render_pass(mPipeline->renderpass_reference(), gvk::context().main_window()->current_backbuffer_reference(), {
					// And within, bind a pipeline and draw three vertices:
					avk::command::bind_pipeline(mPipeline.as_reference()),
					avk::command::draw_indexed(mIndexBuffer.as_reference(), mVertexBuffers[inFlightIndex].as_reference())
				})
			})
			.into_command_buffer(cmdBfr)
			.then_submit_to(mQueue)
			// Do not start to render before the image has become available:
			.waiting_for(imageAvailableSemaphore >> avk::stage::color_attachment_output)
			// Also wait for the data transfer into the vertex buffer has completed:
			.waiting_for(vertexBufferFillSemaphore >> avk::stage::vertex_attribute_input)
			.submit();

		// Let the command buffer handle the semaphore's lifetime:
		cmdBfr->handle_lifetime_of(std::move(vertexBufferFillSemaphore));

		// Use a convenience function of gvk::window to take care of the command buffer's lifetime:
		// It will get deleted in the future after #concurrent-frames have passed by.
		gvk::context().main_window()->handle_lifetime(std::move(cmdBfr));
	}


private: // v== Member variables ==v
	
	avk::queue* mQueue;
	std::vector<avk::buffer> mVertexBuffers;
	avk::buffer mIndexBuffer;
	avk::graphics_pipeline mPipeline;
	float mAdditionalTranslationY;
	float mRotationSpeed;

}; // vertex_buffers_app

int main() // <== Starting point ==
{
	int result = EXIT_FAILURE;
	try {
		// Create a window and open it
		auto mainWnd = gvk::context().create_window("Vertex Buffers");
		mainWnd->set_resolution({ 640, 480 });
		mainWnd->set_presentaton_mode(gvk::presentation_mode::mailbox);
		mainWnd->set_number_of_concurrent_frames(3u);
		mainWnd->open();

		auto& singleQueue = gvk::context().create_queue({}, avk::queue_selection_preference::versatile_queue, mainWnd);
		mainWnd->add_queue_family_ownership(singleQueue);
		mainWnd->set_present_queue(singleQueue);
		
		// Create an instance of our main "invokee" which contains all the functionality:
		auto app = multiple_queues_app(singleQueue);
		// Create another invokee for drawing the UI with ImGui
		auto ui = gvk::imgui_manager(singleQueue);

		// Compile all the configuration parameters and the invokees into a "composition":
		auto composition = configure_and_compose(
			gvk::application_name("Gears-Vk + Auto-Vk Example: Vertex Buffers"),
			[](gvk::validation_layers& config) {
				config.enable_feature(vk::ValidationFeatureEnableEXT::eSynchronizationValidation);
			},
			// Pass windows:
			mainWnd,
			// Pass invokees:
			app, ui
		);

		// Create an invoker object, which defines the way how invokees/elements are invoked
		// (In this case, just sequentially in their execution order):
		gvk::sequential_invoker invoker;

		// With everything configured, let us start our render loop:
		composition.start_render_loop(
			// Callback in the case of update:
			[&invoker](const std::vector<gvk::invokee*>& aToBeInvoked) {
				// Call all the update() callbacks:
				invoker.invoke_updates(aToBeInvoked);
			},
			// Callback in the case of render:
			[&invoker](const std::vector<gvk::invokee*>& aToBeInvoked) {
				// Sync (wait for fences and so) per window BEFORE executing render callbacks
				gvk::context().execute_for_each_window([](gvk::window* wnd) {
					wnd->sync_before_render();
				});

				// Call all the render() callbacks:
				invoker.invoke_renders(aToBeInvoked);

				// Render per window:
				gvk::context().execute_for_each_window([](gvk::window* wnd) {
					wnd->render_frame();
				});
			}
		); // This is a blocking call, which loops until gvk::current_composition()->stop(); has been called (see update())
	
		result = EXIT_SUCCESS;
	}
	catch (gvk::logic_error&) {}
	catch (gvk::runtime_error&) {}
	catch (avk::logic_error&) {}
	catch (avk::runtime_error&) {}
	return result;
}
