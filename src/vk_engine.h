// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
//> intro
#pragma once

#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_loader.h>


struct ComputePushConstants
{
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect
{
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
};

/**
* @brief Holds all of the deletion functions to make the cleanup() function cleaner.
*/
struct DeletionQueue {

	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deltion queue to execute all the deletors
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)();
		}
		deletors.clear();
	}
};

struct FrameData {

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;

	DescriptorAllocatorGrowable _frameDescriptors;

	DeletionQueue _deletionQueue;
};

// Sets the level of multibuffering
constexpr unsigned int FRAME_OVERLAP = 2;

/**
* @brief GLTF PBR Pipelines for rendering
*/
struct GLTFMetallic_Roughness {
	MaterialPipeline opaquePipeline;
	MaterialPipeline transparentPipeline;

	VkDescriptorSetLayout materialLayout;

	struct MaterialConstants {
		glm::vec4 colorFactors;
		glm::vec4 metal_rough_factors;
		// padding to 64 bytes
		glm::vec4 extra[14];
	};

	struct MaterialResources {
		AllocatedImage colorImage;
		VkSampler colorSampler;
		AllocatedImage metalRoughImage;
		VkSampler metalRoughSampler;
		VkBuffer dataBuffer;
		uint32_t dataBufferOffset;
	};

	DescriptorWriter writer;

	void build_pipelines(VulkanEngine* engine);
	void clear_resources(VkDevice device);

	MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};


/**
* @brief Holds the data for a Render Object	
* 
*/
struct RenderObject {
	uint32_t indexCount;
	uint32_t firstIndex;
	VkBuffer indexBuffer;

	MaterialInstance* material;

	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

struct DrawContext {
	std::vector<RenderObject> OpaqueSurfaces;
};

/**
* @brief Holds the data for a mesh asset
* 
*/
struct MeshNode : public Node {
	std::shared_ptr<MeshAsset> mesh;
	
	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};


class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{ 1200 , 800 };
	float renderScale = 1.0f;

	struct SDL_Window* _window{ nullptr };


	// Main Draw Context
	DrawContext mainDrawContext;
	std::unordered_map<std::string, std::shared_ptr<Node>> _sceneGraph;

	void update_scene();

	static VulkanEngine& Get();

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();
	void draw_background(VkCommandBuffer cmd);
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
	void draw_geometry(VkCommandBuffer cmd);

	//run main loop
	void run();

private:
	// Mouse velocity change
	glm::vec2 _mousePos{ 0.0f, 0.0f };
	glm::vec2 _mouseVel{ 0.0f, 0.0f };
	bool _mouseButtons[3]{ false, false, false }; // Left, right, middle


	// Camera settings
	glm::vec3 _camPos{ 0.0f, 0.0f, 5.0f };
	glm::vec3 _camRot{ 0.0f, 0.0f, 0.0f };
	float _camZoom{ 0.0f };
public:
	
	VkInstance _instance; // Vulkan library handles
	VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
	VkPhysicalDevice _chosenGPU; // GPU chosen as default device
	VkDevice _device; // Vulkan device with EXT for commands
	VkSurfaceKHR _surface; // Vulkan window surface

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;

	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	VmaAllocator _allocator;

	DescriptorAllocatorGrowable globalDescriptorAllocator;

	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;

	// GPU Scene Data
	GPUSceneData _sceneData{};
	VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

	// Immediate Submit Structures needed later on for ImGUI
	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;

	// Public Draw resources
	AllocatedImage _drawImage;
	AllocatedImage _depthImage;
	VkExtent2D _drawExtent;

	// GLTF Draw resources
	MaterialInstance defaultData;
	GLTFMetallic_Roughness metalRoughMaterial;

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
private:

	DeletionQueue _mainDeletionQueue;
	FrameData _frames[FRAME_OVERLAP];

	// Draw Resources
	bool resize_requested{ false };

	// Texture Resources
	AllocatedImage _whiteImage;
	AllocatedImage _blackImage;
	AllocatedImage _grayImage;
	AllocatedImage _errorCheckerboardImage;

	VkDescriptorSetLayout _singleImageDescriptorLayout;

	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	// Compute Resources
	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{ 0 };
	std::vector<std::shared_ptr<MeshAsset>> _meshAssets;

public:
	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	VkPipeline _gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;

	VkPipelineLayout _trianglePipelineLayout;
	VkPipeline _trianglePipeline;

	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;

	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
private:

	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();

	void init_descriptors();

	void init_pipelines();
	void init_background_pipelines();
	void init_triangle_pipeline();
	void init_mesh_pipeline();

	void init_default_data();

	void init_imgui();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void resize_swapchain();

	// Texture functions
	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroy_image(const AllocatedImage& image);

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroy_buffer(const AllocatedBuffer& buffer);
};
//< intro