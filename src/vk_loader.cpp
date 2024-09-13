#include "stb_image.h"
#include <iostream>
#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath)
{
	fmt::print("Loading GLTF file: {}\n", filePath.string());

	fastgltf::GltfDataBuffer data;
	data.loadFromFile(filePath);

	constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

	fastgltf::Asset gltf;
	fastgltf::Parser parser{};

	auto load = parser.loadBinaryGLTF(&data, filePath.parent_path(), gltfOptions);

	if (load)
	{
		gltf = std::move(load.get());
	}
	else 
	{
		fmt::print("Failed to load GLTF: {} \n", fastgltf::to_underlying(load.error()));
		return {};
	}

	std::vector<std::shared_ptr<MeshAsset>> meshes;

	// Use the same vectors for all meshes so that the memory doesnt reallocate every time
	std::vector<uint32_t> indices;
	std::vector<Vertex> vertices;
	for (fastgltf::Mesh& mesh : gltf.meshes)
	{
		MeshAsset newMesh;

		newMesh.name = mesh.name;

		// Clear the vector arrays for each mesh
		indices.clear();
		vertices.clear();

		for (auto& primitive : mesh.primitives)
		{
			GeoSurface newSurface;
			newSurface.startIdx = (uint32_t)indices.size();
			newSurface.idxCount = (uint32_t)gltf.accessors[primitive.indicesAccessor.value()].count;

			size_t initial_vtx = vertices.size();

			// Load indexes
			{
				fastgltf::Accessor& accessor = gltf.accessors[primitive.indicesAccessor.value()];
				indices.reserve(indices.size() + accessor.count);

				fastgltf::iterateAccessor<std::uint32_t>(gltf, accessor, [&](std::uint32_t idx) {
					indices.push_back(idx + initial_vtx);
					});
			}

			// Load vertex positions
			{
				fastgltf::Accessor& accessor = gltf.accessors[primitive.findAttribute("POSITION")->second];
				vertices.resize(vertices.size() + accessor.count);

				fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, accessor, [&](glm::vec3 v, size_t idx) {
					Vertex newvtx;
					newvtx.position = v;
					newvtx.normal = { 1, 0, 0 };
					newvtx.color = glm::vec4{ 1.f };
					newvtx.uv_x = 0;
					newvtx.uv_y = 0;
					vertices[idx + initial_vtx] = newvtx;
					});
			}

			// Load vertex normals
			auto normals = primitive.findAttribute("NORMAL");
			if (normals != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[normals->second], [&](glm::vec3 v, size_t idx) {
					vertices[initial_vtx + idx].normal = v;
					});
			}

			// Load UVs
			auto uv = primitive.findAttribute("TEXCOORD_0");
			if (uv != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[uv->second], [&](glm::vec2 v, size_t idx) {
					vertices[initial_vtx + idx].uv_x = v.x;
					vertices[initial_vtx + idx].uv_y = v.y;
					});
			}

			// Load vertex colors
			auto colors = primitive.findAttribute("COLOR_0");
			if (colors != primitive.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[colors->second], [&](glm::vec4 v, size_t idx) {
					vertices[initial_vtx + idx].color = v;
					});
			}

			newMesh.surfaces.push_back(newSurface);
		}

		// Display the vertex normals
		constexpr bool overrideColors = false;
		if (overrideColors)
		{
			for (Vertex& vtx : vertices)
			{
				vtx.color = glm::vec4{ vtx.normal, 1.f };
			}
		}

		newMesh.meshBuffers = engine->uploadMesh(indices, vertices);
		meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));

	}
	return meshes;
}
