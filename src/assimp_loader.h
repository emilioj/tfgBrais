#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>

struct AssimpMesh
{
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    std::vector<unsigned int> indices;
    std::string texturePath; // empty if no texture
};

// Loads the first mesh from a file (FBX, OBJ, etc.) using Assimp
// Supports both external and embedded textures (FBX, etc.)
// Returns true on success, false on failure
bool loadMeshWithAssimp(const std::string &filename, AssimpMesh &mesh, std::string &errorMsg);
