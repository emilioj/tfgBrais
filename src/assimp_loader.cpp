#include "assimp_loader.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>

bool loadMeshWithAssimp(const std::string &filename, AssimpMesh &mesh, std::string &errorMsg)
{
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs);
    if (!scene || !scene->HasMeshes())
    {
        errorMsg = importer.GetErrorString();
        return false;
    }
    const aiMesh *ai_mesh = scene->mMeshes[0];
    mesh.vertices.clear();
    mesh.normals.clear();
    mesh.texCoords.clear();
    mesh.indices.clear();
    // Vertices
    for (unsigned int i = 0; i < ai_mesh->mNumVertices; ++i)
    {
        mesh.vertices.emplace_back(
            ai_mesh->mVertices[i].x,
            ai_mesh->mVertices[i].y,
            ai_mesh->mVertices[i].z);
        if (ai_mesh->HasNormals())
        {
            mesh.normals.emplace_back(
                ai_mesh->mNormals[i].x,
                ai_mesh->mNormals[i].y,
                ai_mesh->mNormals[i].z);
        }
        if (ai_mesh->HasTextureCoords(0))
        {
            mesh.texCoords.emplace_back(
                ai_mesh->mTextureCoords[0][i].x,
                ai_mesh->mTextureCoords[0][i].y);
        }
    }
    // Indices
    for (unsigned int i = 0; i < ai_mesh->mNumFaces; ++i)
    {
        const aiFace &face = ai_mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; ++j)
        {
            mesh.indices.push_back(face.mIndices[j]);
        }
    }
    // Texture (diffuse)
    mesh.texturePath.clear();
    if (scene->HasMaterials())
    {
        aiMaterial *material = scene->mMaterials[ai_mesh->mMaterialIndex];
        std::cout << "Material index: " << ai_mesh->mMaterialIndex << std::endl;

        // Print all texture types and their counts
        std::cout << "Texture counts by type:" << std::endl;
        for (int t = aiTextureType_NONE; t <= aiTextureType_UNKNOWN; ++t)
        {
            int count = material->GetTextureCount((aiTextureType)t);
            if (count > 0)
            {
                std::cout << "  Type " << t << ": " << count << std::endl;
                for (int i = 0; i < count; ++i)
                {
                    aiString str;
                    if (material->GetTexture((aiTextureType)t, i, &str) == AI_SUCCESS)
                    {
                        std::cout << "    Texture[" << i << "]: " << str.C_Str() << std::endl;
                    }
                }
            }
        }

        // Print all material property keys
        std::cout << "Material properties (key, type, index):" << std::endl;
        for (unsigned int i = 0; i < material->mNumProperties; ++i)
        {
            aiMaterialProperty *prop = material->mProperties[i];
            std::cout << "  Key: " << prop->mKey.C_Str() << ", Type: " << (int)prop->mType << ", Index: " << prop->mIndex << std::endl;
        }

        // Keep original diffuse texture logic for compatibility
        std::cout << "Material has " << material->GetTextureCount(aiTextureType_DIFFUSE) << " diffuse textures" << std::endl;
        if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
        {
            aiString str;
            if (material->GetTexture(aiTextureType_DIFFUSE, 0, &str) == AI_SUCCESS)
            {
                std::string texPath = str.C_Str();
                std::cout << "Texture path: " << texPath << std::endl;
                mesh.texturePath = texPath;
            }
            else
            {
                std::cout << "Failed to get diffuse texture path from material." << std::endl;
            }
        }
        else
        {
            std::cout << "No diffuse texture found in material." << std::endl;
        }
    }
    return true;
}
