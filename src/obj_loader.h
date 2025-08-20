#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/core.hpp>

struct OBJMesh
{
    std::vector<cv::Point3f> vertices;
    std::vector<cv::Point3f> normals;
    std::vector<cv::Point2f> texCoords;
    std::vector<unsigned int> indices;

    bool loadFromFile(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Failed to open OBJ file: " << filename << std::endl;
            return false;
        }

        std::vector<cv::Point3f> temp_vertices;
        std::vector<cv::Point3f> temp_normals;
        std::vector<cv::Point2f> temp_texcoords;

        // Temporary storage for indices
        std::vector<unsigned int> vertexIndices;
        std::vector<unsigned int> normalIndices;
        std::vector<unsigned int> texCoordIndices;

        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "v")
            {
                // Vertex position
                float x, y, z;
                iss >> x >> y >> z;
                temp_vertices.push_back(cv::Point3f(x, y, z));
            }
            else if (prefix == "vn")
            {
                // Vertex normal
                float x, y, z;
                iss >> x >> y >> z;
                temp_normals.push_back(cv::Point3f(x, y, z));
            }
            else if (prefix == "vt")
            {
                // Texture coordinate
                float x, y;
                iss >> x >> y;
                temp_texcoords.push_back(cv::Point2f(x, y));
            }
            else if (prefix == "f")
            {
                // Face
                std::string vertex1, vertex2, vertex3;
                iss >> vertex1 >> vertex2 >> vertex3;

                // Process each vertex of the face
                // Format could be: v/vt/vn, v//vn, or v/vt
                unsigned int vertexIndex, texCoordIndex, normalIndex;

                // First vertex
                std::istringstream v1(vertex1);
                std::string segment;
                std::vector<std::string> seglist;
                while (std::getline(v1, segment, '/'))
                {
                    seglist.push_back(segment);
                }

                if (seglist.size() >= 1 && !seglist[0].empty())
                    vertexIndices.push_back(std::stoi(seglist[0]) - 1);
                if (seglist.size() >= 2 && !seglist[1].empty())
                    texCoordIndices.push_back(std::stoi(seglist[1]) - 1);
                if (seglist.size() >= 3 && !seglist[2].empty())
                    normalIndices.push_back(std::stoi(seglist[2]) - 1);

                // Second vertex
                seglist.clear();
                std::istringstream v2(vertex2);
                while (std::getline(v2, segment, '/'))
                {
                    seglist.push_back(segment);
                }

                if (seglist.size() >= 1 && !seglist[0].empty())
                    vertexIndices.push_back(std::stoi(seglist[0]) - 1);
                if (seglist.size() >= 2 && !seglist[1].empty())
                    texCoordIndices.push_back(std::stoi(seglist[1]) - 1);
                if (seglist.size() >= 3 && !seglist[2].empty())
                    normalIndices.push_back(std::stoi(seglist[2]) - 1);

                // Third vertex
                seglist.clear();
                std::istringstream v3(vertex3);
                while (std::getline(v3, segment, '/'))
                {
                    seglist.push_back(segment);
                }

                if (seglist.size() >= 1 && !seglist[0].empty())
                    vertexIndices.push_back(std::stoi(seglist[0]) - 1);
                if (seglist.size() >= 2 && !seglist[1].empty())
                    texCoordIndices.push_back(std::stoi(seglist[1]) - 1);
                if (seglist.size() >= 3 && !seglist[2].empty())
                    normalIndices.push_back(std::stoi(seglist[2]) - 1);
            }
        }

        // Now that we have all the data, fill in the mesh's vectors
        // For simplicity, we'll just use the vertex indices and create a flat array
        for (unsigned int i = 0; i < vertexIndices.size(); i++)
        {
            vertices.push_back(temp_vertices[vertexIndices[i]]);
            indices.push_back(i);

            if (i < normalIndices.size())
            {
                normals.push_back(temp_normals[normalIndices[i]]);
            }
            else if (!temp_normals.empty())
            {
                // Default normal if indices don't match
                normals.push_back(temp_normals[0]);
            }

            if (i < texCoordIndices.size())
            {
                texCoords.push_back(temp_texcoords[texCoordIndices[i]]);
            }
            else if (!temp_texcoords.empty())
            {
                // Default texture coordinate if indices don't match
                texCoords.push_back(temp_texcoords[0]);
            }
        }

        std::cout << "Loaded OBJ: " << filename << std::endl;
        std::cout << "  Vertices: " << vertices.size() << std::endl;
        std::cout << "  Normals: " << normals.size() << std::endl;
        std::cout << "  TexCoords: " << texCoords.size() << std::endl;
        std::cout << "  Indices: " << indices.size() << std::endl;

        return true;
    }
};
