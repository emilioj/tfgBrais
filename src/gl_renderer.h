#pragma once
#include "obj_loader.h"
#include "assimp_loader.h"
#include "marker_tracker.h"
#include <string>
#include <vector>
#include <memory>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Simple shader program for rendering 3D models
class Shader
{
public:
    Shader();
    ~Shader();

    bool initialize(const std::string &vertexSource, const std::string &fragmentSource);
    void use();

    // Uniform setters
    void setMat4(const std::string &name, const glm::mat4 &mat);
    void setVec3(const std::string &name, const glm::vec3 &value);
    void setFloat(const std::string &name, float value);

public:
    GLuint getProgram() const { return program; }

private:
    GLuint program;
};

// A class for rendering 3D OBJ models using OpenGL
class GLRenderer
{
public:
    GLRenderer();
    ~GLRenderer();

    bool initialize(int width, int height,
                    const std::string &modelPath,
                    const cv::Mat &cameraMatrix);

    // Render the model on the image using the provided pose
    cv::Mat render(const PoseMatrix4x4 &pose, const ImageData &background);

    // Render the model directly using rvec and tvec
    cv::Mat renderDirect(const double rvec[3], const double tvec[3], const ImageData &background);

    // Window management
    void showWindow(bool show);
    bool isWindowVisible() const;
    bool shouldClose() const;

private:
    // Model loading and processing
    bool loadModel(const std::string &modelPath);
    cv::Mat convertPoseToCvMat(const PoseMatrix4x4 &pose);
    glm::mat4 createModelViewMatrix(const double rvec[3], const double tvec[3]);
    glm::mat4 createProjectionMatrix(const cv::Mat &cameraMatrix, int width, int height);

    // OpenGL resources
    GLFWwindow *window;
    std::unique_ptr<Shader> shader;

    // GPU buffers
    GLuint VAO, VBO, EBO;
    GLuint textureBackground;
    GLuint framebuffer, renderedTexture;

    // Model data
    // Model data
    OBJMesh modelMesh;
    AssimpMesh assimpMesh;
    bool useAssimpMesh = false;
    GLuint modelTexture = 0;
    bool hasModelTexture = false;

    // Camera parameters
    cv::Mat cameraMatrix;
    glm::mat4 projectionMatrix;

    // Window parameters
    std::string windowName;
    bool windowVisible;
    bool initialized;
    int windowWidth, windowHeight;

    // Rendering parameters
    glm::vec3 lightPos;
    glm::vec3 modelColor;
};
