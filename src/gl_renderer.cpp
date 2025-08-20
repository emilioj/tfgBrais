#include "gl_renderer.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "assimp_loader.h"
#include <opencv2/imgcodecs.hpp>

// Default vertex shader
const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    // Apply normal matrix to preserve normals when scaling is non-uniform
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    TexCoord = aTexCoord;
    
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

// Default fragment shader
const char *fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 lightPos;
uniform vec3 modelColor;
uniform sampler2D texture1;
uniform bool hasTexture;

void main()
{
    // Ambient lighting
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
    
    // Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
    
    // Specular lighting
    float specularStrength = 0.5;
    vec3 viewDir = normalize(-FragPos); // Camera is at origin in view space
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
    
    // Final color
    vec3 baseColor = hasTexture ? texture(texture1, TexCoord).rgb : modelColor;
    vec3 result = (ambient + diffuse + specular) * baseColor;
    FragColor = vec4(result, 1.0);
}
)";

// Shader implementation
Shader::Shader() : program(0) {}

Shader::~Shader()
{
    if (program > 0)
    {
        glDeleteProgram(program);
    }
}

bool Shader::initialize(const std::string &vertexSource, const std::string &fragmentSource)
{
    // Create vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const char *vShaderCode = vertexSource.c_str();
    glShaderSource(vertexShader, 1, &vShaderCode, NULL);
    glCompileShader(vertexShader);

    // Check for vertex shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
                  << infoLog << std::endl;
        return false;
    }

    // Create fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    const char *fShaderCode = fragmentSource.c_str();
    glShaderSource(fragmentShader, 1, &fShaderCode, NULL);
    glCompileShader(fragmentShader);

    // Check for fragment shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n"
                  << infoLog << std::endl;
        return false;
    }

    // Link shaders
    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Check for linking errors
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
                  << infoLog << std::endl;
        return false;
    }

    // Delete shaders (they're linked into the program now)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return true;
}

void Shader::use()
{
    glUseProgram(program);
}

void Shader::setMat4(const std::string &name, const glm::mat4 &mat)
{
    glUniformMatrix4fv(glGetUniformLocation(program, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

void Shader::setVec3(const std::string &name, const glm::vec3 &value)
{
    glUniform3fv(glGetUniformLocation(program, name.c_str()), 1, glm::value_ptr(value));
}

void Shader::setFloat(const std::string &name, float value)
{
    glUniform1f(glGetUniformLocation(program, name.c_str()), value);
}

// GLRenderer implementation
GLRenderer::GLRenderer() : window(nullptr), VAO(0), VBO(0), EBO(0),
                           textureBackground(0), framebuffer(0), renderedTexture(0),
                           windowVisible(false), initialized(false),
                           windowWidth(0), windowHeight(0),
                           windowName("OpenGL AR Renderer"),
                           lightPos(0.5f, 1.0f, 0.3f),
                           modelColor(0.8f, 0.8f, 0.8f) {}

GLRenderer::~GLRenderer()
{
    // Clean up OpenGL resources
    if (initialized)
    {
        if (VAO != 0)
            glDeleteVertexArrays(1, &VAO);
        if (VBO != 0)
            glDeleteBuffers(1, &VBO);
        if (EBO != 0)
            glDeleteBuffers(1, &EBO);
        if (textureBackground != 0)
            glDeleteTextures(1, &textureBackground);
        if (renderedTexture != 0)
            glDeleteTextures(1, &renderedTexture);
        if (framebuffer != 0)
            glDeleteFramebuffers(1, &framebuffer);
    }

    // Close window if it's still open
    if (window)
    {
        glfwDestroyWindow(window);
    }

    // Terminate GLFW if we initialized it
    if (initialized)
    {
        glfwTerminate();
    }
}

bool GLRenderer::initialize(int width, int height, const std::string &modelPath, const cv::Mat &camMatrix)
{
    windowWidth = width;
    windowHeight = height;
    cameraMatrix = camMatrix.clone();

    std::cout << "Initializing OpenGL renderer with size: " << width << "x" << height << std::endl;

    // Initialize GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_VISIBLE, GL_TRUE); // Make window visible for debugging

    // Create a window
    window = glfwCreateWindow(windowWidth, windowHeight, windowName.c_str(), NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(err) << std::endl;
        return false;
    }

    // Clear any errors that might have occurred during GLEW initialization
    while (glGetError() != GL_NO_ERROR)
        ;

    // Print OpenGL version info
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

    // Create and compile our shader program
    shader = std::make_unique<Shader>();
    if (!shader->initialize(vertexShaderSource, fragmentShaderSource))
    {
        std::cerr << "Failed to initialize shader program" << std::endl;
        return false;
    }

    // Set up frame buffer for off-screen rendering
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    // Create texture for rendering to
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderedTexture, 0);

    // Create a renderbuffer object for depth and stencil attachments
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, windowWidth, windowHeight);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Attach the renderbuffer to the framebuffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    // Create texture for background image
    glGenTextures(1, &textureBackground);
    glBindTexture(GL_TEXTURE_2D, textureBackground);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Check framebuffer status
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "Framebuffer is not complete! Status: " << status << std::endl;
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Create projection matrix based on camera intrinsics
    projectionMatrix = createProjectionMatrix(cameraMatrix, windowWidth, windowHeight);

    // Set up OpenGL viewport
    glViewport(0, 0, windowWidth, windowHeight);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Enable face culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    // Load the model
    if (!loadModel(modelPath))
    {
        std::cerr << "Failed to load model: " << modelPath << std::endl;
        return false;
    }

    // Test drawing a simple colored cube to verify GL setup
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0.2f, 0.3f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader->use();
    glm::mat4 model = glm::mat4(1.0f);
    shader->setMat4("projection", projectionMatrix);
    shader->setMat4("view", glm::mat4(1.0f));
    shader->setMat4("model", model);
    shader->setVec3("lightPos", lightPos);
    shader->setVec3("modelColor", glm::vec3(0.9f, 0.1f, 0.1f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, modelMesh.indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // Read the test render
    cv::Mat testRender(windowHeight, windowWidth, CV_8UC3);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_BGR, GL_UNSIGNED_BYTE, testRender.data);
    cv::flip(testRender, testRender, 0);

    // Save the test render to verify OpenGL is working
    cv::imwrite("opengl_test_render.png", testRender);
    std::cout << "Saved test render to opengl_test_render.png" << std::endl;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    initialized = true;
    std::cout << "OpenGL renderer initialized successfully" << std::endl;
    return true;
}

glm::mat4 GLRenderer::createProjectionMatrix(const cv::Mat &cameraMatrix, int width, int height)
{
    // Get camera intrinsic parameters
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    // Set up projection matrix for OpenGL
    // Note: OpenGL uses a right-handed coordinate system
    float near = 0.1f;
    float far = 100.0f;

    // Create OpenGL-style projection matrix
    glm::mat4 projMatrix = glm::mat4(1.0f);

    // Set up the projection matrix with camera intrinsics
    projMatrix[0][0] = 2.0f * fx / width;
    projMatrix[1][1] = 2.0f * fy / height;
    projMatrix[2][0] = 1.0f - 2.0f * cx / width;
    projMatrix[2][1] = 2.0f * cy / height - 1.0f;
    projMatrix[2][2] = -(far + near) / (far - near);
    projMatrix[2][3] = -1.0f;
    projMatrix[3][2] = -2.0f * far * near / (far - near);
    projMatrix[3][3] = 0.0f;

    return projMatrix;
}

glm::mat4 GLRenderer::createModelViewMatrix(const double rvec[3], const double tvec[3])
{
    // Convert rotation vector to rotation matrix
    cv::Mat rotMat;
    cv::Rodrigues(cv::Vec3d(rvec), rotMat);

    // Create model-view matrix (transforms from model space to camera space)
    glm::mat4 modelView = glm::mat4(1.0f);

    // Copy rotation matrix
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            modelView[i][j] = rotMat.at<double>(j, i); // Note: OpenGL is column-major
        }
    }

    // Copy translation vector
    modelView[3][0] = tvec[0];
    modelView[3][1] = tvec[1];
    modelView[3][2] = tvec[2];

    // Apply OpenGL coordinate system adjustments
    // OpenCV: right-handed, Y down
    // OpenGL: right-handed, Y up
    glm::mat4 coordinateSystemAdj = glm::mat4(1.0f);
    coordinateSystemAdj[1][1] = -1.0f; // Flip Y
    coordinateSystemAdj[2][2] = -1.0f; // Flip Z

    return coordinateSystemAdj * modelView;
}

bool GLRenderer::loadModel(const std::string &modelPath)
{
    // Determine file extension
    std::string ext = modelPath.substr(modelPath.find_last_of('.') + 1);
    for (auto &c : ext)
        c = tolower(c);
    useAssimpMesh = (ext == "fbx" || ext == "obj");

    if (useAssimpMesh)
    {
        std::string errorMsg;
        if (!loadMeshWithAssimp(modelPath, assimpMesh, errorMsg))
        {
            std::cerr << "Failed to load model with Assimp: " << errorMsg << std::endl;
            return false;
        }
        std::cout << "Model loaded with Assimp: " << modelPath << std::endl;
        std::cout << "  Vertices: " << assimpMesh.vertices.size() << std::endl;
        std::cout << "  Normals: " << assimpMesh.normals.size() << std::endl;
        std::cout << "  TexCoords: " << assimpMesh.texCoords.size() << std::endl;
        std::cout << "  Indices: " << assimpMesh.indices.size() << std::endl;

        // Create vertex array object
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        // Create vertex buffer
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        // Combine vertex data (position, normal, texcoord) into a single buffer
        std::vector<float> vertexData;
        for (size_t i = 0; i < assimpMesh.vertices.size(); i++)
        {
            // Position
            vertexData.push_back(assimpMesh.vertices[i].x);
            vertexData.push_back(assimpMesh.vertices[i].y);
            vertexData.push_back(assimpMesh.vertices[i].z);
            // Normal
            if (i < assimpMesh.normals.size())
            {
                vertexData.push_back(assimpMesh.normals[i].x);
                vertexData.push_back(assimpMesh.normals[i].y);
                vertexData.push_back(assimpMesh.normals[i].z);
            }
            else
            {
                vertexData.push_back(0.0f);
                vertexData.push_back(1.0f);
                vertexData.push_back(0.0f);
            }
            // TexCoord
            if (i < assimpMesh.texCoords.size())
            {
                vertexData.push_back(assimpMesh.texCoords[i].x);
                vertexData.push_back(assimpMesh.texCoords[i].y);
            }
            else
            {
                vertexData.push_back(0.0f);
                vertexData.push_back(0.0f);
            }
        }

        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

        // Create element buffer
        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        if (!assimpMesh.indices.empty())
        {
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, assimpMesh.indices.size() * sizeof(unsigned int),
                         assimpMesh.indices.data(), GL_STATIC_DRAW);
        }

        // Set up vertex attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // Load texture if available
        hasModelTexture = false;
        if (!assimpMesh.texturePath.empty())
        {
            std::string texPath = assimpMesh.texturePath;
            // If relative path, prepend model directory
            size_t lastSlash = modelPath.find_last_of("/\\");
            if (lastSlash != std::string::npos)
            {
                std::string modelDir = modelPath.substr(0, lastSlash + 1);
                texPath = modelDir + texPath;
            }
            cv::Mat texImg = cv::imread(texPath, cv::IMREAD_COLOR);
            if (!texImg.empty())
            {
                glGenTextures(1, &modelTexture);
                glBindTexture(GL_TEXTURE_2D, modelTexture);
                cv::cvtColor(texImg, texImg, cv::COLOR_BGR2RGB);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texImg.cols, texImg.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, texImg.data);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                hasModelTexture = true;
                std::cout << "Loaded model texture: " << texPath << std::endl;
            }
            else
            {
                std::cerr << "Failed to load model texture: " << texPath << std::endl;
            }
        }
        glBindVertexArray(0);
        return true;
    }
    // Fallback: use OBJ loader for .obj
    if (!modelMesh.loadFromFile(modelPath))
    {
        std::cerr << "Failed to load OBJ file: " << modelPath << std::endl;
        return false;
    }
    std::cout << "Model loaded: " << modelPath << std::endl;
    std::cout << "  Vertices: " << modelMesh.vertices.size() << std::endl;
    std::cout << "  Normals: " << modelMesh.normals.size() << std::endl;
    std::cout << "  TexCoords: " << modelMesh.texCoords.size() << std::endl;
    std::cout << "  Indices: " << modelMesh.indices.size() << std::endl;
    if (modelMesh.vertices.empty())
    {
        std::cerr << "No vertices found in the model!" << std::endl;
        return false;
    }
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    std::vector<float> vertexData;
    if (!modelMesh.indices.empty())
    {
        for (size_t i = 0; i < modelMesh.vertices.size(); i++)
        {
            vertexData.push_back(modelMesh.vertices[i].x);
            vertexData.push_back(modelMesh.vertices[i].y);
            vertexData.push_back(modelMesh.vertices[i].z);
            if (i < modelMesh.normals.size())
            {
                vertexData.push_back(modelMesh.normals[i].x);
                vertexData.push_back(modelMesh.normals[i].y);
                vertexData.push_back(modelMesh.normals[i].z);
            }
            else
            {
                vertexData.push_back(0.0f);
                vertexData.push_back(1.0f);
                vertexData.push_back(0.0f);
            }
            if (i < modelMesh.texCoords.size())
            {
                vertexData.push_back(modelMesh.texCoords[i].x);
                vertexData.push_back(modelMesh.texCoords[i].y);
            }
            else
            {
                vertexData.push_back(0.0f);
                vertexData.push_back(0.0f);
            }
        }
    }
    else
    {
        // If we don't have indices, create a simple cube for testing
        std::cout << "No indices found, creating a simple cube for testing" << std::endl;
        float cube_vertices[] = {
            // positions          // normals           // texture coords
            -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
            0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f,
            0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,
            0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,
            -0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,
            -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
            -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
            0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
            0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
            -0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
            -0.5f, 0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            -0.5f, -0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
            0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
            0.5f, 0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
            -0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,
            0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f,
            0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
            0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
            -0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
            -0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,
            -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
            0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
            0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            -0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
            -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
        for (int i = 0; i < 36 * 8; i++)
        {
            vertexData.push_back(cube_vertices[i]);
        }
        for (int i = 0; i < 36; i++)
        {
            modelMesh.indices.push_back(i);
        }
    }
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    if (!modelMesh.indices.empty())
    {
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, modelMesh.indices.size() * sizeof(unsigned int),
                     modelMesh.indices.data(), GL_STATIC_DRAW);
    }
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return true;
}

cv::Mat GLRenderer::render(const PoseMatrix4x4 &pose, const ImageData &background)
{
    // Extract rotation and translation vectors from the pose matrix
    cv::Mat cvPose = convertPoseToCvMat(pose);
    cv::Mat rotMat(3, 3, CV_64F);
    cv::Mat tvec(3, 1, CV_64F);

    // Extract rotation and translation components
    for (int i = 0; i < 3; i++)
    {
        tvec.at<double>(i) = cvPose.at<double>(i, 3);
        for (int j = 0; j < 3; j++)
        {
            rotMat.at<double>(i, j) = cvPose.at<double>(i, j);
        }
    }

    // Convert rotation matrix to rotation vector
    cv::Vec3d rvec;
    cv::Rodrigues(rotMat, rvec);

    // Call renderDirect with the extracted vectors
    double rvecArray[3] = {rvec[0], rvec[1], rvec[2]};
    double tvecArray[3] = {tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};

    return renderDirect(rvecArray, tvecArray, background);
}

cv::Mat GLRenderer::renderDirect(const double rvec[3], const double tvec[3], const ImageData &background)
{
    if (!initialized)
    {
        std::cerr << "Renderer not initialized!" << std::endl;
        return cv::Mat();
    }

    // Debug print
    std::cout << "Rendering with rvec: [" << rvec[0] << ", " << rvec[1] << ", " << rvec[2]
              << "], tvec: [" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << "]" << std::endl;

    // Convert ImageData to cv::Mat
    cv::Mat bgImage;
    if (!background.isEmpty())
    {
        bgImage = cv::Mat(background.height, background.width, CV_8UC(background.channels),
                          const_cast<unsigned char *>(background.data.data()));
        std::cout << "Background image dimensions: " << bgImage.cols << "x" << bgImage.rows << std::endl;
    }
    else
    {
        // Create a black background if no image is provided
        bgImage = cv::Mat(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    // Bind framebuffer for off-screen rendering
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // --- Render background as textured quad ---
    static GLuint quadVAO = 0, quadVBO = 0;
    static GLuint bgShaderProgram = 0;
    if (quadVAO == 0)
    {
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f};
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
        glBindVertexArray(0);
    }
    if (bgShaderProgram == 0)
    {
        const char *bgVert = R"(
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main() {
                TexCoord = aTexCoord;
                gl_Position = vec4(aPos, 0.0, 1.0);
            }
        )";
        const char *bgFrag = R"(
            #version 330 core
            in vec2 TexCoord;
            out vec4 FragColor;
            uniform sampler2D bgTexture;
            void main() {
                FragColor = texture(bgTexture, TexCoord);
            }
        )";
        GLuint v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &bgVert, NULL);
        glCompileShader(v);
        GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &bgFrag, NULL);
        glCompileShader(f);
        bgShaderProgram = glCreateProgram();
        glAttachShader(bgShaderProgram, v);
        glAttachShader(bgShaderProgram, f);
        glLinkProgram(bgShaderProgram);
        glDeleteShader(v);
        glDeleteShader(f);
    }
    // Upload background image as texture
    glBindTexture(GL_TEXTURE_2D, textureBackground);
    cv::Mat flippedBg;
    cv::flip(bgImage, flippedBg, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, flippedBg.cols, flippedBg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, flippedBg.data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Draw quad
    glDisable(GL_DEPTH_TEST);
    glUseProgram(bgShaderProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureBackground);
    glUniform1i(glGetUniformLocation(bgShaderProgram, "bgTexture"), 0);
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);

    // --- Render 3D model on top ---
    glm::mat4 modelView = createModelViewMatrix(rvec, tvec);
    float scaleFactor = 0.001f;
    glm::mat4 scaledModel = glm::scale(modelView, glm::vec3(scaleFactor, scaleFactor, scaleFactor));
    shader->use();
    shader->setMat4("projection", projectionMatrix);
    shader->setMat4("view", glm::mat4(1.0f));
    shader->setMat4("model", scaledModel);
    shader->setVec3("lightPos", lightPos);

    // Bind model texture if available and set shader uniform
    if (useAssimpMesh && hasModelTexture)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, modelTexture);
    }
    shader->setVec3("modelColor", glm::vec3(1.0f, 0.2f, 0.2f));
    shader->use();
    // Always set texture1 uniform to 0 (GL_TEXTURE0)
    GLint texLoc = glGetUniformLocation(shader->getProgram(), "texture1");
    if (texLoc >= 0)
    {
        glUniform1i(texLoc, 0);
    }
    else
    {
        std::cerr << "Warning: 'texture1' uniform not found in shader!" << std::endl;
    }
    shader->setFloat("hasTexture", (useAssimpMesh && hasModelTexture) ? 1.0f : 0.0f);
    glBindVertexArray(VAO);
    if (VAO == 0)
    {
        std::cerr << "VAO is invalid (0)!" << std::endl;
    }
    else
    {
        int drawCount = useAssimpMesh ? assimpMesh.indices.size() : modelMesh.indices.size();
        std::cout << "Drawing " << drawCount << " vertices..." << std::endl;
        glDrawElements(GL_TRIANGLES, drawCount, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);

    // Read the rendered image back from GPU
    cv::Mat renderedImage(windowHeight, windowWidth, CV_8UC3);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_BGR, GL_UNSIGNED_BYTE, renderedImage.data);
    cv::flip(renderedImage, renderedImage, 0);
    cv::imshow("OpenGL Render", renderedImage);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (windowVisible)
    {
        glfwMakeContextCurrent(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawPixels(renderedImage.cols, renderedImage.rows, GL_BGR, GL_UNSIGNED_BYTE, renderedImage.data);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    return renderedImage;
}

cv::Mat GLRenderer::convertPoseToCvMat(const PoseMatrix4x4 &pose)
{
    cv::Mat cvPose(4, 4, CV_64F);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            cvPose.at<double>(i, j) = pose.m[i * 4 + j];
        }
    }
    return cvPose;
}

void GLRenderer::showWindow(bool show)
{
    windowVisible = show;
    if (window)
    {
        if (show)
        {
            glfwShowWindow(window);
        }
        else
        {
            glfwHideWindow(window);
        }
    }
}

bool GLRenderer::isWindowVisible() const
{
    return windowVisible;
}

bool GLRenderer::shouldClose() const
{
    return window && glfwWindowShouldClose(window);
}
