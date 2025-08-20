#pragma once
#include "marker_tracker.h"
#include "obj_loader.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// A class for rendering 3D OBJ models onto markers using OpenCV
class OpenCVRenderer
{
public:
    OpenCVRenderer();
    ~OpenCVRenderer();

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
    bool loadModel(const std::string &modelPath);
    cv::Mat renderModel(const cv::Mat &image, const PoseMatrix4x4 &pose);
    cv::Mat convertPoseToCvMat(const PoseMatrix4x4 &pose);
    void extractPoseVectors(const PoseMatrix4x4 &pose, cv::Vec3d &rvec, cv::Vec3d &tvec);

    // Model data
    OBJMesh modelMesh;

    // OpenCV resources
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs; // Zero distortion for undistorted images

    // Window parameters
    std::string windowName;
    bool windowVisible;
    bool closeRequested;

    // Rendering parameters
    int windowWidth, windowHeight;
    cv::Vec3b modelColor;

    // Helper variables for rendering
    std::vector<cv::Point3f> modelPoints;     // 3D model points
    std::vector<std::vector<int>> modelFaces; // Model faces as indices
};
