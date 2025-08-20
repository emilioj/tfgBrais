#include "obj_renderer.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

OpenCVRenderer::OpenCVRenderer() : windowVisible(false),
                                   closeRequested(false),
                                   windowName("AR Model Renderer"),
                                   modelColor(cv::Vec3b(204, 204, 204)) // Light gray color in BGR
{
    // Initialize distortion coefficients with zeros (for undistorted images)
    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
}

OpenCVRenderer::~OpenCVRenderer()
{
    if (windowVisible)
    {
        cv::destroyWindow(windowName);
    }
}

bool OpenCVRenderer::initialize(int width, int height,
                                const std::string &modelPath,
                                const cv::Mat &camMatrix)
{
    windowWidth = width;
    windowHeight = height;
    cameraMatrix = camMatrix.clone(); // Make a copy of the camera matrix

    return loadModel(modelPath);
}

bool OpenCVRenderer::loadModel(const std::string &modelPath)
{
    // Try to load the OBJ file from the provided path first
    if (!modelMesh.loadFromFile(modelPath))
    {
        // Try to get the parent directory of the model path
        std::string parentDir;
        size_t lastSlash = modelPath.find_last_of("/\\");
        if (lastSlash != std::string::npos)
        {
            parentDir = modelPath.substr(0, lastSlash);
            // Go up one more level
            lastSlash = parentDir.find_last_of("/\\");
            if (lastSlash != std::string::npos)
            {
                parentDir = parentDir.substr(0, lastSlash);
            }
        }

        // If not found, try the default cube as fallback
        std::string fallbackPath = parentDir + "/cubo.obj";
        if (!modelMesh.loadFromFile(fallbackPath))
        {
            std::cerr << "Failed to load model from: " << modelPath << std::endl;
            std::cerr << "Also tried fallback: " << fallbackPath << std::endl;
            return false;
        }
        else
        {
            std::cout << "Using fallback model: " << fallbackPath << std::endl;
        }
    }
    else
    {
        std::cout << "Successfully loaded model: " << modelPath << std::endl;
    }

    // Convert OBJMesh vertices to OpenCV 3D points format
    modelPoints.clear();
    for (const auto &vertex : modelMesh.vertices)
    {
        modelPoints.push_back(cv::Point3f(vertex.x, vertex.y, vertex.z));
    }

    // Convert indices to faces format for OpenCV drawing
    modelFaces.clear();
    for (size_t i = 0; i < modelMesh.indices.size(); i += 3)
    {
        if (i + 2 < modelMesh.indices.size())
        {
            std::vector<int> face = {
                static_cast<int>(modelMesh.indices[i]),
                static_cast<int>(modelMesh.indices[i + 1]),
                static_cast<int>(modelMesh.indices[i + 2])};
            modelFaces.push_back(face);
        }
    }

    std::cout << "Model loaded successfully with " << modelPoints.size()
              << " vertices and " << modelFaces.size() << " faces" << std::endl;

    return true;
}

cv::Mat OpenCVRenderer::render(const PoseMatrix4x4 &pose, const ImageData &background)
{
    // Convert ImageData to cv::Mat
    cv::Mat image;
    if (background.isEmpty())
    {
        // Create a blank image if no background is provided
        image = cv::Mat::zeros(windowHeight, windowWidth, CV_8UC3);
    }
    else
    {
        // Convert the ImageData to cv::Mat
        image = cv::Mat(background.height, background.width,
                        CV_8UC(background.channels),
                        const_cast<unsigned char *>(background.data.data()));

        // Convert to 3-channel BGR if needed
        if (image.channels() != 3)
        {
            cv::cvtColor(image, image, image.channels() == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR);
        }

        // Resize if dimensions don't match
        if (image.rows != windowHeight || image.cols != windowWidth)
        {
            cv::resize(image, image, cv::Size(windowWidth, windowHeight));
        }
    }

    // Render the 3D model onto the image
    cv::Mat result = renderModel(image.clone(), pose);

    // Display the result if window is visible
    if (windowVisible)
    {
        cv::imshow(windowName, result);
        int key = cv::waitKey(1);
        if (key == 27)
        { // ESC key
            closeRequested = true;
        }
    }

    return result;
}

cv::Mat OpenCVRenderer::renderDirect(const double rvec[3], const double tvec[3], const ImageData &background)
{
    // Convert ImageData to cv::Mat
    cv::Mat image;
    if (background.isEmpty())
    {
        // Create a blank image if no background is provided
        image = cv::Mat::zeros(windowHeight, windowWidth, CV_8UC3);
    }
    else
    {
        // Convert the ImageData to cv::Mat
        image = cv::Mat(background.height, background.width,
                        CV_8UC(background.channels),
                        const_cast<unsigned char *>(background.data.data()));

        // Convert to 3-channel BGR if needed
        if (image.channels() != 3)
        {
            cv::cvtColor(image, image, image.channels() == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR);
        }

        // Resize if dimensions don't match
        if (image.rows != windowHeight || image.cols != windowWidth)
        {
            cv::resize(image, image, cv::Size(windowWidth, windowHeight));
        }
    }

    cv::Mat result = image.clone();

    // Convert arrays to cv::Vec3d
    cv::Vec3d rvecVec(rvec[0], rvec[1], rvec[2]);
    cv::Vec3d tvecVec(tvec[0], tvec[1], tvec[2]);

    // Scale the model to make it much smaller
    std::vector<cv::Point3f> scaledModelPoints;
    float scale = 0.001f; // Drastically reduced scale factor to match renderModel
    for (const auto &point : modelPoints)
    {
        scaledModelPoints.push_back(cv::Point3f(point.x * scale, point.y * scale, point.z * scale));
    }

    // Project model points to 2D
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(scaledModelPoints, rvecVec, tvecVec, cameraMatrix, distCoeffs, projectedPoints);

    // Draw the model faces
    for (const auto &face : modelFaces)
    {
        if (face.size() >= 3)
        {
            std::vector<cv::Point> facePoints;
            for (int idx : face)
            {
                if (idx < projectedPoints.size())
                {
                    facePoints.push_back(cv::Point(projectedPoints[idx]));
                }
            }

            if (facePoints.size() >= 3)
            {
                // Draw filled triangle
                cv::fillConvexPoly(result, facePoints, modelColor);

                // Draw wireframe
                for (size_t i = 0; i < facePoints.size(); i++)
                {
                    cv::line(result, facePoints[i], facePoints[(i + 1) % facePoints.size()],
                             cv::Scalar(255, 255, 255), 1);
                }
            }
        }
    }

    // Display the result if window is visible
    if (windowVisible)
    {
        cv::imshow(windowName, result);
        int key = cv::waitKey(1);
        if (key == 27)
        { // ESC key
            closeRequested = true;
        }
    }

    return result;
}

void OpenCVRenderer::extractPoseVectors(const PoseMatrix4x4 &pose, cv::Vec3d &rvec, cv::Vec3d &tvec)
{
    // Convert pose to OpenCV format
    cv::Mat poseMat = convertPoseToCvMat(pose);

    // Extract rotation matrix
    cv::Mat rotMat = poseMat(cv::Range(0, 3), cv::Range(0, 3));

    // Extract translation vector
    tvec[0] = poseMat.at<double>(0, 3);
    tvec[1] = poseMat.at<double>(1, 3);
    tvec[2] = poseMat.at<double>(2, 3);

    // Move the model slightly away from camera if needed
    // tvec[2] += 0.05; // Uncomment to adjust Z distance

    // Convert rotation matrix to rotation vector
    cv::Rodrigues(rotMat, rvec);
}

cv::Mat OpenCVRenderer::renderModel(const cv::Mat &image, const PoseMatrix4x4 &pose)
{
    cv::Mat result = image.clone();

    // Get rotation and translation vectors
    cv::Vec3d rvec, tvec;
    extractPoseVectors(pose, rvec, tvec);

    // Scale the model to make it much smaller
    std::vector<cv::Point3f> scaledModelPoints;
    float scale = 0.0001f; // Drastically reduced scale factor to make the model much smaller
    for (const auto &point : modelPoints)
    {
        scaledModelPoints.push_back(cv::Point3f(point.x * scale, point.y * scale, point.z * scale));
    }

    // Project model points to 2D
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(scaledModelPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    // Draw the model faces
    for (const auto &face : modelFaces)
    {
        if (face.size() >= 3)
        {
            std::vector<cv::Point> facePoints;
            for (int idx : face)
            {
                if (idx < projectedPoints.size())
                {
                    facePoints.push_back(cv::Point(projectedPoints[idx]));
                }
            }

            if (facePoints.size() >= 3)
            {
                // Draw filled triangle
                cv::fillConvexPoly(result, facePoints, modelColor);

                // Draw wireframe
                for (size_t i = 0; i < facePoints.size(); i++)
                {
                    cv::line(result, facePoints[i], facePoints[(i + 1) % facePoints.size()],
                             cv::Scalar(255, 255, 255), 1);
                }
            }
        }
    }

    return result;
}

cv::Mat OpenCVRenderer::convertPoseToCvMat(const PoseMatrix4x4 &pose)
{
    cv::Mat poseMat(4, 4, CV_64F);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            poseMat.at<double>(i, j) = pose.m[i * 4 + j];
        }
    }
    return poseMat;
}

void OpenCVRenderer::showWindow(bool show)
{
    windowVisible = show;
    if (show)
    {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, windowWidth, windowHeight);
    }
    else
    {
        cv::destroyWindow(windowName);
    }
}

bool OpenCVRenderer::isWindowVisible() const
{
    return windowVisible;
}

bool OpenCVRenderer::shouldClose() const
{
    return closeRequested;
}
