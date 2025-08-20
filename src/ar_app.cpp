#include "marker_tracker.h"
#include "obj_renderer.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <windows.h>
#include <filesystem>

static double globalRvec[3] = {0.0, 0.0, 0.0};
static double globalTvec[3] = {0.0, 0.0, 0.0};
static ImageData globalImage;
static std::mutex poseMutex;
static std::mutex imageMutex;
static std::atomic<bool> running(true);
static std::atomic<bool> markerDetected(false);

// Helper function to get the absolute path to project files
std::string getProjectPath()
{
    // Get the current executable path
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string execPath(buffer);

    // Remove the executable name and go up one level from Debug directory
    size_t lastSlash = execPath.find_last_of("\\");
    if (lastSlash != std::string::npos)
    {
        execPath = execPath.substr(0, lastSlash); // Remove executable name
        lastSlash = execPath.find_last_of("\\");
        if (lastSlash != std::string::npos)
        {
            execPath = execPath.substr(0, lastSlash); // Remove Debug directory
            lastSlash = execPath.find_last_of("\\");
            if (lastSlash != std::string::npos)
            {
                execPath = execPath.substr(0, lastSlash); // Remove build directory
            }
        }
    }
    return execPath;
}

// Helper function to convert our ImageData to cv::Mat for visualization
cv::Mat imageDataToMat(const ImageData &imageData)
{
    if (imageData.isEmpty())
    {
        return cv::Mat();
    }

    cv::Mat result(imageData.height, imageData.width,
                   CV_8UC(imageData.channels),
                   const_cast<unsigned char *>(imageData.data.data()));
    return result.clone(); // Clone to ensure we have our own copy of the data
}

void captureThreadFunc(bool showViz,
                       double markerSideLen,
                       double markerGapLen,
                       const std::string &calibFile,
                       const std::string &boardsDir)
{
    while (running)
    {
        ImageData undistortedImage;
        double rvec[3] = {0.0, 0.0, 0.0};
        double tvec[3] = {0.0, 0.0, 0.0};

        // Get the pose vectors directly
        bool detected = getCubePoseVectors(
            showViz,
            markerSideLen,
            markerGapLen,
            calibFile,
            boardsDir,
            undistortedImage,
            rvec,
            tvec);

        // Update global pose vectors
        {
            std::lock_guard<std::mutex> lock(poseMutex);
            for (int i = 0; i < 3; i++)
            {
                globalRvec[i] = rvec[i];
                globalTvec[i] = tvec[i];
            }
            markerDetected = detected;
        }

        // Update global image
        {
            std::lock_guard<std::mutex> lock(imageMutex);
            globalImage = undistortedImage;
        }

        // Display the undistorted image if available (for debugging)
        if (!undistortedImage.isEmpty() && showViz)
        {
            // Convert to OpenCV Mat for visualization only
            cv::Mat displayImage = imageDataToMat(undistortedImage);
            if (!displayImage.empty())
            {
                // The undistorted image is now shown in a separate window
                cv::imshow("Undistorted", displayImage);
                cv::waitKey(1);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps
    }
}

int main()
{
    // Get the project root path
    std::string projectPath = getProjectPath();
    std::cout << "Project path: " << projectPath << std::endl;

    // Create full paths
    std::string calibrationPath = projectPath + "/data/camera_calibration/camera_calibration.txt";
    std::string markersPath = projectPath + "/data/markers/runtime";
    std::string modelsPath = projectPath + "/data/models/skull/skull.obj";

    std::cout << "Using calibration file: " << calibrationPath << std::endl;
    std::cout << "Using markers directory: " << markersPath << std::endl;
    std::cout << "Using model file: " << modelsPath << std::endl;

    // Start the tracking thread
    std::thread captureThread(captureThreadFunc, true, 0.015, 0.007, calibrationPath, markersPath);

    // Load camera matrix for renderer
    cv::Mat cameraMatrix;
    cv::FileStorage fs(calibrationPath, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open camera calibration file: " << calibrationPath << std::endl;
        running = false;
        captureThread.join();
        return 1;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs.release();

    // Initialize the OpenCV renderer
    OpenCVRenderer renderer;

    // Get camera resolution (you might want to adapt this to your camera)
    int width = 640;
    int height = 480;

    // Initialize renderer with the OBJ model data
    if (!renderer.initialize(width, height, modelsPath, cameraMatrix))
    {
        std::cerr << "Failed to initialize OpenCV renderer!" << std::endl;
        running = false;
        captureThread.join();
        return 1;
    }

    // Make the renderer window visible
    renderer.showWindow(true);

    // Main rendering loop
    while (!renderer.shouldClose())
    {
        double currentRvec[3], currentTvec[3];
        bool isDetected = false;
        ImageData currentImage;

        // Get the latest pose vectors and image
        {
            std::lock_guard<std::mutex> poseLock(poseMutex);
            for (int i = 0; i < 3; i++)
            {
                currentRvec[i] = globalRvec[i];
                currentTvec[i] = globalTvec[i];
            }
            isDetected = markerDetected;
        }

        {
            std::lock_guard<std::mutex> imageLock(imageMutex);
            currentImage = globalImage;
        }

        // Render the AR scene
        cv::Mat renderedFrame;

        // Only render if we have a valid marker detection and image
        if (isDetected && !currentImage.isEmpty())
        {
            // Use renderDirect to render using the rotation and translation vectors directly
            renderedFrame = renderer.renderDirect(currentRvec, currentTvec, currentImage);
        }
        else if (!currentImage.isEmpty())
        {
            // If no marker is detected but we have an image, just show the camera feed
            renderedFrame = imageDataToMat(currentImage);
        }
        else
        {
            // If we have nothing, create a blank image
            renderedFrame = cv::Mat::zeros(height, width, CV_8UC3);
        }

        // Small delay to avoid hogging CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps
    }

    // Clean up
    running = false;
    captureThread.join();

    return 0;
}
