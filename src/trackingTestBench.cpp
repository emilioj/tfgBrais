#include "tracker.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static PoseMatrix4x4 globalPose;
static std::mutex poseMutex;
static std::atomic<bool> running(true);

PoseMatrix4x4 makeIdentity()
{
    PoseMatrix4x4 pose;
    for (int i = 0; i < 16; ++i) {
        pose.m[i] = ((i % 5) == 0) ? 1.0 : 0.0;
        // i%5 == 0 picks out diagonal elements (0,5,10,15)
    }
    return pose;
}

// Helper function to convert our ImageData to cv::Mat for visualization
cv::Mat imageDataToMat(const ImageData& imageData) {
    if (imageData.isEmpty()) {
        return cv::Mat();
    }
    
    cv::Mat result(imageData.height, imageData.width, 
                  CV_8UC(imageData.channels), 
                  const_cast<unsigned char*>(imageData.data.data()));
    return result.clone(); // Clone to ensure we have our own copy of the data
}

void captureThreadFunc(bool showViz,
    double markerSideLen,
    double markerGapLen,
    const std::string& calibFile,
    const std::string& boardsDir)
{
    // (In a real scenario, you might update this from IMU or other tracking)
    PoseMatrix4x4 headsetPose = makeIdentity();

    while (running)
    {
        ImageData undistortedImage;
        PoseMatrix4x4 newPose = getCubePoseMatrix(
            showViz,
            markerSideLen,
            markerGapLen,
            calibFile,
            boardsDir,
            headsetPose,
            undistortedImage
        );

        {
            std::lock_guard<std::mutex> lock(poseMutex);
            globalPose = newPose;
        }

        // Display the undistorted image if available
        if (!undistortedImage.isEmpty() && showViz) {
            // Convert to OpenCV Mat for visualization only
            cv::Mat displayImage = imageDataToMat(undistortedImage);
            if (!displayImage.empty()) {
                cv::imshow("Undistorted Image", displayImage);
                cv::waitKey(1);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main()
{
    std::thread captureThread(captureThreadFunc, true, 0.015, 0.007, "./data/camera_calibration/camera_calibration.txt", "./data/markers/runtime");
    while (true)
    {
        {
            std::lock_guard<std::mutex> lock(poseMutex);
            std::cout << "Current 4x4 Pose:\n";
            for (int row = 0; row < 4; ++row) {
                std::cout << globalPose.m[row * 4 + 0] << " "
                    << globalPose.m[row * 4 + 1] << " "
                    << globalPose.m[row * 4 + 2] << " "
                    << globalPose.m[row * 4 + 3] << "\n";
            }
            std::cout << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    running = false;
    captureThread.join();
    return 0;
}
