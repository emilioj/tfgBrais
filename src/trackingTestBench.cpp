#include "tracker.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <chrono>

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
        PoseMatrix4x4 newPose = getCubePoseMatrix(
            showViz,
            markerSideLen,
            markerGapLen,
            calibFile,
            boardsDir,
            headsetPose
        );

        {
            std::lock_guard<std::mutex> lock(poseMutex);
            globalPose = newPose;
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
