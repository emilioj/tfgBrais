#include "tracker.h" // no opencv needed
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>

static std::mutex poseMutex;
static TrackerPose globalPose;
static std::atomic<bool> running(true);

// We decide on some marker dimensions:
static double markerSideLength = 0.015;
static double markerGapLength = 0.007;
std::string calibrationFile = "./data/camera_calibration/camera_calibration.txt";
std::string boardDir = "./data/markers/runtime";

void captureThread(bool showVisualization, double mSideLength, double mGapLength, std::string calibrationFile, std::string boardDir) {
    while (running) {
        TrackerPose newPose = getCubePose(showVisualization, mSideLength, mGapLength, calibrationFile, boardDir);
        {
            std::lock_guard<std::mutex> lock(poseMutex);
            globalPose = newPose;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main() {
    // Start the capture thread with visualization and chosen marker params
    std::thread t(captureThread, true, markerSideLength, markerGapLength, calibrationFile, boardDir);

    // Main loop: print current pose every half-second
    while (true) {
        {
            std::lock_guard<std::mutex> lock(poseMutex);
            std::cout << "Current Pose: R["
                << globalPose.rotation[0] << ", "
                << globalPose.rotation[1] << ", "
                << globalPose.rotation[2] << "] T["
                << globalPose.translation[0] << ", "
                << globalPose.translation[1] << ", "
                << globalPose.translation[2] << "]\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Stop the thread
    running = false;
    t.join();
    return 0;
}
