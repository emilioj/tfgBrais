#ifndef TRACKER_H
#define TRACKER_H
#include <string>
#include <vector>

struct TrackerPose
{
    double rotation[3];
    double translation[3];
};

struct PoseMatrix4x4
{
    double m[16];
};

// Use the shared ImageData definition to avoid duplicate-type errors
#include "image_data.h"

TrackerPose getCubePose(bool showVisualization,
                        double markerSideLength,
                        double markerGapLength,
                        const std::string &calibrationFilePath,
                        const std::string &boardDirPath);

PoseMatrix4x4 getCubePoseMatrix(
    bool showVisualization,
    double markerSideLength,
    double markerGapLength,
    const std::string &calibrationFilePath,
    const std::string &boardDirPath,
    ImageData &undistortedImage);

// Get pose as rotation and translation vectors directly
bool getCubePoseVectors(
    bool showVisualization,
    double markerSideLength,
    double markerGapLength,
    const std::string &calibrationFilePath,
    const std::string &boardDirPath,
    ImageData &undistortedImage,
    double rvec[3],
    double tvec[3]);

#endif // TRACKER_H
