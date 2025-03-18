#ifndef TRACKER_H
#define TRACKER_H
#include <string>
struct TrackerPose {
    double rotation[3];    // Rotation vector (Rodrigues)
    double translation[3]; // Translation vector
};
struct PoseMatrix4x4
{
    double m[16]; // 16 doubles, row-major: 
    // m[0..3]   = row 0
    // m[4..7]   = row 1, etc.
};
/// \brief Retrieves the current cube pose.
///        - On the first call, it initializes the camera and other dependencies.
///        - On subsequent calls, it grabs a frame, processes it, and returns the current pose.
/// \return A TrackerPose struct containing the rotation and translation vectors of the cube.
TrackerPose getCubePose(bool showVisualization,
    double markerSideLength,
    double markerGapLength,
    const std::string& calibrationFilePath,
    const std::string& boardDirPath);

PoseMatrix4x4 getCubePoseMatrix(
    bool showVisualization,
    double markerSideLength,
    double markerGapLength,
    const std::string& calibrationFilePath,
    const std::string& boardDirPath,
    const PoseMatrix4x4& headsetPose);

#endif // TRACKER_H
