#ifndef TRACKER_H
#define TRACKER_H
#include <string>
#include <vector>

struct TrackerPose {
    double rotation[3];    
    double translation[3]; 
};

struct PoseMatrix4x4 {
    double m[16]; 
};

struct ImageData {
    std::vector<unsigned char> data;  // Raw image data
    int width;                        // Image width
    int height;                       // Image height
    int channels;                     // Number of channels (3 for RGB, 4 for RGBA)
    bool isEmpty() const { return data.empty(); }
};

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
    const PoseMatrix4x4& headsetPose,
    ImageData& undistortedImage);

#endif // TRACKER_H
