#ifndef TRACKER_H
#define TRACKER_H
#include <string>
struct TrackerPose {
    double rotation[3];    
    double translation[3]; 
};
struct PoseMatrix4x4
{
    double m[16]; 

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
    const PoseMatrix4x4& headsetPose);

#endif // TRACKER_H
