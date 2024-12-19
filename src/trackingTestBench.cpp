#include "tracker.h"
#include <string>
#include <opencv2/calib3d.hpp>




void main()
{
    cv::VideoCapture inputVideo;
    inputVideo.open(0);
    cv::Mat cameraMatrix, distCoeffs,rotationMatrix,rotationMatrixTransposed,aux0,aux1;
    float pi2 = M_PI / 2;
    std::string cameraCalibrationFilePath = "D:/brais/tfg/win64-msvc2022/camera_calibration.txt";
    cv::Vec3d rvec, tvec;
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::Vec3d averageR, averageT;
    readCameraParameters(cameraCalibrationFilePath, cameraMatrix, distCoeffs);
    std::vector<cv::aruco::GridBoard> boards;
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    aruco::DetectorParameters detectorParams;
    aruco::ArucoDetector detector(dictionary, detectorParams);
    float markerSideLength= 0.015;
    float markerGapLength=0.007;
    boards = createBoards(markerSideLength, markerGapLength,dictionary);
    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> markerCorners,rejectedCorners;
       detector.detectMarkers(image, markerCorners, ids, rejectedCorners);
       if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, ids);
            for (int i = 0; i < boards.size(); i++)
            {
                detector.refineDetectedMarkers(imageCopy, boards[i], markerCorners, ids, rejectedCorners);
            }
            for (int i = 0; i < 6; i++)
            {
                cv::Mat objPoints, imgPoints;
                boards[i].matchImagePoints(markerCorners, ids, objPoints, imgPoints);
                if (objPoints.empty() || imgPoints.empty()) {
                    continue;
                }
                bool valid = cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
                if (valid)
                {
                    cubeCoordinates(i, rvec, tvec, markerSideLength, markerGapLength);
                    rvecs.push_back(rvec);
                    tvecs.push_back(tvec);
                }
            }
            averageCube(rvecs, tvecs, rvec, tvec);
            cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, markerSideLength);
    
        }
        rvecs.clear();
        tvecs.clear();
        cv::imshow("out", imageCopy);
        char key = (char)cv::waitKey(1);
       if (key == 27)
            break;
    }
}

