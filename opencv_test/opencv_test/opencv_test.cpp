#include "tracker.h"
#include <string>




void main()
{
    cv::VideoCapture inputVideo;
    inputVideo.open(0);
    cv::Mat cameraMatrix, distCoeffs,rotationMatrix,rotationMatrixTransposed,aux0,aux1;
    float pi2 = M_PI / 2;
    string filename = "camera_calibration/camera_calib.txt";
    cv::Vec3d rvec, tvec;
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::Vec3d averageR, averageT;
    readCameraParameters(filename, cameraMatrix, distCoeffs); // This function is located in detect_markers.cpp
    std::vector<cv::Ptr<cv::aruco::GridBoard>> boards;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    float markerSideLength= 0.0215;
    float markerGapLength=0.0085;
    boards = createBoards(markerSideLength, markerGapLength);
    inputVideo.set(CAP_PROP_FRAME_HEIGHT, 720);
    inputVideo.set(CAP_PROP_FRAME_WIDTH, 1280);
    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> markerCorners,rejectedCorners;
        cv::aruco::detectMarkers(image, dictionary, markerCorners, ids,cv::aruco::DetectorParameters::create(), rejectedCorners);
        // if at least one marker detected
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, ids);
            for (int i = 0; i < boards.size(); i++)
            {
                cv::aruco::refineDetectedMarkers(imageCopy, boards[i], markerCorners, ids, rejectedCorners);

            }

            for (int i = 0; i < 6; i++)
            {

                int valid = cv::aruco::estimatePoseBoard(markerCorners, ids, boards[i], cameraMatrix, distCoeffs, rvec, tvec);
                if (valid)
                {
                    cubeCoordinates(i, rvec, tvec, markerSideLength, markerGapLength);
                    rvecs.push_back(rvec);
                    tvecs.push_back(tvec);
                }
            }
            averageCube(rvecs, tvecs, rvec, tvec);
            debugPrinting(rvec, tvec);
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

