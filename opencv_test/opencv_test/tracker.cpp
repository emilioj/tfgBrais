#include "tracker.h"

std::vector<cv::Ptr<cv::aruco::GridBoard>> createBoards(float markerSideLength, float markerGapLength)
{
    cv::Mat boardImage;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    std::vector<cv::Ptr<cv::aruco::GridBoard>> boards;
    for (int i = 0; i < 6; i++)
    {
        cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(2, 2, markerSideLength, markerGapLength, dictionary, i * 4);
        board->draw(cv::Size(500, 500), boardImage, 10, 1);
        boards.push_back(board);
        string name = "boards/board" + to_string(i) + ".png";
        cv::imwrite(name, boardImage);
    }
    return boards;
}



bool readCameraParameters(std::string filename, cv::Mat& camMatrix, cv::Mat& distCoeffs)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

cv::Vec3d rotateXAxis(cv::Vec3d rotation, double angleRad)
{
    cv::Mat R(3, 3, CV_64F);
    Rodrigues(rotation, R);
    //create a rotation matrix for x axis
    cv::Mat RX = cv::Mat::eye(3, 3, CV_64F);
    RX.at<double>(1, 1) = cos(angleRad);
    RX.at<double>(1, 2) = -sin(angleRad);
    RX.at<double>(2, 1) = sin(angleRad);
    RX.at<double>(2, 2) = cos(angleRad);
    //now multiply
    R = R * RX;
    //finally, the the rodrigues back
    cv::Vec3d output;
    Rodrigues(R, output);
    return output;
}

cv::Vec3d rotateYAxis(cv::Vec3d rotation, double angleRad)
{
    cv::Mat R(3, 3, CV_64F);
    Rodrigues(rotation, R);
    //create a rotation matrix for x axis
    cv::Mat RX = cv::Mat::eye(3, 3, CV_64F);
    RX.at<double>(0, 0) = cos(angleRad);
    RX.at<double>(0, 2) = sin(angleRad);
    RX.at<double>(2, 0) = -sin(angleRad);
    RX.at<double>(2, 2) = cos(angleRad);
    //now multiply
    R = R * RX;
    //finally, the the rodrigues back
    cv::Vec3d output;
    Rodrigues(R, output);
    return output;
}

cv::Vec3d rotateZAxis(cv::Vec3d rotation, double angleRad)
{
    cv::Mat R(3, 3, CV_64F);
    Rodrigues(rotation, R);
    //create a rotation matrix for x axis
    cv::Mat RX = cv::Mat::eye(3, 3, CV_64F);
    RX.at<double>(0, 0) = cos(angleRad);
    RX.at<double>(0, 1) = -sin(angleRad);
    RX.at<double>(1, 0) = sin(angleRad);
    RX.at<double>(1, 1) = cos(angleRad);
    //now multiply
    R = R * RX;
    //finally, the the rodrigues back
    cv::Vec3d output;
    Rodrigues(R, output);
    return output;
}

cv::Vec3d moveAxis(cv::Vec3d& tvec, cv::Vec3d rvec, double distance, int axis)
{
    cv::Mat rotationMatrix, rotationMatrixTransposed;
    Rodrigues(rvec, rotationMatrix);
    rotationMatrixTransposed = rotationMatrix.t();
    double* rz = rotationMatrixTransposed.ptr<double>(axis); // x=0, y=1, z=2
    tvec[0] -= rz[0] * distance;
    tvec[1] -= rz[1] * distance;
    tvec[2] -= rz[2] * distance;
    return tvec;
}

void cubeCoordinates(int id, cv::Vec3d& rvecs, cv::Vec3d& tvecs, float sideLength, float gapLength)
{
    switch (id)
    {
    case 0:
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 0);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 1);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 2);

        break;
    case 1:
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 0);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 1);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 2);
        //rvecs = rotateZAxis(rvecs, M_PI/2);
        rvecs = rotateXAxis(rvecs, -M_PI / 2);
        //tvecs = moveAxis(tvecs, rvecs, SIDELENGTH / 2, 0);
        break;
    case 2:
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 0);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 1);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 2);
        rvecs = rotateXAxis(rvecs, M_PI);

        //rvecs = rotateZAxis(rvecs,-M_PI / 2);
        //rvecs = rotateYAxis(rvecs, M_PI / 2);
        //tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH / 2, 0);

        break;
    case 3:
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 0);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 1);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 2);
        //rvecs = rotateXAxis(rvecs, M_PI);
        rvecs = rotateYAxis(rvecs, M_PI / 2);
        rvecs = rotateZAxis(rvecs, M_PI);

        break;
    case 4:
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 0);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 1);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 2);
        rvecs = rotateXAxis(rvecs, M_PI);
        rvecs = rotateYAxis(rvecs, -M_PI / 2);


        break;
    case 5:
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 0);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 1);
        tvecs = moveAxis(tvecs, rvecs, -SIDELENGTH, 2);
        rvecs = rotateXAxis(rvecs, M_PI / 2);


        break;
    default:
        break;
    }
}

void averageCube(std::vector<cv::Vec3d>& rvecs, std::vector<cv::Vec3d>& tvecs, cv::Vec3d& outputRvec, cv::Vec3d& outputTvec)
{
    cv::Vec3d rvecAverage;
    cv::Vec3d tvecAverage;
    int i, j;
    for (i = 0; i < rvecs.size(); i++)
    {
        for (j = 0; j < 3; j++)
        {
            rvecAverage[j] += rvecs[i][j];
            tvecAverage[j] += tvecs[i][j];
        }
    }
    for (int x = 0; x < 3; x++)
    {
        rvecAverage[x] = rvecAverage[x] / rvecs.size();
        tvecAverage[x] = tvecAverage[x] / rvecs.size();
    }
    outputRvec = rvecAverage;
    outputTvec = tvecAverage;
    rvecs.clear();
    tvecs.clear();
}

void filterOutput(std::vector<cv::Vec3d> rvecs, std::vector<cv::Vec3d> tvecs)
{
}

cv::aruco::DetectorParameters createDetectorParameters()
{
    return cv::aruco::DetectorParameters();
}
