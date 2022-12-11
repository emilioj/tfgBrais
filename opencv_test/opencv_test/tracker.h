#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <iomanip>
#include <glm/gtc/quaternion.hpp>
#include <glm/fwd.hpp>
#include <math.h>   
#include <opencv2/video/tracking.hpp>



# define M_PI 3.14159265358979323846 
# define SIDELENGTH 0.025
using namespace std;
using namespace cv;
std::vector<cv::Ptr<cv::aruco::GridBoard>> createBoards(float markerSideLength, float markerGapLength);
//void createCalibrationBoard(int squaresX, int squaresY, float squareLength, float markerLength,const Ptr<cv::aruco::Dictionary>dictionary);
bool readCameraParameters(std::string filename, cv::Mat& camMatrix, cv::Mat& distCoeffs);
cv::Vec3d rotateXAxis(cv::Vec3d rotation, double angleRad);
cv::Vec3d rotateYAxis(cv::Vec3d rotation, double angleRad);
cv::Vec3d rotateZAxis(cv::Vec3d rotation, double angleRad);
cv::Vec3d moveAxis(cv::Vec3d& tvec, cv::Vec3d rvec, double distance, int axis);
void cubeCoordinates(int id, cv::Vec3d& rvecs, cv::Vec3d& tvecs, float sideLength, float gapLength);
void averageCube(std::vector<cv::Vec3d>& rvecs, std::vector<cv::Vec3d>& tvecs, cv::Vec3d& outputRvec, cv::Vec3d& outputTvec);
void filterOutput(std::vector<cv::Vec3d> rvecs, std::vector<cv::Vec3d>tvecs);
cv::aruco::DetectorParameters createDetectorParameters();


