#define _USE_MATH_DEFINES
#include <cmath>
#include "tracker.h"
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

static bool initialized = false;
static cv::VideoCapture inputVideo;
static cv::Mat cameraMatrix, distCoeffs;
static std::vector<cv::aruco::GridBoard> boards;
static cv::aruco::ArucoDetector detector;

bool readCameraParameters(std::string filename, cv::Mat& camMatrix, cv::Mat& distCoeffs)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}

std::vector<cv::aruco::GridBoard> createBoards(float markerSideLength, float markerGapLength, cv::aruco::Dictionary& dictionary, const std::string& boardDirPath)
{
	cv::Mat boardImage;
	std::vector<cv::aruco::GridBoard> boards;
	for (int i = 0; i < 6; i++)
	{
		int firstId = i * 4;
		std::vector<int> ids = { firstId, firstId + 1, firstId + 2, firstId + 3 };
		cv::Mat idsMat(ids, true);

		cv::aruco::GridBoard board(cv::Size(2, 2), markerSideLength, markerGapLength, dictionary, idsMat);
		board.generateImage(cv::Size(500, 500), boardImage, 10, 1);

		boards.push_back(board);

		// Use the passed-in path for saving
		std::string name = boardDirPath + "/board" + std::to_string(i) + ".png";
		cv::imwrite(name, boardImage);
	}
	return boards;
}


static bool initializeOnce(double markerSideLength, double markerGapLength,
	const std::string& calibrationFilePath,
	const std::string& boardDirPath)
{
	if (!inputVideo.open(0)) {
		std::cerr << "Could not open camera." << std::endl;
		return false;
	}

	if (!readCameraParameters(calibrationFilePath, cameraMatrix, distCoeffs)) {
		std::cerr << "Failed to read camera parameters at: " << calibrationFilePath << std::endl;
		return false;
	}

	cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	boards = createBoards((float)markerSideLength, (float)markerGapLength, dictionary, boardDirPath);

	cv::aruco::DetectorParameters detectorParams; // defaults
	detector = cv::aruco::ArucoDetector(dictionary, detectorParams);

	return true;
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

cv::Vec3d rotateXAxis(cv::Vec3d rotation, double angleRad)
{
	cv::Mat R(3, 3, CV_64F);
	Rodrigues(rotation, R);
	cv::Mat RX = cv::Mat::eye(3, 3, CV_64F);
	RX.at<double>(1, 1) = cos(angleRad);
	RX.at<double>(1, 2) = -sin(angleRad);
	RX.at<double>(2, 1) = sin(angleRad);
	RX.at<double>(2, 2) = cos(angleRad);
	R = R * RX;
	cv::Vec3d output;
	Rodrigues(R, output);
	return output;
}

cv::Vec3d rotateYAxis(cv::Vec3d rotation, double angleRad)
{
	cv::Mat R(3, 3, CV_64F);
	Rodrigues(rotation, R);
	cv::Mat RX = cv::Mat::eye(3, 3, CV_64F);
	RX.at<double>(0, 0) = cos(angleRad);
	RX.at<double>(0, 2) = sin(angleRad);
	RX.at<double>(2, 0) = -sin(angleRad);
	RX.at<double>(2, 2) = cos(angleRad);
	R = R * RX;
	cv::Vec3d output;
	Rodrigues(R, output);
	return output;
}

cv::Vec3d rotateZAxis(cv::Vec3d rotation, double angleRad)
{
	cv::Mat R(3, 3, CV_64F);
	Rodrigues(rotation, R);
	cv::Mat RX = cv::Mat::eye(3, 3, CV_64F);
	RX.at<double>(0, 0) = cos(angleRad);
	RX.at<double>(0, 1) = -sin(angleRad);
	RX.at<double>(1, 0) = sin(angleRad);
	RX.at<double>(1, 1) = cos(angleRad);
	R = R * RX;
	cv::Vec3d output;
	Rodrigues(R, output);
	return output;
}

void averageCube(std::vector<cv::Vec3d>& rvecs, std::vector<cv::Vec3d>& tvecs, cv::Vec3d& outputRvec, cv::Vec3d& outputTvec)
{
	cv::Vec3d rvecAverage;
	cv::Vec3d tvecAverage;
	for (size_t i = 0; i < rvecs.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			rvecAverage[j] += rvecs[i][j];
			tvecAverage[j] += tvecs[i][j];
		}
	}
	for (int x = 0; x < 3; x++)
	{
		rvecAverage[x] = rvecAverage[x] / rvecs.size();
		tvecAverage[x] = tvecAverage[x] / tvecs.size();
	}
	outputRvec = rvecAverage;
	outputTvec = tvecAverage;
	rvecs.clear();
	tvecs.clear();
}

void cubeCoordinates(int id, cv::Vec3d& rvecs, cv::Vec3d& tvecs, float markerSideLength, float markerGapLength)
{
	float boardSideLength = (2 * markerSideLength + markerGapLength) / 2.0f;
	switch (id)
	{
	case 0:
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 0);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 1);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 2);
		break;
	case 1:
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 0);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 1);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 2);
		rvecs = rotateXAxis(rvecs, -M_PI / 2);
		break;
	case 2:
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 0);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 1);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 2);
		rvecs = rotateXAxis(rvecs, M_PI);
		break;
	case 3:
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 0);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 1);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 2);
		rvecs = rotateYAxis(rvecs, M_PI / 2);
		rvecs = rotateZAxis(rvecs, M_PI);
		break;
	case 4:
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 0);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 1);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 2);
		rvecs = rotateXAxis(rvecs, M_PI);
		rvecs = rotateYAxis(rvecs, -M_PI / 2);
		break;
	case 5:
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 0);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 1);
		tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 2);
		rvecs = rotateXAxis(rvecs, M_PI / 2);
		break;
	default:
		break;
	}
}

TrackerPose getCubePose(bool showVisualization,
	double markerSideLengthParam,
	double markerGapLengthParam,
	const std::string& calibrationFilePath,
	const std::string& boardDirPath)
{
	TrackerPose poseResult = { {0.0, 0.0, 0.0},{0.0, 0.0, 0.0} };

	if (!initialized) {
		if (!initializeOnce(markerSideLengthParam, markerGapLengthParam, calibrationFilePath, boardDirPath)) {
			// Initialization failed
			return poseResult;
		}
		initialized = true;
	}

	cv::Mat frame;
	if (!inputVideo.grab()) {
		std::cerr << "Failed to grab frame from camera." << std::endl;
		return poseResult;
	}
	inputVideo.retrieve(frame);

	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCorners;
	detector.detectMarkers(frame, markerCorners, ids, rejectedCorners);

	if (!ids.empty()) {
		for (int i = 0; i < (int)boards.size(); i++) {
			detector.refineDetectedMarkers(frame, boards[i], markerCorners, ids, rejectedCorners);
		}

		std::vector<cv::Vec3d> rvecs, tvecs;
		for (int i = 0; i < (int)boards.size(); i++) {
			cv::Mat objPoints, imgPoints;
			boards[i].matchImagePoints(markerCorners, ids, objPoints, imgPoints);
			if (objPoints.empty() || imgPoints.empty()) {
				continue;
			}

			cv::Vec3d boardRvec, boardTvec;
			bool valid = cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, boardRvec, boardTvec);
			if (valid) {
				cubeCoordinates(i, boardRvec, boardTvec, (float)markerSideLengthParam, (float)markerGapLengthParam);
				rvecs.push_back(boardRvec);
				tvecs.push_back(boardTvec);
			}
		}

		if (!rvecs.empty() && !tvecs.empty()) {
			cv::Vec3d rvec, tvec;
			averageCube(rvecs, tvecs, rvec, tvec);

			// Copy results to the struct
			poseResult.rotation[0] = rvec[0];
			poseResult.rotation[1] = rvec[1];
			poseResult.rotation[2] = rvec[2];

			poseResult.translation[0] = tvec[0];
			poseResult.translation[1] = tvec[1];
			poseResult.translation[2] = tvec[2];

			if (showVisualization) {
				cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, (float)markerSideLengthParam);
			}
		}

	}

	if (showVisualization) {
		cv::imshow("out", frame);
		cv::waitKey(1);
	}

	return poseResult;
}
