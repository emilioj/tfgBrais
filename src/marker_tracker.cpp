#define _USE_MATH_DEFINES
#include <cmath>
#include "marker_tracker.h"
#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

static bool initialized = false;
static cv::VideoCapture inputVideo;
static cv::Mat cameraMatrix, distCoeffs;
static std::vector<cv::aruco::GridBoard> boards;
static cv::aruco::ArucoDetector detector;

namespace
{

	// PoseMatrix4x4 -> cv::Mat (4x4)
	cv::Mat poseMatrixToCvMat(const PoseMatrix4x4 &pose)
	{
		cv::Mat mat(4, 4, CV_64F);
		for (int i = 0; i < 16; i++)
		{
			mat.at<double>(i / 4, i % 4) = pose.m[i];
		}
		return mat;
	}

	// cv::Mat (4x4) -> PoseMatrix4x4
	PoseMatrix4x4 cvMatToPoseMatrix(const cv::Mat &mat)
	{
		PoseMatrix4x4 out;
		for (int row = 0; row < 4; row++)
		{
			for (int col = 0; col < 4; col++)
			{
				out.m[row * 4 + col] = mat.at<double>(row, col);
			}
		}
		return out;
	}

	// cv::Mat from rotation & translation
	cv::Mat buildTransformation(const cv::Vec3d &rvec, const cv::Vec3d &tvec)
	{
		cv::Mat rot3x3;
		cv::Rodrigues(rvec, rot3x3);

		cv::Mat out = cv::Mat::eye(4, 4, CV_64F);
		rot3x3.copyTo(out(cv::Rect(0, 0, 3, 3)));
		out.at<double>(0, 3) = tvec[0];
		out.at<double>(1, 3) = tvec[1];
		out.at<double>(2, 3) = tvec[2];
		return out;
	}

	// Convert cv::Mat to ImageData
	void convertMatToImageData(const cv::Mat &mat, ImageData &imageData)
	{
		if (mat.empty())
		{
			imageData.data.clear();
			imageData.width = 0;
			imageData.height = 0;
			imageData.channels = 0;
			return;
		}

		imageData.width = mat.cols;
		imageData.height = mat.rows;
		imageData.channels = mat.channels();

		size_t totalSize = mat.total() * mat.elemSize();
		imageData.data.resize(totalSize);

		if (mat.isContinuous())
		{
			std::memcpy(imageData.data.data(), mat.data, totalSize);
		}
		else
		{
			size_t rowSize = mat.cols * mat.elemSize();
			for (int i = 0; i < mat.rows; i++)
			{
				std::memcpy(imageData.data.data() + i * rowSize,
							mat.ptr(i), rowSize);
			}
		}
	}

}

bool readCameraParameters(std::string filename, cv::Mat &camMatrix, cv::Mat &distCoeffs)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}

std::vector<cv::aruco::GridBoard> createBoards(float markerSideLength, float markerGapLength, cv::aruco::Dictionary &dictionary, const std::string &boardDirPath)
{
	cv::Mat boardImage;
	std::vector<cv::aruco::GridBoard> boards;
	for (int i = 0; i < 6; i++)
	{
		int firstId = i * 4;
		std::vector<int> ids = {firstId, firstId + 1, firstId + 2, firstId + 3};
		cv::Mat idsMat(ids, true);

		cv::aruco::GridBoard board(cv::Size(2, 2), markerSideLength, markerGapLength, dictionary, idsMat);
		board.generateImage(cv::Size(500, 500), boardImage, 10, 1);

		boards.push_back(board);

		std::string name = boardDirPath + "/board" + std::to_string(i) + ".png";
		cv::imwrite(name, boardImage);
	}
	return boards;
}

static bool initializeOnce(double markerSideLength, double markerGapLength,
						   const std::string &calibrationFilePath,
						   const std::string &boardDirPath)
{
	if (!inputVideo.open(0))
	{
		std::cerr << "Could not open camera." << std::endl;
		return false;
	}

	if (!readCameraParameters(calibrationFilePath, cameraMatrix, distCoeffs))
	{
		std::cerr << "Failed to read camera parameters at: " << calibrationFilePath << std::endl;
		return false;
	}

	cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	boards = createBoards((float)markerSideLength, (float)markerGapLength, dictionary, boardDirPath);

	cv::aruco::DetectorParameters detectorParams;
	detector = cv::aruco::ArucoDetector(dictionary, detectorParams);

	return true;
}

cv::Vec3d moveAxis(cv::Vec3d &tvec, cv::Vec3d rvec, double distance, int axis)
{
	cv::Mat rotationMatrix, rotationMatrixTransposed;
	Rodrigues(rvec, rotationMatrix);
	rotationMatrixTransposed = rotationMatrix.t();
	double *rz = rotationMatrixTransposed.ptr<double>(axis); // x=0, y=1, z=2
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

void averageCube(std::vector<cv::Vec3d> &rvecs, std::vector<cv::Vec3d> &tvecs, cv::Vec3d &outputRvec, cv::Vec3d &outputTvec)
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

void cubeCoordinates(int id, cv::Vec3d &rvecs, cv::Vec3d &tvecs, float markerSideLength, float markerGapLength)
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
						const std::string &calibrationFilePath,
						const std::string &boardDirPath)
{
	TrackerPose poseResult = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

	if (!initialized)
	{
		if (!initializeOnce(markerSideLengthParam, markerGapLengthParam, calibrationFilePath, boardDirPath))
		{

			return poseResult;
		}
		initialized = true;
	}

	cv::Mat frame;
	if (!inputVideo.grab())
	{
		std::cerr << "Failed to grab frame from camera." << std::endl;
		return poseResult;
	}
	inputVideo.retrieve(frame);

	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCorners;
	detector.detectMarkers(frame, markerCorners, ids, rejectedCorners);

	if (!ids.empty())
	{
		for (int i = 0; i < (int)boards.size(); i++)
		{
			detector.refineDetectedMarkers(frame, boards[i], markerCorners, ids, rejectedCorners);
		}

		std::vector<cv::Vec3d> rvecs, tvecs;
		for (int i = 0; i < (int)boards.size(); i++)
		{
			cv::Mat objPoints, imgPoints;
			boards[i].matchImagePoints(markerCorners, ids, objPoints, imgPoints);
			if (objPoints.empty() || imgPoints.empty())
			{
				continue;
			}

			cv::Vec3d boardRvec, boardTvec;
			bool valid = cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, boardRvec, boardTvec);
			if (valid)
			{
				cubeCoordinates(i, boardRvec, boardTvec, (float)markerSideLengthParam, (float)markerGapLengthParam);
				rvecs.push_back(boardRvec);
				tvecs.push_back(boardTvec);
			}
		}

		if (!rvecs.empty() && !tvecs.empty())
		{
			cv::Vec3d rvec, tvec;
			averageCube(rvecs, tvecs, rvec, tvec);

			poseResult.rotation[0] = rvec[0];
			poseResult.rotation[1] = rvec[1];
			poseResult.rotation[2] = rvec[2];

			poseResult.translation[0] = tvec[0];
			poseResult.translation[1] = tvec[1];
			poseResult.translation[2] = tvec[2];

			if (showVisualization)
			{
				cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, (float)markerSideLengthParam);
			}
		}
	}

	if (showVisualization)
	{
		// Show the original camera view in a window
		cv::imshow("Camera", frame);
		cv::waitKey(1);
	}

	return poseResult;
}

static bool initializeTracking(
	double markerSideLength,
	double markerGapLength,
	const std::string &calibrationFilePath,
	const std::string &boardDirPath)
{
	if (initialized)
	{
		return true;
	}

	// Use DirectShow backend explicitly
	inputVideo.setExceptionMode(true); // Enable exceptions for better error handling
	try
	{
		// Use DirectShow (DSHOW) backend explicitly with index 0
		if (!inputVideo.open(0 + cv::CAP_DSHOW))
		{
			std::cerr << "Could not open camera using DirectShow\n";
			return false;
		}

		// Set camera properties for better performance
		inputVideo.set(cv::CAP_PROP_AUTOFOCUS, 0); // Disable autofocus
		inputVideo.set(cv::CAP_PROP_SETTINGS, 0);  // Don't show Windows camera settings dialog
	}
	catch (const cv::Exception &e)
	{
		std::cerr << "OpenCV error: " << e.what() << std::endl;
		return false;
	}

	if (!readCameraParameters(calibrationFilePath, cameraMatrix, distCoeffs))
	{
		std::cerr << "Failed to read camera params from: " << calibrationFilePath << "\n";
		return false;
	}

	cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	boards = createBoards(static_cast<float>(markerSideLength),
						  static_cast<float>(markerGapLength),
						  dict,
						  boardDirPath);

	cv::aruco::DetectorParameters params;
	detector = cv::aruco::ArucoDetector(dict, params);

	initialized = true;
	return true;
}

static bool detectMarkersInFrame(cv::Mat &frame, std::vector<int> &ids,
								 std::vector<std::vector<cv::Point2f>> &corners,
								 std::vector<std::vector<cv::Point2f>> &rejected)
{
	try
	{
		if (!inputVideo.grab())
		{
			std::cerr << "Failed to grab frame.\n";
			return false;
		}

		if (!inputVideo.retrieve(frame))
		{
			std::cerr << "Failed to retrieve frame.\n";
			return false;
		}

		if (frame.empty())
		{
			std::cerr << "Retrieved frame is empty.\n";
			return false;
		}

		detector.detectMarkers(frame, corners, ids, rejected);

		for (size_t i = 0; i < boards.size(); i++)
		{
			detector.refineDetectedMarkers(frame, boards[i], corners, ids, rejected);
		}

		return true;
	}
	catch (const cv::Exception &e)
	{
		std::cerr << "OpenCV error in detectMarkersInFrame: " << e.what() << std::endl;
		return false;
	}
}

static bool calculateBoardPoses(
	const std::vector<int> &ids,
	const std::vector<std::vector<cv::Point2f>> &corners,
	double markerSideLength,
	double markerGapLength,
	std::vector<cv::Vec3d> &foundRvecs,
	std::vector<cv::Vec3d> &foundTvecs)
{
	if (ids.empty())
	{
		return false;
	}

	for (size_t i = 0; i < boards.size(); i++)
	{
		cv::Mat objPoints, imgPoints;
		boards[i].matchImagePoints(corners, ids, objPoints, imgPoints);

		if (objPoints.empty() || imgPoints.empty())
		{
			continue;
		}

		cv::Vec3d boardR, boardT;
		bool valid = cv::solvePnP(objPoints, imgPoints,
								  cameraMatrix, distCoeffs,
								  boardR, boardT);

		if (valid)
		{
			cubeCoordinates(static_cast<int>(i), boardR, boardT,
							static_cast<float>(markerSideLength),
							static_cast<float>(markerGapLength));
			foundRvecs.push_back(boardR);
			foundTvecs.push_back(boardT);
		}
	}

	return !foundRvecs.empty();
}

static void visualizePose(
	cv::Mat &frame,
	const cv::Vec3d &rvec,
	const cv::Vec3d &tvec,
	float markerSideLength,
	bool showVisualization)
{
	// Only draw axes and display if visualization is enabled
	if (showVisualization)
	{
		// Create a copy for AR visualization
		cv::Mat arFrame = frame.clone();

		// Draw axes on the AR visualization frame
		cv::drawFrameAxes(arFrame,
						  cameraMatrix,
						  distCoeffs,
						  rvec,
						  tvec,
						  markerSideLength);

		// Show original camera view in one window
		cv::imshow("Camera", frame);

		// Show AR visualization in a separate window
		cv::imshow("Cube axis", arFrame);
		cv::waitKey(1);
	}
}

PoseMatrix4x4 getCubePoseMatrix(
	bool showVisualization,
	double markerSideLength,
	double markerGapLength,
	const std::string &calibrationFilePath,
	const std::string &boardDirPath,
	ImageData &undistortedImage)
{
	// Initialize tracking if needed
	if (!initializeTracking(markerSideLength, markerGapLength, calibrationFilePath, boardDirPath))
	{
		return PoseMatrix4x4{};
	}

	// Get frame and detect markers
	cv::Mat frame;
	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f>> corners, rejected;
	if (!detectMarkersInFrame(frame, ids, corners, rejected))
	{
		return PoseMatrix4x4{};
	}

	// Undistort the image using OpenCV internally
	cv::Mat cvUndistortedImage;
	if (showVisualization)
	{
		// Only undistort the image if visualization is enabled or if undistortedImage will be used
		cv::undistort(frame, cvUndistortedImage, cameraMatrix, distCoeffs);
		// Convert the undistorted OpenCV Mat to our generic ImageData format
		convertMatToImageData(cvUndistortedImage, undistortedImage);
	}
	else
	{
		// Clear the output image if visualization is disabled
		undistortedImage = ImageData();
	}

	// Calculate poses for detected boards
	std::vector<cv::Vec3d> foundRvecs, foundTvecs;
	calculateBoardPoses(ids, corners, markerSideLength, markerGapLength, foundRvecs, foundTvecs);

	// Average poses and build final transformation
	cv::Mat finalTransform = cv::Mat::eye(4, 4, CV_64F);
	if (!foundRvecs.empty())
	{
		cv::Vec3d rvecFinal(0, 0, 0), tvecFinal(0, 0, 0);
		averageCube(foundRvecs, foundTvecs, rvecFinal, tvecFinal);

		// Get the cube's model matrix in camera space
		cv::Mat cubeTransform = buildTransformation(rvecFinal, tvecFinal);
		cv::Mat cubeModel;
		if (!cv::invert(cubeTransform, cubeModel))
		{
			std::cerr << "Error inverting cubeTransform\n";
			return PoseMatrix4x4{};
		}

		finalTransform = cubeModel;

		// Visualize if requested - using the original frame, not the undistorted one
		if (showVisualization)
		{
			visualizePose(frame, rvecFinal, tvecFinal, static_cast<float>(markerSideLength), showVisualization);
		}
	}

	// Convert the final transform to PoseMatrix4x4
	return cvMatToPoseMatrix(finalTransform);
}

// Get pose as rotation and translation vectors directly
bool getCubePoseVectors(
	bool showVisualization,
	double markerSideLength,
	double markerGapLength,
	const std::string &calibrationFilePath,
	const std::string &boardDirPath,
	ImageData &undistortedImage,
	double rvec[3],
	double tvec[3])
{
	// Initialize tracking if needed
	if (!initializeTracking(markerSideLength, markerGapLength, calibrationFilePath, boardDirPath))
	{
		return false;
	}

	// Get frame and detect markers
	cv::Mat frame;
	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f>> corners, rejected;
	if (!detectMarkersInFrame(frame, ids, corners, rejected))
	{
		return false;
	}

	// Undistort the image using OpenCV internally
	cv::Mat cvUndistortedImage;
	if (showVisualization)
	{
		// Only undistort the image if visualization is enabled or if undistortedImage will be used
		cv::undistort(frame, cvUndistortedImage, cameraMatrix, distCoeffs);
		// Convert the undistorted OpenCV Mat to our generic ImageData format
		convertMatToImageData(cvUndistortedImage, undistortedImage);
	}
	else
	{
		// Clear the output image if visualization is disabled
		undistortedImage = ImageData();
	}

	// Calculate poses for detected boards
	std::vector<cv::Vec3d> foundRvecs, foundTvecs;
	calculateBoardPoses(ids, corners, markerSideLength, markerGapLength, foundRvecs, foundTvecs);

	// If no boards were found, return false
	if (foundRvecs.empty())
	{
		return false;
	}

	// Average poses to get final rotation and translation
	cv::Vec3d rvecFinal(0, 0, 0), tvecFinal(0, 0, 0);
	averageCube(foundRvecs, foundTvecs, rvecFinal, tvecFinal);

	// Copy the values to the output arrays
	for (int i = 0; i < 3; i++)
	{
		rvec[i] = rvecFinal[i];
		tvec[i] = tvecFinal[i];
	}

	// Visualize if requested - using the original frame, not the undistorted one
	if (showVisualization)
	{
		visualizePose(frame, rvecFinal, tvecFinal, static_cast<float>(markerSideLength), showVisualization);
	}

	return true;
}
