#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/aruco.hpp>

using namespace cv;
using namespace std;

int main()
{
	cv::Mat markerImage;
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	cv::aruco::drawMarker(dictionary, 23, 200, markerImage, 1);
	for (int i = 0; i < 10; i++)
	{
		cv::aruco::drawMarker(dictionary, i, 400, markerImage, 1);
		string name = "markers/marker" + to_string(i) + ".png";
		cv::imwrite(name, markerImage);
	}
}