#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <filesystem> 
using namespace cv;
using namespace std;

int generateMarkers() {
    namespace fs = std::filesystem;

    // Print the current working directory
    std::cout << "Current working directory: " << fs::current_path() << std::endl;

    cv::Mat markerImage;
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    // Generate and save multiple markers
    for (int i = 0; i < 10; i++) {
        cv::aruco::generateImageMarker(dictionary, i, 400, markerImage, 1);
        std::string name = "C:/tfg/tfg/data/markers/pruebas/marker" + std::to_string(i) + ".png";

        // Print the path where the image will be saved
        std::cout << "Saving marker " << i << " to: " << name << std::endl;

        cv::imwrite(name, markerImage);
    }

    return 0;
}