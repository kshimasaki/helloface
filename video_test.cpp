#include <opencv2/opencv.hpp>

int main() {
	cv::VideoCapture cap;
	cap.open(0);
	if(!cap.isOpened()) {
		std::cerr << "Could not open camera.";
		return -1;
	}

	cv::VideoWriter writer;
	std::string filename = "./videotest.avi";
	int codec = cv::VideoWriter::fourcc('M','J','P','G');
	float fps = 10.0;
	cv::Size outputSize = cv::Size(640,480);
	writer.open(filename, codec, fps, outputSize);
	if(!writer.isOpened()) {
		std::cerr << "Could not open writer.";
		return -1;
	}
	size_t frameCount = 0;
	cv::UMat frame;
	while(frameCount < 150) {
		cap >> frame;
		if(frame.empty()) {
			break;
		}
		writer.write(frame);
		frameCount++;
	}
	cap.release();
	writer.release();
	return 0;
}
