#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <opencv2/opencv.hpp>

#include "face_detect.hpp"
#include <executor.h>
#include <execution_object.h>
#include <execution_object_pipeline.h>
#include <execution_object_internal.h>
#include <configuration.h>
#include <imgutil.h>

using namespace std::chrono;

template <typename T>
FPSQueue<T>::FPSQueue() : frameCount(0) {}

template <typename T>
void FPSQueue<T>::push(const T& frame) {
  std::lock_guard<std::mutex> lock(mtx);
  frameCount++;
  std::queue<T>::push(frame);
  if(frameCount == 1) { start_time = steady_clock::now(); }
}

template <typename T>
T FPSQueue<T>::get() {
  std::lock_guard<std::mutex> lock(mtx);
  T res = this->front();
  this->pop();
  return res;
}

template <typename T>
float FPSQueue<T>::getFPS() {
  steady_clock::time_point now = steady_clock::now();
  duration<float> elapsed = duration_cast<duration<float>>(now - this->start_time);
  return elapsed / static_cast<float>(frameCount);
}

template <typename T>
void FPSQueue<T>::clear() {
  std::lock_guard<std::mutex> lock(mtx);
  while(!this->empty()) {
    this->pop();
  }
}

using namespace tidl;

int main(int argc, char** argv ) {
	uint32_t num_eve = Executor::GetNumDevices(DeviceType::EVE);
  // Setup webcam input
  // TODO: Set video codec for capture
  cv::VideoCapture cap;
  cap.open(0);
  if(!cap.isOpened()) {
    std::cout << "Failed to open camera. Exiting." << std::endl;
    return -1;
  }
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  cv::VideoWriter writer;
  int codec = cv::VideoWriter::fourcc('M','J','P','G');
  double fps = 5.0;
  std::string filename = "./test.avi";
  cv::Size outputSize = cv::Size(640,480);
  writer.open(filename, codec, fps, outputSize);
  if(!writer.isOpened()) {
    std::cerr << "Could not open writer.";
    return -1;
  }

  // Read in model and weight files and load model into a cv::dnn::Net
  const std::string modelFile = "./opencv_face_detector_uint8.pb";
  const std::string modelConfigFile = "./opencv_face_detector.pbtxt";
  cv::dnn::Net faceDetectNet = cv::dnn::readNetFromTensorflow(modelFile, modelConfigFile);

  // Start capturing frames and processing
  cv::Mat frame;
  while(cv::waitKey(1) < 0) {
    cap >> frame;
    std::cout << "grabbed frame." << std::endl;
    if(frame.empty()) {
      cv::waitKey();
      break;
    }
    //preprocess(frame, Size(640,480), scale, mean, swapRB);
    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 0.5, cv::Size(640, 480), 0.03, true, false);
	cv::UMat inputUMat;
	inputBlob.copyTo(inputUMat);
    faceDetectNet.setInput(inputUMat, "data");

    cv::Mat detection = faceDetectNet.forward("detection_out");

    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    // NOTE: This constructor doesn't copy any data.  It just creates a new
    //   matrix header pointing to detection.
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++) {
      float confidence = detectionMat.at<float>(i, 2);

      if(confidence > 0.3) {
        int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * 640);
        int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * 480);
        int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * 640);
        int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * 480);

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
      }
    }
    //cv::imshow("window name", frame);
    writer.write(frame);
  }
  std::cout << "Processing ending..." << std::endl;
  cv::destroyAllWindows();
  cap.release();
  writer.release();
  return 0;
}
