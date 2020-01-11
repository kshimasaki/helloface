#pragma once
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std::chrono;

template <typename T>
class FPSQueue : public std::queue<T> {
public:
  FPSQueue();
  void push(const T&);
  T get();
  float getFPS();
  void clear();

private:
  std::mutex mtx;
  unsigned int frameCount;
  steady_clock::time_point start_time;
};
