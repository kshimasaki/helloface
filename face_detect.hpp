#pragma once
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <cstdint>
#include <csignal>
#include <opencv2/opencv.hpp>

#include "executor.h"
#include "execution_object.h"
#include "execution_object_pipeline.h"
#include "execution_object_internal.h"
#include "configuration.h"
#include "imgutil.h"

#include <stdio.h>

using namespace std::chrono;
using namespace tidl;
using namespace std;

using EOP = tidl::ExecutionObjectPipeline;

/**
   A threadsafe Queue that measures FPS.
*/
template <typename T>
class FPSQueue : public std::queue<T> {
public:
    /**
       Constructs a FPSQueue object.
    */
    FPSQueue();

    /**
       Pushes a object of type T onto the queue.
       @param frame A frame to push onto the queue
    */
    void push(const T& frame);

    /**
       Grabs and removes the front frame in the queue.
       @return The front frame in the queue
    */
    T get();

    /**
       Gets the current measured FPS.
       @return The current mesaured FPS
    */
    float getFPS();

    /**
       Clears the queue
    */
    void clear();

private:
    std::mutex mtx;
    unsigned int frameCount;
    steady_clock::time_point start_time;
};

const int EVE_LAYER_GROUP = 1;
const int DSP_LAYER_GROUP = 2;
const float DETECTION_THRESHOLD = 70.0;

tidl::Executor* createExecutor(DeviceType dType, uint32_t num,
                         const Configuration& c, int layerGroupId);

void collectExecutionObjects(const Executor* e, std::vector<ExecutionObject*>& eos);

void allocateMemory(const std::vector<ExecutionObject*>& eos);

bool initializeCaptureDevice(cv::VideoCapture& cap, unsigned int capIndex = 0);

bool initializeWriter(cv::VideoWriter& writer,
                      const string outputFilename,
                      const double fps = 20.0,
                      const unsigned int outputWidth = 768,
                      const unsigned int outputHeight = 320);

bool readFrame(EOP* eop, uint32_t frameIdx, const Configuration& config,
               cv::VideoCapture& cap);

void freeEOPMemory(const std::vector<EOP *>& eops);

bool writeProcessedFrame(const EOP* eop, const Configuration& config);
