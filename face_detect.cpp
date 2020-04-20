#include "face_detect.hpp"

using namespace std::chrono;
using namespace std;
using namespace tidl;
using EOP = tidl::ExecutionObjectPipeline;

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

Executor* createExecutor(DeviceType dType, int num,
                         const Configuration& config, int layerGroupId) {

    if (num == 0) { return nullptr; }

    DeviceIds dIds;
    for (int i = 0; i < num; ++i) {
        dIds.insert(static_cast<DeviceId>(i));
    }

    return new Executor(dType, dIds, config, layerGroupId);
}

void collectExecutionObjects(const Executor* e, vector<ExecutionObject*>& eos) {
    /*
      Collects ExecutionObjects from an executor and pushes them into the provided vector.
      Arguments:
      const Executor* e: The Executor object to collect objects from
      vector<ExecutionObject*>& eos: The vector to place the ExecutionObject's into.
      Returns:
      Nothing.
    */
    if(!e) { return; }
    for(unsigned int i = 0; i < e->GetNumExecutionObjects(); ++i) {
        eos.push_back((*e)[i]);
    }
    return;
}

void allocateMemory(const vector<EOP*>& eops) {
    for(auto eop : eops) {
        size_t inSize = eop->GetInputBufferSizeInBytes();
        size_t outSize = eop->GetOutputBufferSizeInBytes();
        void* inPtr = malloc(inSize);
        void* outPtr = malloc(outSize);
        // TODO: setup proper exceptions
        assert(inPtr != nullptr && outPtr != nullptr);

        ArgInfo in = { ArgInfo(inPtr, inSize) };
        ArgInfo out = { ArgInfo(outPtr, outSize) };
        eop->SetInputOutputBuffer(in, out);
    }
}

bool initializeCaptureDevice(cv::VideoCapture& cap,
                             unsigned int capIndex /*= 0*/) {
    cap.open(capIndex);
    if (!cap.isOpened()) {
        return false;
    }

    return true;
}

bool initializeWriter(cv::VideoWriter& writer,
                      const string outputFilename,
                      const double fps /*= 20.0*/,
                      const unsigned int outputWidth /*= 768*/,
                      const unsigned int outputHeight /*= 320*/) {
    int codec = cv::VideoWriter::fourcc('M','J','P','G');
    cv::Size outputSize = cv::Size(outputWidth, outputHeight);

    writer.open(outputFilename, codec, fps, outputSize);

    if (!writer.isOpened()) {
        std::printf("Could not open VideoWriter. Exiting.");
        return false;
    }

    return true;
}

bool readFrame(EOP* eop, uint32_t frameIdx, const Configuration& config,
               cv::VideoCapture& cap) {
    // TODO: add stopped condition test here

    eop->SetFrameIndex(frameIdx);

    char* inputEOPBuffer = eop->GetInputBufferPtr();
    assert (inputEOPBuffer != nullptr);
    int channelSize = config.inWidth * config.inHeight;
    int frameSize = channelSize * config.inNumChannels;

    cv::Mat frame;
    cv::Mat resizedFrame, bgrResizedFrames[3];
    const int numFrameSkip = 4;
    for (int i = 0; i < numFrameSkip; ++i) {
        // Only grab every (numFrameSkip + 1)'th frame
        if (!cap.grab()) { return false; }
    }
    if (!cap.retrieve(frame)) { return false; }
    uint32_t origWidth = frame.cols;
    uint32_t origHeight = frame.rows;
    cv::resize(frame, resizedFrame, cv::Size(config.inWidth, config.inHeight),
               0, 0, cv::INTER_AREA);
    cv::split(resizedFrame, bgrResizedFrames);
    memcpy(inputEOPBuffer, bgrResizedFrames[0].ptr(), channelSize);
    memcpy(inputEOPBuffer + channelSize, bgrResizedFrames[1].ptr(), channelSize);
    memcpy(inputEOPBuffer + 2*channelSize, bgrResizedFrames[2].ptr(), channelSize);

    return true;
}

void freeEOPMemory(const std::vector<EOP *>& eops) {
    for (auto eo : eops) {
        free(eo->GetInputBufferPtr());
        free(eo->GetOutputBufferPtr());
    }
}

bool writeProcessedFrame(const EOP* eop, const Configuration& config) {
    // Create variables for frame
    int inWidth = config.inWidth;
    int inHeight = config.inHeight;
    int channelSize = inWidth * inHeight;
    cv::Mat bufferFrame, outFrame;
    cv::Mat bufferBGR[3];

    // Get frame from output buffer
    unsigned char *inputBuffer = (unsigned char *) eop->GetInputBufferPtr();
    bufferBGR[0] = cv::Mat(inHeight, inWidth, CV_8UC(1), inputBuffer);
    bufferBGR[1] = cv::Mat(inHeight, inWidth, CV_8UC(1), inputBuffer + channelSize);
    bufferBGR[2] = cv::Mat(inHeight, inWidth, CV_8UC(1), inputBuffer + 2 * channelSize);
    cv::merge(bufferBGR, 3, bufferFrame);

    // Draw boxes around detected faces
    float* outBuffer = (float*)(eop->GetOutputBufferPtr());
    size_t numFloats = eop->GetOutputBufferSizeInBytes() / sizeof(float);
    for (size_t i = 0; i < numFloats / 7; ++i) {
        int index = (int) outBuffer[i * 7 + 0];
        float score = outBuffer[i * 7 + 1];
        // TODO: Tune detection threshold, maybe make it a user option?
        if (score * 100 < DETECTION_THRESHOLD) { continue; }

        int label = (int)  outBuffer[i * 7 + 1];
        int xmin  = (int) (outBuffer[i * 7 + 3] * inWidth);
        int ymin  = (int) (outBuffer[i * 7 + 4] * inWidth);
        int xmax  = (int) (outBuffer[i * 7 + 5] * inWidth);
        int ymax  = (int) (outBuffer[i * 7 + 6] * inWidth);
        if (xmin < 0) { xmin = 0; }
        if (ymin < 0) { ymin = 0; }
        if (xmax > inWidth) { xmax = inWidth; }
        if (ymax > inHeight) { ymax = inHeight; }
        cv::rectangle(bufferFrame, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(0,254,0), 2);
    }

    // TODO: Don't hard-code this
    cv::resize(bufferFrame, outFrame, cv::Size(640, 480));

    int frameIndex = eop->GetFrameIndex();
    string outfileName = "helloface_" + std::to_string(frameIndex) + ".png";
    cv::imwrite(outfileName, outFrame);
    std::cout << "saving frame " << frameIndex << "\n";

    return true;
}

using namespace tidl;

int main(int argc, char** argv) {
    signal(SIGABRT, exit);
    signal(SIGTERM, exit);

    uint32_t numEves = Executor::GetNumDevices(DeviceType::EVE);
    uint32_t numDsps = Executor::GetNumDevices(DeviceType::DSP);
    std::cout << "Found " << numEves << " EVEs and " << numDsps << " DSPs.";
    std::cout << std::endl;

    Configuration config;
    std::string configFname = "/home/debian/helloface/face_detect_config.txt";
    bool status = config.ReadFromFile(configFname);
    if (!status) {
        std::cerr << "Error reading config file: " << configFname << "\n";
        return false;
    }

    cv::VideoCapture cap;
    initializeCaptureDevice(cap);

    string outFilename  = "test.avi";
    cv::VideoWriter writer;
    initializeWriter(writer, outFilename);

    try {
        cout << "Status variable is " << status << endl;
        cout << "Doing unique_ptr createExecutor calls"
             << " with the following vars\n"
             << "numEves: " << numEves << "\n"
             << "numDsps: " << numDsps << "\n";
        /*
        unique_ptr<Executor> eveExecutor(createExecutor(DeviceType::EVE,
                                                        numEves,
                                                        config,
                                                        EVE_LAYER_GROUP));
        unique_ptr<Executor> dspExecutor(createExecutor(DeviceType::DSP,
                                                        numDsps,
                                                        config,
                                                        DSP_LAYER_GROUP));
        */
        Executor* eveExecutor = createExecutor(DeviceType::EVE, (int) numEves,
                                               config, EVE_LAYER_GROUP);

        Executor* dspExecutor = createExecutor(DeviceType::DSP, (int) numDsps,
                                               config, DSP_LAYER_GROUP);

        cout << "Finished unique_ptr executor creation" << endl;
        // Make two EOPs for each DSP-EVE group for double buffering
        std::vector<EOP *> eops;
        const uint32_t pipelineDepth = 2;
        const uint32_t numPipes = std::max(numEves, numDsps);
        for (uint32_t i = 0; i < numPipes; ++i) {
            for (uint32_t j = 0; j < pipelineDepth; ++j) {
                eops.push_back(new EOP( {(*eveExecutor)[i % numEves],
                                          (*dspExecutor)[i % numDsps]}));
            }
        }

        allocateMemory(eops);

        uint32_t numEOPs = eops.size();

        // Process frames with EOs, with an extra numEOPs iterations to flush
        //   the pipeline.
        // TODO: Don't hardcode how long to run
        const uint32_t numIters = 100;
        for (uint32_t frameIdx = 0; frameIdx < numIters + numEOPs; ++frameIdx) {
            EOP* eop = eops[frameIdx % numEOPs];

            // Wait for EOP to finish processing last frame
            if (eop->ProcessFrameWait()) {
                writeProcessedFrame(eop, config);
            }

            if (readFrame(eop, frameIdx, config, cap)) {
                eop->ProcessFrameStartAsync();
            }
        }

        freeEOPMemory(eops);
        for (auto eop : eops) { delete eop; }
        delete eveExecutor;
        delete dspExecutor;

    } catch (tidl::Exception &e) {
        std::cerr << e.what() << std::endl;
        status = false;}

    return status;
}

/*
  THIS IS ALL OLD.
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
*/
