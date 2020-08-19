#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#if __has_include("opencv2/core/cuda.hpp")
  #define USE_CUDA true
  #include <opencv2/core/cuda.hpp>
#else
  #define USE_CUDA false
#endif

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

//#define SSTR( x ) static_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()
#define TO_UPPER( str ) std::transform(str.begin(), str.end(), str.begin(), ::toupper)


using namespace std;
using namespace cv;

const auto IMG_WINDOW_NAME = "Football Detection";
const auto NET_INP_WIDTH = 416, NET_INP_HEIGHT = 416;
const auto NET_IMG_SIZE = Size(NET_INP_WIDTH, NET_INP_HEIGHT);
const auto NET_SCALE_FACTOR = 1/255.0;
const auto SCALAR_ZERO = Scalar(0, 0, 0);
const auto DETECTION_COLOUR = Scalar(255, 178, 50);
const auto TRACKING_COLOUR = Scalar(50, 170, 50);
const auto ERROR_COLOUR = Scalar(0, 0, 255);
auto confThreshold = 0.75, nmsThreshold = 0.4;
auto showDetectionLabel = false;
Mat currentFrame, frame;
string detectorType;
int detectedClassId;
dnn::Net net;
Rect2d detectedBbox;

struct ObjectClass {
  string className;
  Scalar colour;
};
vector<ObjectClass> classes;

static RNG rng(0xFFFFFFFF);
static Scalar randomColor() {
  int icolor = (unsigned) rng;
  return Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
}

bool track();
void detect();
void createDetector();
void detectAndTrack();
vector<String> getOutputsNames(const dnn::Net& net);
void postProcess(Mat& frame, const vector<Mat>& outs);
void drawPrediction(int classId, float conf, int left, int top, int right, int bottom, Scalar colour, Mat& frame);

int main(int argc, char** argv) {
  cout << "Press Esc to exit" << endl;

  const String keys =
        "{help h usage ? |                     | print this message                                                     }"
        "{video          |data/football_1.mp4  | name of the video fi                                                   }"
        "{detector       |YOLOV3               | detector: YOLOV3, YOLOV3-TINY                                          }"
        "{confThreshold  |0.75                 | confidence threshold                                                   }"
        "{labels         |false                | display object labels                                                  }";
  CommandLineParser parser(argc, argv, keys);
  parser.about("OpenCV YOLO Detector");
  if (parser.has("help")) {
      parser.printMessage();
      return 0;
  }
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }
  
  detectorType = parser.get<string>("detector");
  TO_UPPER(detectorType);
  confThreshold = parser.get<double>("confThreshold");
  showDetectionLabel = parser.get<bool>("labels");

  // Load names of classes
  string classesFile = "models/coco.names";
  ifstream ifs(classesFile.c_str());
  string line;
  while (getline(ifs, line))
    classes.push_back(ObjectClass{
      line,
      randomColor()
    });

  // Load network
  createDetector();

  string videoFile = parser.get<string>("video");
  VideoCapture cap(videoFile);
  if(!cap.isOpened()) {
    cout << "Error opening video file " << videoFile << endl;
    return -1;
  }

  cap >> currentFrame;
  if(currentFrame.empty()) { 
    cout << "Empty video file " << videoFile << endl;
    return -1;
  }

  namedWindow(IMG_WINDOW_NAME, WINDOW_AUTOSIZE);

  while(!currentFrame.empty()) {
    detectAndTrack();

    imshow(IMG_WINDOW_NAME, frame);

    auto c = static_cast<char>(waitKey(20));
    if(c == 27) {
      break;
    }

    cap >> currentFrame;
  }

  cout << "Video has ended, press any key" << endl;
  cap.release();
  destroyAllWindows();
  return 0;
}

void detectAndTrack() {
  frame = currentFrame.clone();
  detect();
}

void createDetector() {
  string modelConfiguration;
  string modelWeights;
  if (detectorType == "YOLOV3") {
    modelConfiguration = "models/yolov3.cfg";
    modelWeights = "models/yolov3.weights";
  } else if (detectorType == "YOLOV3-TINY") {
    modelConfiguration = "models/yolov3-tiny.cfg";
    modelWeights = "models/yolov3-tiny.weights";
  } else {
    cout << "INVALID DETECTOR SPECIFIED: " << detectorType << endl;
    throw -1;
  }
  net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);
  #if USE_CUDA
    net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
  #endif
}

void detect() {
  Mat blob;
  dnn::blobFromImage(frame, blob, NET_SCALE_FACTOR, NET_IMG_SIZE, SCALAR_ZERO, true, false);

  net.setInput(blob);

  // Runs the forward pass to get output of the output layers
  vector<Mat> outs;
  net.forward(outs, getOutputsNames(net));

  // Remove the bounding boxes with low confidence
  postProcess(frame, outs);

  // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
  vector<double> layersTimes;
  double freq = getTickFrequency() / 1000;
  double t = net.getPerfProfile(layersTimes) / freq;

  // Display tracker type and inference time on frame
  putText(frame, detectorType + " Detector", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, DETECTION_COLOUR, 2);
  string label = format("Inference time for a frame : %.2f ms", t);
  putText(frame, label, Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, DETECTION_COLOUR, 2);
}

// Draw the predicted bounding box
void drawPrediction(int classId, float conf, int left, int top, int right, int bottom, Scalar colour, Mat& frame) {
    //Draw a rectangle displaying the bounding box
    if (!classes.empty()) {
      colour = classes[classId].colour;
    }
    rectangle(frame, Point(left, top), Point(right, bottom), colour, 3);

    if (showDetectionLabel) {
      //Get the label for the class name and its confidence
      string label = format("%d - %.2f", classId, conf);
      if (!classes.empty()) {
          CV_Assert(classId < (int)classes.size());
          label = classes[classId].className + ":" + label;
      }

      //Display the label at the top of the bounding box
      int baseLine;
      Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      top = max(top, labelSize.height);
      rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
      putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
    }
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postProcess(Mat& frame, const vector<Mat>& outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i=0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j=0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        detectedBbox = boxes[idx];
        detectedClassId = classIds[idx];
        drawPrediction(detectedClassId, confidences[idx], 
              detectedBbox.x, detectedBbox.y, detectedBbox.x + detectedBbox.width, detectedBbox.y + detectedBbox.height,
              DETECTION_COLOUR, frame);
    }
}

// Get the names of the output layers
vector<String> getOutputsNames(const dnn::Net& net) {
  static vector<String> names;
  if (names.empty()) {
    cout << "Getting output layer names" << endl;
    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    vector<int> outLayers = net.getUnconnectedOutLayers();

    //get the names of all the layers in the network
    vector<String> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
      names[i] = layersNames[outLayers[i] - 1];
      cout << "Output layer " << i << " name is " << names[i] << endl;
    }
  }
  return names;
}
