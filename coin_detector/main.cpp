#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

const auto IMG_WINDOW_NAME = "Coins";
const auto H_WINDOW_NAME = "Hue";
const auto S_WINDOW_NAME = "Saturation";
const auto V_WINDOW_NAME = "Value";
const auto BD_WINDOW_NAME = "Blob Detector";
const auto CA_WINDOW_NAME = "Contour Analysis";


void displayImage(Mat image, const string window_name=IMG_WINDOW_NAME) {
  imshow(window_name, image);
  waitKey();
}

Mat displayConnectedComponents(Mat &im) {
  // Make a copy of the image
  Mat imLabels = im.clone();

  // First let's find the min and max values in imLabels
  Point minLoc, maxLoc;
  double min, max;

  // The following line finds the min and max pixel values
  // and their locations in an image.
  minMaxLoc(imLabels, &min, &max, &minLoc, &maxLoc);

  // Normalize the image so the min value is 0 and max value is 255.
  imLabels = 255 * (imLabels - min) / (max - min);

  // Convert image to 8-bits
  imLabels.convertTo(imLabels, CV_8U);

  // Apply a color map
  Mat imColorMap;
  applyColorMap(imLabels, imColorMap, COLORMAP_JET);

  return imColorMap;
}

int main(int argc, char** argv) {
  cout << "Press Esc to exit" << endl;

  const String keys =
        "{help h usage ? |                | print this message           }"
        "{img            |data/CoinsA.png | path to the image to process }";
  CommandLineParser parser(argc, argv, keys);
  parser.about("OpenCV Coin Detector");
  if (parser.has("help")) {
      parser.printMessage();
      return 0;
  }
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }
  
  string imagePath = parser.get<string>("img");

  Mat image = imread(imagePath);
  namedWindow(IMG_WINDOW_NAME, WINDOW_AUTOSIZE);	
  displayImage(image);
	
  // Convert image to grayscale
  Mat imageGray;
  cvtColor(image, imageGray, COLOR_BGR2GRAY);
  displayImage(imageGray);
	
  // Split cell into channels
  // With RGB system this will only work for specific images
  Mat hsvImage;
  cvtColor(image, hsvImage, COLOR_BGR2HSV);
  Mat hsvChannels[4], bgrChannels[4];
  split(hsvImage, hsvChannels);
  Mat imageH = hsvChannels[0];
  Mat imageS = hsvChannels[1];
  Mat imageV = hsvChannels[2];

  // Perform thresholding
  // By the look of the image below we need inv or to-zero-inv thresholding
  Mat thi1, thi2, thi3;
  int thresh=200, maxVal=255;  // Use a UI slider to allow customization
  Mat dst;
  threshold(imageS, dst, thresh, maxVal, THRESH_BINARY);
  displayImage(dst);
	
  // Perform morphological operations
  int kdSize = 3; // Allow user to select size an shape of kernel
  Mat kerneld = getStructuringElement(MORPH_ELLIPSE, Size(kdSize, kdSize));
  Mat imageDilated;
  dilate(dst, imageDilated, kerneld, Point(-1,-1), 3);

  displayImage(imageDilated);
	
  // Get structuring element/kernel which will be used for dilation
  int keSize = 3; // Allow user to select size an shape of kernel
  Mat kernele = getStructuringElement(MORPH_ELLIPSE, Size(keSize, keSize));
  Mat imageEroded;
  erode(imageDilated, imageEroded, kernele, Point(-1,-1), 4);
  displayImage(imageEroded);
	
  // Setup SimpleBlobDetector parameters.
  SimpleBlobDetector::Params params;
  params.blobColor = 0;
  params.minDistBetweenBlobs = 2;
  // Filter by Area
  params.filterByArea = false;
  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = 0.8;
  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = 0.8;
  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = 0.8;

  // Set up detector with params
  Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	
  // Detect blobs
  vector<KeyPoint> keypoints;
  detector->detect(imageEroded, keypoints);

  // Print number of coins detected
  cout << "Number of coins detected = " << keypoints.size() << endl;
	
  // Mark coins using image annotation concepts we have studied so far
  Mat imageCopy = image.clone();
  int x, y;
  int radius;
  double diameter;
  for (int i=0; i < keypoints.size(); i++) {
    KeyPoint k = keypoints[i];
    Point keyPt;
    keyPt = k.pt;
    x = (int)keyPt.x;
    y = (int)keyPt.y;
    circle(imageCopy, Point(x,y), 5, Scalar(255,0,0), -1);
    diameter = k.size;
    radius = (int)diameter/2.0;
    circle(imageCopy, Point(x,y), radius,Scalar(0,255,0), 2);
  }
  displayImage(imageCopy, BD_WINDOW_NAME);
	
  Mat inverseImageEroded = 255-imageEroded;
  Mat imLabels;
  int nComponents = connectedComponents(inverseImageEroded, imLabels);
  Mat colorMap = displayConnectedComponents(imLabels);
  cout << "Number of connected components detected = " << nComponents << endl;
  // I think it makes sense trying to use connected components to find out the number of coins. Hoever this method
  // won't be pricese, it will include the noise around the border and the backgorund I think
  displayImage(colorMap, CA_WINDOW_NAME);
	
  // Detect coins using Contour Detection
  // Find all contours in the image
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(inverseImageEroded, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	
  // Print the number of contours found
  cout << "Number of contours found = " << contours.size() << endl;
	
  // Draw all contours
  Mat countourImage = image.clone();
  Moments M;
  for (size_t i=0; i < contours.size(); i++){
    M = moments(contours[i]);
    x = int(M.m10/double(M.m00));
    y = int(M.m01/double(M.m00));
    circle(countourImage, Point(x,y), 10, Scalar(0,0,255), -1);
  }
  drawContours(countourImage, contours, -1, Scalar(0,0,0), 6);
  displayImage(countourImage, CA_WINDOW_NAME);
	

  // Print area and perimeter of all contours
  const auto minArea = 2.0;
  const auto maxArea = 20000.0;
  vector<int> removals;
  double area, perimeter;
  for (size_t i=0; i < contours.size(); i++) {
    area = contourArea(contours[i]);
    perimeter = arcLength(contours[i], true);
    cout << "Contour #" << i+1 << " has area = " << area << " and perimeter = " << perimeter << " hierarchy = " << hierarchy[i] << endl;
    // Note that I'm finding the minimum since my image seems to have detcted a small portion of the outtermost
    // contour. We also remo hierarchy[i][3] which is contours which has a parent contour index
    if (hierarchy[i][3] != -1 || area < minArea || area > maxArea) {
      removals.push_back(i);
    }
  }

  Mat cleanCountourImage = image.clone();
  drawContours(cleanCountourImage, contours, -1, Scalar(0,255,0), 6);

  // Fit circles on coins
  cout << "Number of coins detected = " << contours.size() << endl;
  Mat finalCountourImage = image.clone();
  for (size_t i=0; i < contours.size(); i++) {
    M = moments(contours[i]);
    x = int(M.m10/double(M.m00));
    y = int(M.m01/double(M.m00));
    circle(finalCountourImage, Point(x,y), 10, Scalar(0,0,255), -1);
  }
  drawContours(finalCountourImage, contours, -1, Scalar(255,0,0), 6);
  displayImage(finalCountourImage, CA_WINDOW_NAME);

  destroyAllWindows();
  return 0;
}