#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include <sys/types.h>
#include <dirent.h>
// #include <filesystem> 
#include <regex>
#include <string>
#include <algorithm>


using namespace std;
using namespace cv;

static RNG rng(0xFFFFFFFF);
static Scalar randomColor() {
  int icolor = (unsigned) rng;
  return Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
}

const auto BASE_IMG_WINDOW_NAME = "Template Matching Base Image";
const auto DETECTION_IMG_WINDOW_NAME = "Template Matching Detection";
const auto DEFAULT_NON_MAX_SUPRESSION_THRESHOLD = 0.1;

struct Template {
  string name;
  string label;
  Scalar colour;
  Mat template_mat;
  double match_threshold;
};


struct TemplateMatch {
  string label;
  Scalar colour;
  Point location;
  int width;
  int height;
  double match_value;
};

vector<Template> load_templates(const string& location, const double match_threshold);
vector<TemplateMatch> detect(const Mat& base_image, const vector<Template>& templates, const int match_method);
Mat draw_detections(const Mat& base_image, const vector<TemplateMatch>& detections);
vector<TemplateMatch> non_max_suppression(const vector<TemplateMatch>& detections, double non_max_suppression_threshold=DEFAULT_NON_MAX_SUPRESSION_THRESHOLD);

int main(int argc, char** argv) {
  cout << "Press Esc to exit" << endl;

  const String keys =
        "{help h usage ? |          | print this message                        }"
        "{data_dir       |data      | directory where the images are stored     }"
        "{thresh         |0.6       | match threshold                           }";
  CommandLineParser parser(argc, argv, keys);
  parser.about("OpenCV Template Matching");
  if (parser.has("help")) {
      parser.printMessage();
      return 0;
  }
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  // Load templates
  auto match_threshold = parser.get<double>("thresh");
  auto data_dir = parser.get<string>("data_dir");
  vector<Template> templates = load_templates(data_dir, match_threshold);
  // hack the annoying template with loads of false positives
  //templates[1].match_threshold = 0.8; // These are hypermarameters that could read from a text file

  // Load base image
  Mat base_image = imread(data_dir + "/base.jpg");

  namedWindow(BASE_IMG_WINDOW_NAME, WINDOW_AUTOSIZE);
  namedWindow(DETECTION_IMG_WINDOW_NAME, WINDOW_AUTOSIZE);

  while (true) {
     imshow(BASE_IMG_WINDOW_NAME, base_image);

     auto detections = detect(base_image, templates, TM_CCOEFF_NORMED);
     auto detections_imgage = draw_detections(base_image, detections);

     imshow(DETECTION_IMG_WINDOW_NAME, detections_imgage);

     auto c = static_cast<char>(waitKey(20));
     if (c == 27) {
       break;
    }
  }

  destroyAllWindows();
  return 0;
}

vector<Template> load_templates(const string& location, const double match_threshold) {
  vector<Template> templates;
  const regex template_regex(location + "/template_.*");
  smatch template_match;
  DIR* dirp = opendir(location.c_str());
  struct dirent * dp;
  while ((dp = readdir(dirp)) != NULL) {
    auto filename = dp->d_name;
    auto path = location + "/" + filename;
    if (regex_match(path, template_match, template_regex)) {
      cout << "Found template: " << path << endl;
      auto mat = imread(path);
      templates.push_back(Template {
        path,
        filename,
        randomColor(),
        mat,
        match_threshold
      });
    }
  }
  closedir(dirp);
  // for (const auto & entry : filesystem::directory_iterator(location)) {
  //   auto path = entry.path().string();
  //   if (regex_match(path, template_match, template_regex)) {
  //     cout << "Found template: " << entry << endl;
  //     auto mat = imread(path);
  //     templates.push_back(Template {
  //       path,
  //       entry.path().filename(),
  //       randomColor(),
  //       mat,
  //       match_threshold
  //     });
  //   }
  // }
  return templates;
}

vector<TemplateMatch> detect(const Mat& base_image, const vector<Template>& templates, const int match_method) {
  vector<TemplateMatch> detections;
  for (const auto& t : templates) {
    Mat result;
    matchTemplate(base_image, t.template_mat, result, match_method);
    // imshow(DETECTION_IMG_WINDOW_NAME, result);
    // auto c = static_cast<char>(waitKey());

    if (match_method != TM_CCOEFF_NORMED) {
      normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    }

    // Options 1: Localizing the best match with minMaxLoc
    // double minVal, maxVal;
    // Point minLoc, maxLoc, matchLoc;
    // minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    // // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    // if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) {
    //   matchLoc = minLoc;
    // } else {
    //   matchLoc = maxLoc;
    // }
    // detections.push_back(TemplateMatch {
    //   t.label,
    //   t.colour,
    //   matchLoc,
    //   t.template_mat.size().width,
    //   t.template_mat.size().height
    // });

    // Options 2: get all matches above certain threshold
    // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) {
      threshold(result, result, t.match_threshold, 1.0, CV_THRESH_TOZERO_INV);
    } else {
      threshold(result, result, t.match_threshold, 1.0, CV_THRESH_TOZERO);
    }
    // imshow(DETECTION_IMG_WINDOW_NAME, result);
    // auto c = static_cast<char>(waitKey());
    Mat locations;
    findNonZero(result, locations);
    for (int i=0; i<locations.total(); ++i) {
      auto location = locations.at<Point>(i);
      detections.push_back(TemplateMatch {
        t.label,
        t.colour,
        location,
        t.template_mat.size().width,
        t.template_mat.size().height,
        result.at<double>(location)
      });
    }
  }
  return non_max_suppression(detections);
}

bool template_match_desc(const TemplateMatch& a, const TemplateMatch& b) {
  return a.match_value > b.match_value;
}

double union_area(const TemplateMatch& a, const TemplateMatch& b, const double intersection_area) {
  return (a.width * a.height) + (b.width * b.height) - intersection_area;
}

double intersection_area(const TemplateMatch& a, const TemplateMatch& b) {
  auto x = max(a.location.x, b.location.x);
  auto y = max(a.location.y, b.location.y);
  auto w = min(a.location.x+a.width, b.location.x+b.width) - x;
  auto h = min(a.location.y+a.height, b.location.y+b.height) - y;
  if (w<0 || h<0) return 0.0;
  return w * h;
}

double compute_iou(const TemplateMatch& a, const TemplateMatch& b) {
  auto intersection = intersection_area(a, b);
  auto u = union_area(a, b, intersection);
  return intersection == 0.0? 0.0 : intersection/union_area(a, b, intersection);
}

vector<TemplateMatch> non_max_suppression(const vector<TemplateMatch>& detections, double non_max_suppression_threshold) {
  vector<TemplateMatch> descendingDetections{ detections };
  sort(descendingDetections.begin(), descendingDetections.end(), template_match_desc);
  vector<TemplateMatch> filtered;
  for (const auto& m : descendingDetections) {
    bool overlap = false;
    for (const auto& f : filtered) {
      auto iou = compute_iou(m, f);
      if (iou > non_max_suppression_threshold) {
        overlap = true;
        break;
      }
    }
    if (!overlap) {
      filtered.push_back(m);
    }
  }
  return filtered;
}

Mat draw_detections(const Mat& base_image, const vector<TemplateMatch>& detections) {
  Mat result = base_image.clone();
  for(const auto& d : detections) {
    rectangle(result, d.location, Point(d.location.x + d.width , d.location.y + d.height), d.colour, 2, 8, 0);
  }
  return result;
}
