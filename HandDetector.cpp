#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <iostream>
#include <vector>
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/executor.h"
#include "mediapipe/framework/tool/executor_util.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_videoio_inc.h"

namespace mediapipe {
    #include "mediapipe/calculators/core/concatenate_vector_calculator.pb.h"
    #include "mediapipe/calculators/image/convolution_calculator.pb.h"
    #include "mediapipe/calculators/image/matrix_to_uint8_calculator.pb.h"
    #include "mediapipe/calculators/image/rect_to_render_data_calculator.pb.h"
    #include "mediapipe/calculators/image/resize_calculator.pb.h"
    #include "mediapipe/calculators/image/rotated_rect_to_rect_calculator.pb.h"
    #include "mediapipe/calculators/image/thresholding_calculator.pb.h"
}

class HandDetector {
public:
    HandDetector(bool mode = false, int maxHands = 2, float detectionCon = 0.5, float trackCon = 0.5)
        : mode(mode), maxHands(maxHands), detectionCon(detectionCon), trackCon(trackCon) {
            mpHands = std::make_shared<mediapipe::Hands>();
            mpHands->Initialize();
    }

    cv::Mat findHands(cv::Mat img, bool draw = true) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        auto results = mpHands->Process(mediapipe::formats::MatToPacket(img));
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        return img;
    }

    std::vector<std::vector<int>> findPosition(cv::Mat img, int handNo = 0, bool draw = true) {
        std::vector<std::vector<int>> lmlist;
        return lmlist;
    }

private:
    bool mode;
    int maxHands;
    float detectionCon;
    float trackCon;
    std::shared_ptr<mediapipe::Hands> mpHands;
};

int main() {
    int pTime = 0;
    int cTime = 0;
    cv::VideoCapture cap(0);
    HandDetector detector;

    while (true) {
        cv::Mat img;
        cap.read(img);
        img = detector.findHands(img);
        auto lmlist = detector.findPosition(img);
        if (!lmlist.empty()) {
            std::cout << lmlist[4][0] << ", " << lmlist[4][1] << std::endl;
        }

        cTime = time(0);
        double fps = 1 / (cTime - pTime);
        pTime = cTime;

        cv::putText(img, std::to_string(fps), cv::Point(10, 70), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 0, 255), 3);

        cv::imshow("Image", img);
        cv::waitKey(1);
    }

    return 0;
}
