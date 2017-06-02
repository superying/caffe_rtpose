#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <utility> //std::pair

#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <stdio.h>  // snprintf
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

#include <boost/thread/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
// #include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/cpm/frame.h"
#include "caffe/cpm/layers/imresize_layer.hpp"
#include "caffe/cpm/layers/nms_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/blocking_queue.hpp"
// #include "caffe/util/render_functions.hpp"
// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/db.hpp"
// #include "caffe/util/io.hpp"
// #include "caffe/util/benchmark.hpp"

#include "rtpose/modelDescriptor.h"
#include "rtpose/modelDescriptorFactory.h"
#include "rtpose/renderFunctions.h"
#include "rtpose/rtPose.hpp"
#include "json/json.h"

int main(int argc, char *argv[]) {

	RTPose* rp = new RTPose("model/coco/pose_iter_440000.caffemodel", "model/coco/pose_deploy_linevec.prototxt", 0);

	cv::Mat img_mat = cv::imread("images/test.jpg", CV_LOAD_IMAGE_COLOR);

	std::cout << "init ready! \n";

	//usleep(30*1000.0);

	//std::cout << "continue! \n";

	for(int i=0; i<100; i++) {
		std::string res_json = rp->getPoseEstimation(img_mat);
	}


	std::cout << "Run 100 times! \n";

	//usleep(30*1000.0);

	//std::cout << "continue! \n";

	
	rp->freeGPU();

	//delete rp;

	//for(int j=0; j<10; j++) {
	//	usleep(10*1000.0);
	//	std::cout << "usleeping! \n";
	//}

	
	
	RTPose rp2("model/coco/pose_iter_440000.caffemodel", "model/coco/pose_deploy_linevec.prototxt", 0);
	

	cv::Mat img_mat2 = cv::imread("images/test.jpg", CV_LOAD_IMAGE_COLOR);	

	std::cout << "init ready! \n";
	
	for(int k=0; k<100; k++) {
                std::string res_json2 = rp2.getPoseEstimation(img_mat2);
        }
	

	std::cout << "Run 100 times! \n";


	//for(int m=0; m<10; m++) {
        //        usleep(10*1000.0);
        //        std::cout << "usleeping! \n";
        //}
	


	//std::cout << "Free GPU! \n";

	//usleep(30*1000.0);

	//std::cout << "continue! \n";

	//free(rp);

	//std::cout << "Free RTPose! \n";

	//usleep(30*1000.0);

	//std::cout << "continue! \n";
	


	//cv::Mat img_mat2 = cv::imread("images/test2.jpg", CV_LOAD_IMAGE_COLOR);

	//std::string res_json2 = rp.getPoseEstimation(img_mat2);

	//std::cout << "Output2 json result.\n";

	//std::cout << res_json2;

	//std::cout << "\n";

	//Json::Value root;
	//Json::Reader reader;
	//reader.parse(res_json, root);

	//std::cout << "Output Json Parse result.\n";
	//std::cout << root["version"];
	//std::cout << "\n";
	//std::cout << root["bodies"][0]["joints"][3];

    return 1;
}
