#ifndef RT_POSE_CPP
#define RT_POSE_CPP

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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/cpm/frame.h"
#include "caffe/cpm/layers/imresize_layer.hpp"
#include "caffe/cpm/layers/nms_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/blocking_queue.hpp"

#include "rtpose/modelDescriptor.h"

class RTPose
{

public:
	RTPose(const std::string caffemodel, const std::string caffeproto, int gpu_id);

    double get_wall_time();

    void process_and_pad_image(float* target, cv::Mat oriImg, int tw, int th, bool normalize);

    void render(int gid, float *heatmaps /*GPU*/);

    int connectLimbsCOCO(
        std::vector< std::vector<double>> &subset,
        std::vector< std::vector< std::vector<double> > > &connection,
        const float *heatmap_pointer,
        const float *in_peaks,
        int max_peaks,
        float *joints,
        ModelDescriptor *model_descriptor);

    std::string getPoseEstimation(cv::Mat oriImg);

};


#endif //RT_POSE_CPP
