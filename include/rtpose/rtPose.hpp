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
#include "rtpose/renderFunctions.h"


class RTPose
{
	// global queues for I/O
	struct Global {
	    caffe::BlockingQueue<Frame> input_queue; //have to pop
	    caffe::BlockingQueue<Frame> output_queue; //have to pop
	    caffe::BlockingQueue<Frame> output_queue_ordered;
	    caffe::BlockingQueue<Frame> output_queue_mated;
	    std::priority_queue<int, std::vector<int>, std::greater<int> > dropped_index;
	    std::vector< std::string > image_list;
	    std::mutex mutex;
	    int part_to_show;
	    bool quit_threads;
	    // Parameters
	    float nms_threshold;
	    int connect_min_subset_cnt;
	    float connect_min_subset_score;
	    float connect_inter_threshold;
	    int connect_inter_min_above_threshold;

	    struct UIState {
	        UIState() :
	            is_fullscreen(0),
	            is_video_paused(0),
	            is_shift_down(0),
	            is_googly_eyes(0),
	            current_frame(0),
	            seek_to_frame(-1),
	            fps(0) {}
	        bool is_fullscreen;
	        bool is_video_paused;
	        bool is_shift_down;
	        bool is_googly_eyes;
	        int current_frame;
	        int seek_to_frame;
	        double fps;
	    };
	    UIState uistate;
	 };

	// network copy for each gpu thread
	struct NetCopy {
	    caffe::Net<float> *person_net;
	    std::vector<int> num_people;
	    int nblob_person;
	    int nms_max_peaks;
	    int nms_num_parts;
	    std::unique_ptr<ModelDescriptor> up_model_descriptor;
	    float* canvas; // GPU memory
	    float* joints; // GPU memory
	};

	struct ColumnCompare
	{
	    bool operator()(const std::vector<double>& lhs,
	                    const std::vector<double>& rhs) const
	    {
	        return lhs[2] > rhs[2];
	    }
	};


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

    void freeGPU();

private:
    // Global parameters
    int DISPLAY_RESOLUTION_WIDTH = 1280;
    int DISPLAY_RESOLUTION_HEIGHT = 720;
    int NET_RESOLUTION_WIDTH = 656;
    int NET_RESOLUTION_HEIGHT = 368;
    int BATCH_SIZE = 1;
    double SCALE_GAP = 0.3;
    double START_SCALE = 1;
    int NUM_GPU = 1;
    int start_device = 0;
    std::string PERSON_DETECTOR_CAFFEMODEL; //person detector
    std::string PERSON_DETECTOR_PROTO;      //person detector
    int MAX_PEOPLE = RENDER_MAX_PEOPLE;  // defined in render_functions.hpp
    int BOX_SIZE = 368;
    int BUFFER_SIZE = 4;    //affects latency
    int MAX_NUM_PARTS = 70;

    Global global;
    std::vector<NetCopy> net_copies;
    
    Frame frame;


};


#endif //RT_POSE_CPP
