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
#include "rtpose/modelDescriptorFactory.h"
#include "rtpose/renderFunctions.h"

#include "rtpose/rtPose.hpp"


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
const auto MAX_PEOPLE = RENDER_MAX_PEOPLE;  // defined in render_functions.hpp
const auto BOX_SIZE = 368;
const auto BUFFER_SIZE = 4;    //affects latency
const auto MAX_NUM_PARTS = 70;

Global global;
std::vector<NetCopy> net_copies;


RTPose::RTPose(const std::string caffemodel, const std::string caffeproto, int gpu_id)
{
	PERSON_DETECTOR_CAFFEMODEL = caffemodel;
	PERSON_DETECTOR_PROTO = caffeproto;
	net_copies = std::vector<NetCopy>(NUM_GPU);
	start_device = gpu_id;
}


double RTPose::get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time,NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}


void RTPose::process_and_pad_image(float* target, cv::Mat oriImg, int tw, int th, bool normalize) {
    int ow = oriImg.cols;
    int oh = oriImg.rows;
    int offset2_target = tw * th;

    int padw = (tw-ow)/2;
    int padh = (th-oh)/2;
    //LOG(ERROR) << " padw " << padw << " padh " << padh;
    CHECK_GE(padw,0) << "Image too big for target size.";
    CHECK_GE(padh,0) << "Image too big for target size.";
    //parallel here
    unsigned char* pointer = (unsigned char*)(oriImg.data);

    for(int c = 0; c < 3; c++) {
        for(int y = 0; y < th; y++) {
            int oy = y - padh;
            for(int x = 0; x < tw; x++) {
                int ox = x - padw;
                if (ox>=0 && ox < ow && oy>=0 && oy < oh ) {
                    if (normalize)
                        target[c * offset2_target + y * tw + x] = float(pointer[(oy * ow + ox) * 3 + c])/256.0f - 0.5f;
                    else
                        target[c * offset2_target + y * tw + x] = float(pointer[(oy * ow + ox) * 3 + c]);
                }
                else {
                    target[c * offset2_target + y * tw + x] = 0;
                }
            }
        }
    }
}

void RTPose::render(int gid, float *heatmaps /*GPU*/) {
    float* centers = 0;
    float* poses    = net_copies[gid].joints;

    double tic = get_wall_time();
    if (net_copies[gid].up_model_descriptor->get_number_parts()==15) {
        render_mpi_parts(net_copies[gid].canvas, DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT,
        heatmaps, BOX_SIZE, centers, poses, net_copies[gid].num_people, global.part_to_show);
    } else if (net_copies[gid].up_model_descriptor->get_number_parts()==18) {
        if (global.part_to_show-1<=net_copies[gid].up_model_descriptor->get_number_parts()) {
            render_coco_parts(net_copies[gid].canvas,
            DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT,
            NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT,
            heatmaps, BOX_SIZE, centers, poses,
            net_copies[gid].num_people, global.part_to_show, global.uistate.is_googly_eyes);
        } else {
            int aff_part = ((global.part_to_show-1)-net_copies[gid].up_model_descriptor->get_number_parts()-1)*2;
            int num_parts_accum = 1;
            if (aff_part==0) {
                num_parts_accum = 19;
            } else {
                aff_part = aff_part-2;
                }
                aff_part += 1+net_copies[gid].up_model_descriptor->get_number_parts();
                render_coco_aff(net_copies[gid].canvas, DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT,
                heatmaps, BOX_SIZE, centers, poses, net_copies[gid].num_people, aff_part, num_parts_accum);
        }
    }
    VLOG(2) << "Render time " << (get_wall_time()-tic)*1000.0 << " ms.";
}


int RTPose::connectLimbsCOCO(
    std::vector< std::vector<double>> &subset,
    std::vector< std::vector< std::vector<double> > > &connection,
    const float *heatmap_pointer,
    const float *in_peaks,
    int max_peaks,
    float *joints,
    ModelDescriptor *model_descriptor) {
        /* Parts Connection ---------------------------------------*/
        const auto num_parts = model_descriptor->get_number_parts();
        const auto limbSeq = model_descriptor->get_limb_sequence();
        const auto mapIdx = model_descriptor->get_map_idx();
        const auto number_limb_seq = model_descriptor->number_limb_sequence();

        CHECK_EQ(num_parts, 18) << "Wrong connection function for model";
        CHECK_EQ(number_limb_seq, 19) << "Wrong connection function for model";

        int SUBSET_CNT = num_parts+2;
        int SUBSET_SCORE = num_parts+1;
        int SUBSET_SIZE = num_parts+3;

        const int peaks_offset = 3*(max_peaks+1);

        const float *peaks = in_peaks;
        subset.clear();
        connection.clear();

        for(int k = 0; k < number_limb_seq; k++) {
            const float* map_x = heatmap_pointer + mapIdx[2*k] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
            const float* map_y = heatmap_pointer + mapIdx[2*k+1] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;

            const float* candA = peaks + limbSeq[2*k]*peaks_offset;
            const float* candB = peaks + limbSeq[2*k+1]*peaks_offset;

            std::vector< std::vector<double> > connection_k;
            int nA = candA[0];
            int nB = candB[0];

            // add parts into the subset in special case
            if (nA ==0 && nB ==0) {
                continue;
            } else if (nA ==0) {
                for(int i = 1; i <= nB; i++) {
                    int num = 0;
                    int indexB = limbSeq[2*k+1];
                    for(int j = 0; j < subset.size(); j++) {
                            int off = limbSeq[2*k+1]*peaks_offset + i*3 + 2;
                            if (subset[j][indexB] == off) {
                                    num = num+1;
                                    continue;
                            }
                    }
                    if (num!=0) {
                        //LOG(INFO) << " else if (nA==0) shouldn't have any nB already assigned?";
                    } else {
                        std::vector<double> row_vec(SUBSET_SIZE, 0);
                        row_vec[ limbSeq[2*k+1] ] = limbSeq[2*k+1]*peaks_offset + i*3 + 2; //store the index
                        row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                        row_vec[SUBSET_SCORE] = candB[i*3+2]; //second last number in each row is the total score
                        subset.push_back(row_vec);
                    }
                    //LOG(INFO) << "nA==0 New subset on part " << k << " subsets: " << subset.size();
                }
                continue;
            } else if (nB ==0) {
                for(int i = 1; i <= nA; i++) {
                    int num = 0;
                    int indexA = limbSeq[2*k];
                    for(int j = 0; j < subset.size(); j++) {
                            int off = limbSeq[2*k]*peaks_offset + i*3 + 2;
                            if (subset[j][indexA] == off) {
                                    num = num+1;
                                    continue;
                            }
                    }
                    if (num==0) {
                        std::vector<double> row_vec(SUBSET_SIZE, 0);
                        row_vec[ limbSeq[2*k] ] = limbSeq[2*k]*peaks_offset + i*3 + 2; //store the index
                        row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                        row_vec[SUBSET_SCORE] = candA[i*3+2]; //second last number in each row is the total score
                        subset.push_back(row_vec);
                        //LOG(INFO) << "nB==0 New subset on part " << k << " subsets: " << subset.size();
                    } else {
                        //LOG(INFO) << "nB==0 discarded would have added";
                    }
                }
                continue;
            }

            std::vector< std::vector<double>> temp;
            const int num_inter = 10;

            for(int i = 1; i <= nA; i++) {
                for(int j = 1; j <= nB; j++) {
                    float s_x = candA[i*3];
                    float s_y = candA[i*3+1];
                    float d_x = candB[j*3] - candA[i*3];
                    float d_y = candB[j*3+1] - candA[i*3+1];
                    float norm_vec = sqrt( d_x*d_x + d_y*d_y );
                    if (norm_vec<1e-6) {
                        // The peaks are coincident. Don't connect them.
                        continue;
                    }
                    float vec_x = d_x/norm_vec;
                    float vec_y = d_y/norm_vec;

                    float sum = 0;
                    int count = 0;

                    for(int lm=0; lm < num_inter; lm++) {
                        int my = round(s_y + lm*d_y/num_inter);
                        int mx = round(s_x + lm*d_x/num_inter);
                        if (mx>=NET_RESOLUTION_WIDTH) {
                            //LOG(ERROR) << "mx " << mx << "out of range";
                            mx = NET_RESOLUTION_WIDTH-1;
                        }
                        if (my>=NET_RESOLUTION_HEIGHT) {
                            //LOG(ERROR) << "my " << my << "out of range";
                            my = NET_RESOLUTION_HEIGHT-1;
                        }
                        CHECK_GE(mx,0);
                        CHECK_GE(my,0);
                        int idx = my * NET_RESOLUTION_WIDTH + mx;
                        float score = (vec_x*map_x[idx] + vec_y*map_y[idx]);
                        if (score > global.connect_inter_threshold) {
                            sum = sum + score;
                            count ++;
                        }
                    }
                    //float score = sum / count; // + std::min((130/dist-1),0.f)

                    if (count > global.connect_inter_min_above_threshold) {//num_inter*0.8) { //thre/2
                        // parts score + cpnnection score
                        std::vector<double> row_vec(4, 0);
                        row_vec[3] = sum/count + candA[i*3+2] + candB[j*3+2]; //score_all
                        row_vec[2] = sum/count;
                        row_vec[0] = i;
                        row_vec[1] = j;
                        temp.push_back(row_vec);
                    }
                }
            }

            //** select the top num connection, assuming that each part occur only once
            // sort rows in descending order based on parts + connection score
            if (temp.size() > 0)
                std::sort(temp.begin(), temp.end(), ColumnCompare());

            int num = std::min(nA, nB);
            int cnt = 0;
            std::vector<int> occurA(nA, 0);
            std::vector<int> occurB(nB, 0);

            for(int row =0; row < temp.size(); row++) {
                if (cnt==num) {
                    break;
                }
                else{
                    int i = int(temp[row][0]);
                    int j = int(temp[row][1]);
                    float score = temp[row][2];
                    if ( occurA[i-1] == 0 && occurB[j-1] == 0 ) { // && score> (1+thre)
                        std::vector<double> row_vec(3, 0);
                        row_vec[0] = limbSeq[2*k]*peaks_offset + i*3 + 2;
                        row_vec[1] = limbSeq[2*k+1]*peaks_offset + j*3 + 2;
                        row_vec[2] = score;
                        connection_k.push_back(row_vec);
                        cnt = cnt+1;
                        occurA[i-1] = 1;
                        occurB[j-1] = 1;
                    }
                }
            }

            //** cluster all the joints candidates into subset based on the part connection
            // initialize first body part connection 15&16
            if (k==0) {
                std::vector<double> row_vec(num_parts+3, 0);
                for(int i = 0; i < connection_k.size(); i++) {
                    double indexB = connection_k[i][1];
                    double indexA = connection_k[i][0];
                    row_vec[limbSeq[0]] = indexA;
                    row_vec[limbSeq[1]] = indexB;
                    row_vec[SUBSET_CNT] = 2;
                    // add the score of parts and the connection
                    row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                    //LOG(INFO) << "New subset on part " << k << " subsets: " << subset.size();
                    subset.push_back(row_vec);
                }
            }/* else if (k==17 || k==18) { // TODO: Check k numbers?
                //   %add 15 16 connection
                for(int i = 0; i < connection_k.size(); i++) {
                    double indexA = connection_k[i][0];
                    double indexB = connection_k[i][1];

                    for(int j = 0; j < subset.size(); j++) {
                    // if subset(j, indexA) == partA(i) && subset(j, indexB) == 0
                    //         subset(j, indexB) = partB(i);
                    // elseif subset(j, indexB) == partB(i) && subset(j, indexA) == 0
                    //         subset(j, indexA) = partA(i);
                    // end
                        if (subset[j][limbSeq[2*k]] == indexA && subset[j][limbSeq[2*k+1]]==0) {
                            subset[j][limbSeq[2*k+1]] = indexB;
                        } else if (subset[j][limbSeq[2*k+1]] == indexB && subset[j][limbSeq[2*k]]==0) {
                            subset[j][limbSeq[2*k]] = indexA;
                        }
                }
                continue;
            }
        }*/ else{
            if (connection_k.size()==0) {
                continue;
            }

            // A is already in the subset, find its connection B
            for(int i = 0; i < connection_k.size(); i++) {
                int num = 0;
                double indexA = connection_k[i][0];
                double indexB = connection_k[i][1];

                for(int j = 0; j < subset.size(); j++) {
                    if (subset[j][limbSeq[2*k]] == indexA) {
                        subset[j][limbSeq[2*k+1]] = indexB;
                        num = num+1;
                        subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
                        subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] + peaks[int(indexB)] + connection_k[i][2];
                    }
                }
                // if can not find partA in the subset, create a new subset
                if (num==0) {
                    //LOG(INFO) << "New subset on part " << k << " subsets: " << subset.size();
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[limbSeq[2*k]] = indexA;
                    row_vec[limbSeq[2*k+1]] = indexB;
                    row_vec[SUBSET_CNT] = 2;
                    row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                    subset.push_back(row_vec);
                }
            }
        }
    }

    //** joints by deleteing some rows of subset which has few parts occur
    int cnt = 0;
    for(int i = 0; i < subset.size(); i++) {
        if (subset[i][SUBSET_CNT]<1) {
            LOG(INFO) << "BAD SUBSET_CNT";
        }
        if (subset[i][SUBSET_CNT]>=global.connect_min_subset_cnt && (subset[i][SUBSET_SCORE]/subset[i][SUBSET_CNT])>global.connect_min_subset_score) {
            for(int j = 0; j < num_parts; j++) {
                int idx = int(subset[i][j]);
                if (idx) {
                    joints[cnt*num_parts*3 + j*3 +2] = peaks[idx];
                    joints[cnt*num_parts*3 + j*3 +1] = peaks[idx-1]* DISPLAY_RESOLUTION_HEIGHT/ (float)NET_RESOLUTION_HEIGHT;//(peaks[idx-1] - padh) * ratio_h;
                    joints[cnt*num_parts*3 + j*3] = peaks[idx-2]* DISPLAY_RESOLUTION_WIDTH/ (float)NET_RESOLUTION_WIDTH;//(peaks[idx-2] -padw) * ratio_w;
                }
                else{
                    joints[cnt*num_parts*3 + j*3 +2] = 0;
                    joints[cnt*num_parts*3 + j*3 +1] = 0;
                    joints[cnt*num_parts*3 + j*3] = 0;
                }
            }
            cnt++;
            if (cnt==MAX_PEOPLE) break;
        }
    }

    return cnt;
}


std::string RTPose::getPoseEstimation(cv::Mat oriImg) {

	DISPLAY_RESOLUTION_WIDTH = oriImg.cols;
	DISPLAY_RESOLUTION_HEIGHT = oriImg.rows;
	NET_RESOLUTION_WIDTH = 656;
	NET_RESOLUTION_HEIGHT = 368;

	//warm up, load caffe model in GPU
	caffe::Caffe::SetDevice(0); //cudaSetDevice(device_id) inside
	caffe::Caffe::set_mode(caffe::Caffe::GPU); //

	net_copies[0].person_net = new caffe::Net<float>(PERSON_DETECTOR_PROTO, caffe::TEST);
	net_copies[0].person_net->CopyTrainedLayersFrom(PERSON_DETECTOR_CAFFEMODEL);

	net_copies[0].nblob_person = net_copies[0].person_net->blob_names().size();
	net_copies[0].num_people.resize(BATCH_SIZE);
	const std::vector<int> shape { {BATCH_SIZE, 3, NET_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH} };

	net_copies[0].person_net->blobs()[0]->Reshape(shape);
	net_copies[0].person_net->Reshape();

	caffe::NmsLayer<float> *nms_layer = (caffe::NmsLayer<float>*)net_copies[0].person_net->layer_by_name("nms").get();
	net_copies[0].nms_max_peaks = nms_layer->GetMaxPeaks();


	caffe::ImResizeLayer<float> *resize_layer =
		(caffe::ImResizeLayer<float>*)net_copies[0].person_net->layer_by_name("resize").get();

	resize_layer->SetStartScale(START_SCALE);
	resize_layer->SetScaleGap(SCALE_GAP);

	net_copies[0].nms_max_peaks = nms_layer->GetMaxPeaks();

	net_copies[0].nms_num_parts = nms_layer->GetNumParts();
	CHECK_LE(net_copies[0].nms_num_parts, MAX_NUM_PARTS)
		<< "num_parts in NMS layer (" << net_copies[0].nms_num_parts << ") "
		<< "too big ( MAX_NUM_PARTS )";

	if (net_copies[0].nms_num_parts==18) {
		ModelDescriptorFactory::createModelDescriptor(ModelDescriptorFactory::Type::COCO_18, net_copies[0].up_model_descriptor);
		global.nms_threshold = 0.05;
		global.connect_min_subset_cnt = 3;
		global.connect_min_subset_score = 0.4;
		global.connect_inter_threshold = 0.050;
		global.connect_inter_min_above_threshold = 9;
	} else {
		CHECK(0) << "Unknown number of parts! Couldn't set COCO model";
	}

	net_copies[0].person_net->ForwardFrom(0);
	cudaMalloc(&net_copies[0].canvas, DISPLAY_RESOLUTION_WIDTH * DISPLAY_RESOLUTION_HEIGHT * 3 * sizeof(float));
	cudaMalloc(&net_copies[0].joints, MAX_NUM_PARTS*3*MAX_PEOPLE * sizeof(float) );




	//pre-process input Image Mat
	cv::Mat image_uchar;
	cv::Mat image_uchar_orig;
	cv::Mat image_uchar_prev;
	image_uchar_orig = oriImg;
	double scale = 0;
	if (image_uchar_orig.cols/(double)image_uchar_orig.rows>DISPLAY_RESOLUTION_WIDTH/(double)DISPLAY_RESOLUTION_HEIGHT) {
		scale = DISPLAY_RESOLUTION_WIDTH/(double)image_uchar_orig.cols;
	} else {
		scale = DISPLAY_RESOLUTION_HEIGHT/(double)image_uchar_orig.rows;
	}
	cv::Mat M = cv::Mat::eye(2,3,CV_64F);
	M.at<double>(0,0) = scale;
	M.at<double>(1,1) = scale;
	cv::warpAffine(image_uchar_orig, image_uchar, M,
						 cv::Size(DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT),
						 CV_INTER_CUBIC,
						 cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
	image_uchar_prev = image_uchar;

	Frame frame;
	frame.ori_width = image_uchar_orig.cols;
	frame.ori_height = image_uchar_orig.rows;
	frame.index = 0;
	frame.video_frame_number = 0;
	frame.data_for_wrap = new unsigned char [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3]; //fill after process
	frame.data_for_mat = new float [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3];
	process_and_pad_image(frame.data_for_mat, image_uchar, DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT, 0);

	frame.scale = scale;
	//pad and transform to float
	int offset = 3 * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
	frame.data = new float [BATCH_SIZE * offset];
	int target_width, target_height;
	cv::Mat image_temp;
	//LOG(ERROR) << "frame.index: " << frame.index;
	for(int i=0; i < BATCH_SIZE; i++) {
		float scale = START_SCALE - i*SCALE_GAP;
		target_width = 16 * ceil(NET_RESOLUTION_WIDTH * scale /16);
		target_height = 16 * ceil(NET_RESOLUTION_HEIGHT * scale /16);

		CHECK_LE(target_width, NET_RESOLUTION_WIDTH);
		CHECK_LE(target_height, NET_RESOLUTION_HEIGHT);

		resize(image_uchar, image_temp, cv::Size(target_width, target_height), 0, 0, CV_INTER_AREA);
		process_and_pad_image(frame.data + i * offset, image_temp, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT, 1);
	}





	//process Image Mat
	//int offset = NET_RESOLUTION_WIDTH * NET_RESOLUTION_HEIGHT * 3;

	std::vector< std::vector<double>> subset;
	std::vector< std::vector< std::vector<double> > > connection;

	const boost::shared_ptr<caffe::Blob<float>> heatmap_blob = net_copies[0].person_net->blob_by_name("resized_map");
	const boost::shared_ptr<caffe::Blob<float>> joints_blob = net_copies[0].person_net->blob_by_name("joints");

	//caffe::NmsLayer<float> *nms_layer = (caffe::NmsLayer<float>*)net_copies[0].person_net->layer_by_name("nms").get();

	cudaMemcpy(net_copies[0].canvas, frame.data_for_mat, DISPLAY_RESOLUTION_WIDTH * DISPLAY_RESOLUTION_HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice);

	float* pointer = net_copies[0].person_net->blobs()[0]->mutable_gpu_data();

	cudaMemcpy(pointer + 0 * offset, frame.data, BATCH_SIZE * offset * sizeof(float), cudaMemcpyHostToDevice);

	nms_layer->SetThreshold(global.nms_threshold);
	net_copies[0].person_net->ForwardFrom(0);

	float* heatmap_pointer = heatmap_blob->mutable_cpu_data();
	const float* peaks = joints_blob->mutable_cpu_data();

	float joints[MAX_NUM_PARTS*3*MAX_PEOPLE]; //10*15*3

	int cnt = 0;
	const int num_parts = net_copies[0].nms_num_parts;
	if (net_copies[0].nms_num_parts==18) {
		cnt = connectLimbsCOCO(subset, connection,
											 heatmap_pointer, peaks,
											 net_copies[0].nms_max_peaks, joints, net_copies[0].up_model_descriptor.get());
	}

	net_copies[0].num_people[0] = cnt;
	//VLOG(2) << "num_people[i] = " << cnt;

	cudaMemcpy(net_copies[0].joints, joints,
		MAX_NUM_PARTS*3*MAX_PEOPLE * sizeof(float),
		cudaMemcpyHostToDevice);

	if (subset.size() != 0) {
		render(0, heatmap_pointer); //only support batch size = 1!!!!

		frame.numPeople = net_copies[0].num_people[0];
		frame.gpu_computed_time = get_wall_time();
		frame.joints = boost::shared_ptr<float[]>(new float[frame.numPeople*MAX_NUM_PARTS*3]);
		for (int ij=0;ij<frame.numPeople*num_parts*3;ij++) {
			frame.joints[ij] = joints[ij];
		}
		cudaMemcpy(frame.data_for_mat, net_copies[0].canvas, DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	}
	else {
		render(0, heatmap_pointer);

		frame.numPeople = 0;
		frame.gpu_computed_time = get_wall_time();
		cudaMemcpy(frame.data_for_mat, net_copies[0].canvas, DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	}


	//generate json string
	scale = 1.0/frame.scale;
	int new_num_parts = net_copies.at(0).up_model_descriptor->get_number_parts();

	std::string res_json = "";

	res_json += "{\n";
	res_json +=  "\"version\":0.1,\n";
	res_json +=  "\"bodies\":[\n";
	for (int ip=0;ip<frame.numPeople;ip++) {
		res_json +=  "{\n";
		res_json +=  "\"joints\":";
		res_json +=  "[";
		for (int ij=0;ij<new_num_parts;ij++) {
			res_json += std::to_string(scale*frame.joints[ip*new_num_parts*3 + ij*3+0]);
			res_json += ",";
			res_json += std::to_string(scale*frame.joints[ip*new_num_parts*3 + ij*3+1]);
			res_json += ",";
			res_json += std::to_string(frame.joints[ip*new_num_parts*3 + ij*3+2]);
			if (ij<new_num_parts-1) res_json += ",";
		}
		res_json += "]\n";
		res_json += "}";
		if (ip<frame.numPeople-1) {
			res_json += ",\n";
		}
	}
	res_json += "]\n";
	res_json += "}\n";

	return res_json;

}




