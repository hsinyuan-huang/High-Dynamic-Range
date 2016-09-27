#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#define MAXN 30

using namespace std;
using namespace cv;

int img_num;

void showImage(string window_name, Mat& input) {
    namedWindow(window_name);
    imshow(window_name, input);
}

void imageShrink2(Mat& img, Mat& sml_img) {
    resize(img, sml_img, Size(img.cols / 2, img.rows / 2), 0, 0, INTER_AREA);
}

int bitmapTotal(Mat& img) {
    return countNonZero(img);
}

void calBitmaps(Mat& img, Mat& tb, Mat& eb) {
    Mat tmp = img.reshape(1, 1);
    Mat sorted;
    cv::sort(tmp, sorted, CV_SORT_ASCENDING);
    int med = sorted.at<uchar>(sorted.cols / 2);
    threshold(img, tb, med, 255, THRESH_BINARY);
    eb.create(img.rows, img.cols, CV_8UC1);
    eb = Scalar(255);
    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            int pixel_value = img.at<uchar>(y, x);
            if(med - 4 <= pixel_value && pixel_value <= med + 4) {
                eb.at<uchar>(y, x) = 0;
            }
        }
    }
}

void calShiftMask(Rect& mask, int xs, int ys, int _w, int _h, bool is_base) {
    int w = _w - ((xs > 0) ? xs : -xs);
    int h = _h - ((ys > 0) ? ys : -ys);
    int x = 0, y = 0;
    if((is_base && xs > 0) || (!is_base && xs < 0))
        x = (xs > 0) ? xs : -xs;
    if((is_base && ys > 0) || (!is_base && ys < 0))
        y = (ys > 0) ? ys : -ys;
    mask = Rect(x, y, w, h);
}

void getExpShift(Mat& img1, Mat& img2, int shift_bits, int* shift_ret) {
   	int min_err;
	int cur_shift[2];
	Mat tb1, tb2;
	Mat eb1, eb2;
	int i, j;
	if (shift_bits > 0) {
		Mat sml_img1, sml_img2;
		imageShrink2(img1, sml_img1);
		imageShrink2(img2, sml_img2);
		getExpShift(sml_img1, sml_img2, shift_bits - 1, cur_shift);
		cur_shift[0] *= 2;
		cur_shift[1] *= 2;
	} else {
		cur_shift[0] = 0;
		cur_shift[1] = 0;
	}

	calBitmaps(img1, tb1, eb1);
	calBitmaps(img2, tb2, eb2);
	min_err = img1.rows * img1.cols;

    int err_without_shift = 0;
	for (i = -1 ; i <= 1 ; i++){
		for (j = -1; j <= 1; j++) {
			int xs = cur_shift[0] + i;
			int ys = cur_shift[1] + j;
            Rect shift_mask[2];
            calShiftMask(shift_mask[0], xs, ys, img1.cols, img1.rows, true);
            calShiftMask(shift_mask[1], xs, ys, img2.cols, img2.rows, false);
            Mat diff = tb1(shift_mask[0]) ^ tb2(shift_mask[1]);
            diff = diff & eb1(shift_mask[0]);
            diff = diff & eb2(shift_mask[1]);
			int err = 0;
			err = countNonZero(diff);
            if(i == 0 && j == 0)
                err_without_shift = err;
			if (err < min_err) {
				shift_ret[0] = xs;
				shift_ret[1] = ys;
				min_err = err;
			}
		}
	}
    if(err_without_shift == min_err) {
        shift_ret[0] = cur_shift[0];
        shift_ret[1] = cur_shift[1];
    }
}

void calMaxShiftBits(int delxy[MAXN][2], int* max_shift_bits) {
    /* 0 = x+, 1 = x-, 2 = y+, 3 = y- */
    for(int i = 0; i < img_num; i++) {
        if(delxy[i][0] > max_shift_bits[0]) {
            max_shift_bits[0] = delxy[i][0];    
        }
        if(delxy[i][0] < max_shift_bits[1]) {
            max_shift_bits[1] = delxy[i][0];
        }
        if(delxy[i][1] > max_shift_bits[2]) {
            max_shift_bits[2] = delxy[i][1];
        }
        if(delxy[i][1] < max_shift_bits[3]) {
            max_shift_bits[3] = delxy[i][1];
        }
    }
}

void cropImage(Mat& input, int* delxy, int* max_shift_bits, bool is_base, Mat& output) {
   Rect mask;
   int dx[2], dy[2];
   for(int i = 0; i < 2; i++) {
       dx[i] = max_shift_bits[i];
       dy[i] = max_shift_bits[i + 2];
   }
   if(is_base) {
       mask = Rect(dx[0], dy[0], input.cols - (dx[0] - dx[1]), input.rows - (dy[0] - dy[1]));
   }
   else {
       mask = Rect(dx[0] - delxy[0], dy[0] - delxy[1], input.cols - (dx[0] - dx[1]), input.rows - (dy[0] - dy[1]));
   }
   output = input(mask);
}

int main(int argc, char** argv) {
	if(argc != 3){
		fprintf(stderr, "Usage: ./MTB #_of_pics <folder>\n");
		exit(-1);
	}

    Mat inputs[MAXN];
    Mat inputs_gray[MAXN];
    img_num = atoi(argv[1]);
    int base = img_num / 2;
    
    /* load images. */
    char cwd[100];
    getcwd(cwd, 100);
    char* folder_name = argv[2];
    for(int i = 0; i < img_num; i++) {
        char file_name[100];
        sprintf(file_name, "%s/%s/img%02d.jpg", cwd, folder_name, i + 1);
		fprintf(stderr, "Reading: %s/%s/img%02d.jpg\n", cwd, folder_name, i + 1);
        
		inputs[i] = imread(file_name);
		
        if(!inputs[i].data) {
			printf("No image data\n");
			return -1;
		}

        cvtColor(inputs[i], inputs_gray[i], CV_BGR2GRAY);
    }

    int delxy[MAXN][2] = {0};

    /* apply median threshold method. */
    for(int i = 0; i < img_num; i++) {
        if(i != base)
            getExpShift(inputs_gray[base], inputs_gray[i], 6, delxy[i]);
    }


    /* get the size of mask for cropping images. */
    int max_shift_bits[4] = {0};
    calMaxShiftBits(delxy, max_shift_bits);
    Mat outputs[MAXN];
    

    /* crop images. */
    for(int i = 0; i < img_num; i++) {
		cropImage(inputs[i], delxy[i], max_shift_bits, ((i == base) ? true : false), outputs[i]);
    }
    

    /* export images. */
    sprintf(folder_name, "%s_aligned", folder_name);
    mkdir(folder_name, 0755);
    chmod(folder_name, 0755);
    char folder_path[100];
    sprintf(folder_path, "%s/%s", cwd, folder_name);
    for(int i = 0; i < img_num; i++) {
        char output_name[100];
        sprintf(output_name, "%s/img%02d.jpg", folder_path, i + 1);
        vector<int> params;
        params.push_back(CV_IMWRITE_JPEG_QUALITY);
        params.push_back(95);
        imwrite(output_name, outputs[i], params);
    }
}
