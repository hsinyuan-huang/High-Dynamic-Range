#include <stdio.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Debevec Weighting
float weight(int bit8)
{
	if(bit8 < 128) return bit8 + 1.f;
	else return 256.f - bit8;
}

bool OUTFASHION = 0; // TODO

float THRESHOLD;
int SAMPLE_N = 70;
float var_x[5500][5500];
float est_x[5500][5500];

// Disjoint Set
int fa[5500 * 5500], mem_cnt[5500 * 5500];
int disc_fa[5500 * 5500];

int find(int x)
{
	return fa[x] = (fa[x] == x? x: find(fa[x]));
}
void unio(int a, int b)
{
	if(find(a) != find(b))
		mem_cnt[find(b)] += mem_cnt[find(a)];
	fa[find(a)] = find(b);
}

// Traverse check
bool inside(int r, int c, int rows, int cols)
{
	return r >= 0 && r < rows && c >= 0 && c < cols;
}

int main(int argc, char** argv )
{
	srand(1023);
	
	if(argc != 4)
	{
		printf("usage: ./Debevec_HDR #Pics threshold <Image_Folder>\n");
		return -1;
	}

	int num_pics = atoi(argv[1]);
	THRESHOLD = atof(argv[2]);
	string img_folder(argv[3]);

	if(img_folder.back() != '/')
		img_folder.push_back('/');

	string img_type = ".png";
	if(!imread(img_folder + "img01.png").data)
		img_type = ".jpg";

	vector<Mat> image; // From 0 ~ #Pics-1
	for(int i = 0; i < num_pics; i++)
	{
		char tmp[10];
		sprintf(tmp, "%02d", i+1);
		string number(tmp);

		string img_Name = img_folder + "img" + number + img_type;
		image.push_back(imread(img_Name, 1));

		// Print the file name
		cout << img_Name << endl;

		if(!image[i].data)
		{
			printf("No image data\n");
			return -1;
		}
	}

	vector<float> times;
	FILE* timedata = fopen((img_folder + "time.data").c_str(), "r");
	if(timedata == NULL)
	{
		printf("No time metadata\n");
		return -1;
	}
	for(int i = 0; i < num_pics; i++)
	{
		char buf[50];
		fscanf(timedata, "%s", buf);
		if(buf[1] == '/')
		{
			times.push_back(1.0 / atof(buf + 2));
			printf("time: %f\n", 1.0 / atof(buf + 2));
		}
		else
		{
			times.push_back(atof(buf));
			printf("time: %f\n", atof(buf));
		}
	}

	int rows = image[0].rows;
	int cols = image[0].cols;

	// Generate Samples (Inspired by OpenCV documentation)
	int sample_x[100];
	int sample_y[100];

	int col_num = (int)sqrt(1.f * SAMPLE_N * cols / rows);
	int row_num = SAMPLE_N / col_num;

	SAMPLE_N = 0;
	int col = (cols / col_num) / 2;
	for(int i = 0; i < col_num; i++){
		int row = (rows / row_num) / 2;
		for(int j = 0; j < row_num; j++){
			if(inside(row, col, rows, cols)){
				sample_x[SAMPLE_N] = col;
				sample_y[SAMPLE_N] = row;
				SAMPLE_N ++;
			}
			row += (rows / row_num);
		}
		col += (cols / col_num);
	}

	// SVD for Finding Response Function
	float Im[3][256];
	for(int cBGR = 0; cBGR < 3; cBGR ++)
	{
		Mat A = Mat::zeros(num_pics * SAMPLE_N + 1 + 254, 256 + SAMPLE_N, CV_32F);
		Mat B = Mat::zeros(num_pics * SAMPLE_N + 1 + 254, 1, CV_32F);

		int rowcnt = 0;

		for(int s = 0; s < SAMPLE_N; s++)
		{
			for(int t = 0; t < num_pics; t++)
			{
				int bit8 = image[t].at<Vec3b>(sample_y[s], sample_x[s])[cBGR];
				float w = weight(bit8);

				A.at<float>(rowcnt, bit8) = w;
				A.at<float>(rowcnt, 256 + s) = -w;
				B.at<float>(rowcnt, 0) = w * log(times[t]);
				rowcnt++;
			}
		}

		A.at<float>(rowcnt, 128) = 1;
		rowcnt++;

		for(int i = 1; i <= 254; i++)
		{
			float lambda = 10; // Using OpenCV default
			A.at<float>(rowcnt,   i) = -2 * lambda * weight(i);
			A.at<float>(rowcnt, i-1) = lambda * weight(i);
			A.at<float>(rowcnt, i+1) = lambda * weight(i);
			rowcnt++;
		}	
	
		Mat x_star;
		solve(A, B, x_star, DECOMP_SVD); // Pseudo Inverse

		for(int i = 0; i < 256; i++)
			Im[cBGR][i] = exp(x_star.at<float>(i));
	}
	
	// Calculate Variance for Irradiance
	for(int cBGR = 0; cBGR < 3; cBGR ++)
	{
		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < cols; c++)
			{
				float wsum = 0, Esum = 0, E2sum = 0;

				for(int t = 0; t < num_pics; t++)
				{
					int bit8 = image[t].at<Vec3b>(r, c)(cBGR);

					float E = Im[cBGR][bit8] / times[t];
					Esum += weight(bit8) * E;
					E2sum += weight(bit8) * E * E;
					wsum += weight(bit8);
				}

				float cur_var = (E2sum / wsum) / (Esum * Esum / wsum / wsum) - 1;
				var_x[r][c] = max(var_x[r][c], cur_var);
			}
		}
	}

	// Init Disjoint Set
	for(int r = 0; r < rows; r++)
		for(int c = 0; c < cols; c++)
		{
			fa[r * cols + c] = r * cols + c;
			mem_cnt[r * cols + c] = 1;
		}

	// Create Disjoint Sets of High-Variance Region
	for(int r = 0; r < rows; r++)
	{
		for(int c = 0; c < cols; c++)
		{
			int id = r * cols + c;
			if(var_x[r][c] > THRESHOLD)
			{
				for(int dx = -1; dx <= 1; dx ++) for(int dy = -1; dy <= 1; dy ++)
				{
					if(inside(r + dx, c + dy, rows, cols))
						unio(id, id + dx * cols + dy);
				}
			}
		}
	}

	// Discretize the Disjoint Sets of High-Var Region
	vector<int> best_expo;
	vector<float> best_expo_val;
	vector<Vec3b> seg_color;

	int disc_cnt = 1;
	for(int r = 0; r < rows; r++)
	{
		for(int c = 0; c < cols; c++)
		{
			int id = r * cols + c;
			if(mem_cnt[find(id)] > rows * cols * 0.001)
			{
				if(disc_fa[find(id)] == 0)
				{
					disc_fa[find(id)] = disc_cnt ++;

					best_expo.push_back(-1);
					best_expo_val.push_back(0);
					seg_color.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
				}
				disc_fa[id] = disc_fa[find(id)];
			}
		}
	}
	printf("Moving Segments: %d\n", disc_cnt - 1);

	// Find the best exposure time for each Disjoint Sets
	for(int t = 0; t < num_pics; t++)
	{
		vector<float> expo_val;
		for(int i = 0; i < disc_cnt - 1; i++)
			expo_val.push_back(0);

		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < cols; c++)
			{
				int id = r * cols + c;
				if(disc_fa[find(id)] != 0)
				{
					for(int cBGR = 0; cBGR < 3; cBGR ++)
					{
						int bit8 = image[t].at<Vec3b>(r, c)(cBGR);
						expo_val[disc_fa[find(id)] - 1] += weight(bit8);
					}
				}
			}
		}

		for(int i = 0; i < disc_cnt - 1; i++)
		{
			if(expo_val[i] > best_expo_val[i])
			{
				best_expo[i] = t;
				best_expo_val[i] = expo_val[i];
			}
		}
	}

	for(int i = 0; i < disc_cnt - 1; i++)
		printf("%d (%d %d %d)\n", best_expo[i],
				seg_color[i][0], seg_color[i][1], seg_color[i][2]);

	// Visualize Disjoint Sets of High Var
	Mat var_seg = Mat::zeros(image[0].size(), CV_8UC3);
	for(int r = 0; r < rows; r++)
	{
		for(int c = 0; c < cols; c++)
		{
			int id = r * cols + c;
			if(disc_fa[find(id)] != 0)
				var_seg.at<Vec3b>(r, c) = seg_color[disc_fa[find(id)]-1];
		}
	}
	imwrite("var_Deb.jpg", var_seg);

	// Response Outputs
	FILE* response_out = fopen("response_Deb.txt", "w");
	for(int cBGR = 0; cBGR < 3; cBGR ++)
	{
		for(int i = 0; i < 256; i++)
			fprintf(response_out, "%f\t", Im[cBGR][i]);
		fprintf(response_out, "\n");
	}
	fclose(response_out);

	// Final HDR
	Mat HDR = Mat::zeros(image[0].size(), CV_32FC3);

	if(OUTFASHION)
	{
		for(int cBGR = 0; cBGR < 3; cBGR ++)
		{
			for(int r = 0; r < rows; r++)
			{
				for(int c = 0; c < cols; c++)
				{
					float wsum = 0, logIsum = 0;

					for(int t = 0; t < num_pics; t++)
					{
						int bit8 = image[t].at<Vec3b>(r, c)(cBGR);

						logIsum += weight(bit8) * log(Im[cBGR][bit8] / times[t]);
						wsum += weight(bit8);
					}

					HDR.at<Vec3f>(r, c)(cBGR) = exp(logIsum / wsum);
				}
			}
		}
	}
	else
	{
		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < cols; c++)
			{
				Vec3f altern(0, 0, 0);

				if(disc_fa[find(r * cols + c)] != 0)
				{
					int t = best_expo[disc_fa[find(r * cols + c)] - 1], bit8;

					bit8 = image[t].at<Vec3b>(r, c)(0);
					altern[0] = Im[0][bit8] / times[t];
					bit8 = image[t].at<Vec3b>(r, c)(1);
					altern[1] = Im[1][bit8] / times[t];
					bit8 = image[t].at<Vec3b>(r, c)(2);
					altern[2] = Im[2][bit8] / times[t];
				}

				float wsum = 0;
				Vec3f logIsum(0, 0, 0);

				for(int t = 0; t < num_pics; t++)
				{
					float BGRweight = 0;

					for(int cBGR = 0; cBGR < 3; cBGR ++)
					{
						int bit8 = image[t].at<Vec3b>(r, c)(cBGR);
						BGRweight += weight(bit8) / 3.0;
					}

					for(int cBGR = 0; cBGR < 3; cBGR ++)
					{
						int bit8 = image[t].at<Vec3b>(r, c)(cBGR);
						logIsum[cBGR] += BGRweight * log(Im[cBGR][bit8] / times[t]);
					}
					wsum += BGRweight;
				}

				Vec3f origin = logIsum / wsum;
				for(int cBGR = 0; cBGR < 3; cBGR ++)
					origin[cBGR] = exp(origin[cBGR]);

				// Soft Change should be better
				if(disc_fa[find(r * cols + c)] != 0)
				{
					float alpha = min(var_x[r][c] / THRESHOLD, 1.f);
					HDR.at<Vec3f>(r, c) = alpha * altern + (1 - alpha) * origin;
				}
				else
					HDR.at<Vec3f>(r, c) = origin;
			}
		}
	}

	imwrite("HDR_Deb_image.exr", HDR);
	return 0;
}
