/*
 * countCarsFinal.cpp
 *
 *  Created on: 5 mai 2018
 *      Author: tux
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define PI 3.1415926

using namespace std;
using namespace cv;

void Morpho(Mat&);
const string Date();
const string Heure();
void matrixTransformation(Mat&, Mat&);

int main()
{
	static int sum_of_elems;

	Mat frame, destination;
	Mat mask, threshold_output;

	Point pt1, pt2;
	Rect zoneRoi;

	vector<int>::iterator it;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Ptr<BackgroundSubtractorMOG2> mog;
	mog = createBackgroundSubtractorMOG2();

	string video = "cars.mp4";
	VideoCapture cap(video);

	while(true)
	{
		cap >> frame;

		matrixTransformation(frame, destination);

		pt1 = Point(100, destination.rows/2);
		pt2 = Point(destination.cols-50, destination.rows/2+25);
		zoneRoi = Rect(pt1, pt2);
		rectangle(destination, zoneRoi, Scalar(255,0,0), 1);

		mog->apply(destination, mask);
		blur(mask, mask, Size(5,5));
		Morpho(mask);

		Mat src_copy = frame.clone();

		threshold(mask, threshold_output, 50, 255, CV_ADAPTIVE_THRESH_MEAN_C);

		findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

		vector<vector<Point> >hull(contours.size());
		vector<vector<Point> > contours_poly( contours.size());
		vector<Rect> boundRect( contours.size());
		vector<Point2f>center( contours.size());
		vector<float>radius( contours.size());
		vector<int> nbVehicule;

		Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

		for(size_t i = 0; i < contours.size(); i++)
		{
			convexHull(Mat(contours[i]), hull[i], false);
			drawContours(drawing, hull, i, Scalar(255,255,255), -1, 8, vector<Vec4i>(), -1, Point());

			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);

			//rectangle(destination, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 1, 8, 0 );
			//circle(destination, center[i], 2, Scalar(0,0,255), 2, 8, 0 );

			if(center[i].inside(zoneRoi) != 0)
			{
				nbVehicule.push_back(center[i].inside(zoneRoi));
				rectangle(destination, zoneRoi, Scalar(255,255,255), 2);
			}
		}

		for(it = nbVehicule.begin(); it != nbVehicule.end(); ++it)
		{
			sum_of_elems += *it;
		}

		stringstream a;
		a << sum_of_elems;

		putText(destination, "Count: ", Point(destination.cols/2-80, 50), 1, 1, Scalar(255,255,255), 1);
		putText(destination, a.str(), Point(destination.cols/2, 50), 1, 1, Scalar(255,255,255), 2);
		putText(destination, Date(),Point(3,20),1,1,Scalar(255,255,255),1);
		putText(destination, Heure(),Point(3,40),1,1,Scalar(255,255,255),1);

		//imshow("Frame", frame);
		imshow("Projection", destination);
		//imshow("Mask", mask);
		//imshow("ConvexHull", drawing);
		waitKey(32);
	}
	destroyAllWindows();
	return 0;
}

void Morpho(Mat &a)
{
	Mat erodeElement = getStructuringElement( MORPH_RECT,Size(5,5));
	Mat dilateElement = getStructuringElement( MORPH_RECT,Size(15,15));

	erode(a, a, erodeElement);
	erode(a, a, erodeElement);
	dilate(a, a,dilateElement);
	dilate(a, a,dilateElement);

	blur(a, a, Size(50,50));

	threshold(a, a, 180, 255, CV_THRESH_BINARY);
}

const string Date()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);

    strftime(buf, sizeof(buf), "%d/%m/%Y", &tstruct);

    return buf;
}

const string Heure()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);

    strftime(buf, sizeof(buf), "%X", &tstruct);

    return buf;
}

void matrixTransformation(Mat &a, Mat &b)
{
	int H = 640;
	int W = 480;

	resize(a, a,Size(H,W));

	int alpha_ = 30;

	double alpha;

	alpha =((double)alpha_ -90) * PI/180;

	Size image_size = a.size();
	double w = (double)image_size.width;
	double h = (double)image_size.height;

	Mat A1 = (Mat_<float>(4, 3)<<
		1, 0, -w/2,
		0, 1, -h/2,
		0, 0, 0,
		0, 0, 1 );

	Mat RX = (Mat_<float>(4, 4) <<
		1, 0, 0, 0,
		0, cos(alpha), -sin(alpha), 0,
		0, sin(alpha), cos(alpha), 0,
		0, 0, 0, 1 );

	Mat T = (Mat_<float>(4, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 500,
		0, 0, 0, 1);

	Mat K = (Mat_<float>(3, 4) <<
		500, 0, w/2, 0,
		0, 500, h/2, 0,
		0, 0, 1, 0);

	Mat transformationMat = K * (T * (RX * A1));

	warpPerspective(a, b, transformationMat, image_size, INTER_CUBIC | WARP_INVERSE_MAP);
}
