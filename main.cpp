/****************************************************************************\
* Vorlage fuer das Praktikum "Graphische Datenverarbeitung" WS 2018/19
* FB 03 der Hochschule Niedderrhein
* Regina Pohle-Froehlich
*
* Der Code basiert auf den c++-Beispielen der Bibliothek royale
\****************************************************************************/

#include <royale.hpp>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>

#define ENTER 13
#define Evaluation 1
#define RECORDVIDEO 2
#define PLAYVIDEO 3

// defines for sorting vectors of stats
#define AREA 4
#define HORIZONTAL_SORT 5
#define VERTICAL_SORT 6


using namespace cv;
using namespace std;

class Stats
{
public:
	Stats(int x, int y, int area, int width, int height, Scalar colour)
	{
		X = x;
		Y = y;
		Area = area;
		Width = width;
		Height = height;
		Colour = colour;
	}

	Stats()
	{
	}

	int X;
	int Y;
	double Area;
	int Width;
	int Height;
	Scalar Colour;
};

struct statsSortProperty {
	int property;

	statsSortProperty(int property) {
		this->property = property;
	}

	bool operator()(const Stats &s1, const Stats &s2) const {
		if (property == AREA) {
			return s1.Area < s2.Area;
		}
		else if (property == HORIZONTAL_SORT) {
			return s1.X < s2.X;
		}
		else {
			return s1.Y < s2.Y;
		}
	}
};

class MyListener : public royale::IDepthDataListener
{

public:
	// lab02
	MyListener() {
		this->frameCounter = 0;
		this->segmentationColourSet = false;
	}


	void onNewData(const royale::DepthData *data)
	{
		// this callback function will be called for every new depth frame

		std::lock_guard<std::mutex> lock(flagMutex);
		zImage.create(cv::Size(data->width, data->height), CV_32FC1);
		grayImage.create(cv::Size(data->width, data->height), CV_32FC1);
		zImage = 0;
		grayImage = 0;
		int k = 0;
		for (int y = 0; y < zImage.rows; y++)
		{
			for (int x = 0; x < zImage.cols; x++)
			{
				auto curPoint = data->points.at(k);
				if (curPoint.depthConfidence > 0)
				{
					// if the point is valid
					zImage.at<float>(y, x) = curPoint.z;
					grayImage.at<float>(y, x) = curPoint.grayValue;
				}
				k++;
			}
		}

		cv::Mat temp = zImage.clone();
		undistort(temp, zImage, cameraMatrix, distortionCoefficients);
		temp = grayImage.clone();
		undistort(temp, grayImage, cameraMatrix, distortionCoefficients);

		showImage();

		// assignment 5
		if (mode == RECORDVIDEO) {
			if (zVideo.isOpened())
				zVideo << zImage;		// assingment 5: read zImage into zVideo
			if (grayVideo.isOpened())
				grayVideo << grayImage;	// assingment 5: read grayImage into grayVideo
		}

	}

	// L2A1
	void createAverageImage() {
		if (frameCounter == 0) {
			accGrayImage = Mat::zeros(grayImage.size(), CV_64F);
		}

		frameCounter = (++frameCounter) % 20;	// ringbuffer, ranging from 0-19
												// accumulateWeighted(grayImage, accGrayImage, 0.5);
		accumulate(grayImage, accGrayImage);

		if (frameCounter == 0) {
			accGrayImage.convertTo(accFrameGrayImage, CV_8U, 1.0 / 20);

			imshow("20 Frames", accFrameGrayImage);
		}
	}

	void setLensParameters(const royale::LensParameters &lensParameters)
	{
		// Construct the camera matrix
		// (fx   0    cx)
		// (0    fy   cy)
		// (0    0    1 )
		cameraMatrix = (cv::Mat1d(3, 3) << lensParameters.focalLength.first, 0, lensParameters.principalPoint.first,
			0, lensParameters.focalLength.second, lensParameters.principalPoint.second,
			0, 0, 1);

		// Construct the distortion coefficients
		// k1 k2 p1 p2 k3
		distortionCoefficients = (cv::Mat1d(1, 5) << lensParameters.distortionRadial[0],
			lensParameters.distortionRadial[1],
			lensParameters.distortionTangential.first,
			lensParameters.distortionTangential.second,
			lensParameters.distortionRadial[2]);
	}

	// old and ugly
#if 0
	// manual hisogram equalisation (assignment 1)
	void spreadHistogram(Mat pic) {
		Mat zeroFreeMask = Mat::zeros(pic.size(), pic.type());	// initialise empty mask
		double min = 0.0;
		double max = 0.0;

		if (!pic.empty()) {
			// compare depth image against zeroscale to remove zero values (CMP_GT)
			compare(pic, Scalar(0, 0, 0, 0), zeroFreeMask, CMP_GT);

			minMaxLoc(pic, &min, &max, NULL, NULL, zeroFreeMask);

			// lineare scaling using min and max calculated via minMaxLoc
			for (int i = 0; i < pic.rows; i++)
			{
				for (int j = 0; j < pic.cols; j++)
				{
					// exclude pixel value 0
					if (pic.at<float>(i, j) > 0) {
						pic.at<float>(i, j) = (pic.at<float>(i, j) - min) * (255 / (max - min)); // formular from the lecture XD
					}
				}
			}
		}
		else
			perror("Hisogramequalisation failed!");
	}
#endif

	void spreadHistogram(Mat *pic) {
		Mat nonZeroMask = pic->clone();
		compare(*pic, 0, nonZeroMask, CV_CMP_NE);

		double min, max, scale, shift;
		minMaxLoc(*pic, &min, &max, NULL, NULL, nonZeroMask);
		scale = 255.0 / (max - min),
			shift = -255.0 * min / (max - min);
		convertScaleAbs(*pic, *pic, scale, shift);
	}

	// assignment 1
	void showImage() {
		if (!zImage.empty()) {
			spreadHistogram(&zImage);
			zImage.convertTo(zImage, CV_8U);
			applyColorMap(zImage, zImage, COLORMAP_RAINBOW);

			imshow("zMeow", zImage);
		}
		else {
			perror("Displaying zImage failed!");
		}

		if (!grayImage.empty()) {
			spreadHistogram(&grayImage);
			grayImage.convertTo(grayImage, CV_8U);

			// L2A1
			createAverageImage();

			imshow("grayMeow", grayImage);

			// filter tests
			blur(grayImage, avgGrayImage, Size(3, 3));
			imshow("Average", avgGrayImage);

			Mat otsu;


			medianBlur(grayImage, medianGrayImage, 3);
			imshow("Median", medianGrayImage);

			threshold(avgGrayImage, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			imshow("Otsu", otsu);

			segment(otsu);

			// why not bilateral
			bilateralFilter(grayImage, bilateralGrayImage, 9, 75, 75, BORDER_CONSTANT);
			imshow("Bilaterial", bilateralGrayImage);



			//			if (frameCounter == 0) {
			//				imshow("grayMeow", grayImage);
			//			}
		}
		else {
			perror("Displaying grayImage failed!");
		}
		waitKey(1);
	}

	// assingment 5
	void openVideoWriter(Size size, string zName, string grayName, double fps) {
		zVideo.open(zName, CV_FOURCC('M', 'J', 'P', 'G'), fps, size, true);
		grayVideo.open(grayName, CV_FOURCC('M', 'J', 'P', 'G'), fps, size, false);
	}

	void closeVideoWriter() {
		if (zVideo.isOpened()) {
			zVideo.release();
		}
		if (grayVideo.isOpened()) {
			grayVideo.release();
		}

	}

	// assingment 3 / 4 /5
	void videoHandler(string prefix, Size size, uint16_t framerate, bool streamCapture) {
		cout << "Maximum Imagesize is: " << size << endl;

		string zName = prefix + "_depth.avi";
		string grayName = prefix + "_gray.avi";

		openVideoWriter(size, zName, grayName, framerate);
	}

	// assignment 6
	void openStreamCapture(string *prefix) {
		zStream.open(*prefix + "_depth.avi");
		grayStream.open(*prefix + "_gray.avi");
	}

	void closeStreamCapture() {
		if (zStream.isOpened()) {
			zStream.release();
		}
		if (grayStream.isOpened()) {
			grayStream.release();
		}
	}

	void showCapture(string prefix, double framerate) {
		openStreamCapture(&prefix);
		// assignment 6: read, grab frames, display
		if (zStream.isOpened() && grayStream.isOpened()) {
			namedWindow("deepStream", cv::WINDOW_AUTOSIZE);
			namedWindow("grayStream", cv::WINDOW_AUTOSIZE);

			Mat z;
			Mat gray;
			while (zStream.grab() && grayStream.grab()) {
				zStream.retrieve(z);
				imshow("deepStream", z);

				grayStream.retrieve(gray);
				imshow("grayStream", gray);
				waitKey(framerate * 40);
			}
		}
		else {
			perror("Error Show Capture");
		}
		closeStreamCapture();
	}

	void setMode(int mode) {
		this->mode = mode;
	}

	bool positionDetection(Mat pic) {
		int horizontalMax = 0;
		int verticalMax = 0;

		Scalar avgGray = mean(pic);
		Point start(-1, -1);
		Point end(-1, -1);

		for (int x = 10; x < pic.cols - 10; x++) {
			for (int y = 10; y < pic.rows - 10; y++) {
				if ((double)pic.at<uint8_t>(Point(x, y)) <= avgGray[0]) {
					if (start == Point(-1, -1)) {
						start = Point(x, y);
					}
				}
				else {
					if (start != Point(-1, -1)) {
						end = Point(x, y - 1);
					}

					if (start != Point(-1, -1) && end != Point(-1, -1)) {
						if (end.y - start.y > verticalMax) {
							verticalMax = end.y - start.y;
						}
						start = Point(-1, -1);
						end = Point(-1, -1);
					}

				}

				if (start != Point(-1, -1) && y == pic.rows) {
					end = Point(x, y);
					if (end.y - start.y > verticalMax) {
						verticalMax = end.y - start.y;
					}
					start = Point(-1, -1);
					end = Point(-1, -1);
				}
			}
		}

		for (int y = 10; y < pic.rows - 10; y++) {
			for (int x = 10; x < pic.cols - 10; x++) {
				if ((double)pic.at<uint8_t>(Point(x, y)) <= avgGray[0]) {
					if (start == Point(-1, -1)) {
						start = Point(x, y);
					}
				}
				else {
					if (start != Point(-1, -1)) {
						end = Point(x, y - 1);
					}

					if (start != Point(-1, -1) && end != Point(-1, -1)) {
						if (end.x - start.x > horizontalMax) {
							horizontalMax = end.x - start.x;
						}
						start = Point(-1, -1);
						end = Point(-1, -1);
					}
				}

				if (start != Point(-1, -1) && x == pic.cols) {
					end = Point(x, y);
					if (end.x - start.x > horizontalMax) {
						horizontalMax = end.x - start.x;
					}
					start = Point(-1, -1);
					end = Point(-1, -1);
				}
			}
		}
		return horizontalMax > verticalMax;
	}

	void linePlot(Mat pic) {
		if (pic.empty()) {
			cerr << "Image is empty, can't plot anything!" << endl;
		}
		else {
			Mat lineImg = pic.clone();
			lineImg = 0;
			cvtColor(lineImg, lineImg, CV_GRAY2BGR);

			Point start(-1, -1);
			Point end(-1, -1);
			int lineThickness = 2;

			vector<Point>plotValues;

			double min, max;
			minMaxLoc(pic, &min, &max);
			srand(time(NULL));

			// ensures that image has undergone histogram equalisation
			if (max < 255) {
				spreadHistogram(&pic);
			}

			// checks image direction
			bool horizontal = positionDetection(pic);

			if (horizontal) {
				for (int x = 0; x < pic.cols; x++) {
					plotValues.push_back(Point(x, pic.at<uint8_t>(Point(x, pic.rows / 2))));
				}
			}
			else {
				for (int y = 0; y < pic.rows; y++) {
					plotValues.push_back(Point(pic.at<uint8_t>(Point(pic.cols / 2, y)), y));
				}
			}

			for (int x = 0; x < plotValues.size() - 1; x++) {
				start = plotValues[x];
				end = plotValues[x + 1];
				line(lineImg, start, end, Scalar(rand() % 256, rand() % 256, rand() % 256), lineThickness, 8);
			}

			imshow("LinePlot", lineImg);
		}
	}

	void segment(Mat pic) {
		Mat segments, centroids, stats;
		vector<Stats> labelStats;


		int labels = connectedComponentsWithStats(pic, segments, stats, centroids, CV_32S);

		for (int x = 0; x < labels; x++) {
			Stats stat;

			stat.Area = stats.at<int>(x, CC_STAT_AREA);
			stat.X = stats.at<int>(x, CC_STAT_LEFT);
			stat.Y = stats.at<int>(x, CC_STAT_TOP);
			stat.Width = stats.at<int>(x, CC_STAT_WIDTH);
			stat.Height = stats.at<int>(x, CC_STAT_HEIGHT);
			stat.Colour = Scalar(0, 0, 0);	// default colour black to detect errors

			labelStats.push_back(stat);
		}

		// sort(labelStats.begin(), labelStats.end());
		sort(labelStats.begin(), labelStats.end(), statsSortProperty(AREA)); // sort by AREA property


		int avgSize = 1200;// labelStats[labelStats.size() / 2].Area;
		srand(time(NULL));

		// old version does not sort by x or y
#if 0
		vector<Stats> TastenKIEZ;
		for (auto stat : labelStats) {
			if (abs(stat.Area - avgSize) < 200)
				stat.Colour = Scalar(rand() % 256, rand() % 256, rand() % 256);	// random, but consistent colours
				TastenKIEZ.push_back(stat);
		}
#endif
		// hit and miss version, attempts to sortt by x and y coordinates, depending on how the paper is placed
		vector<Stats> TastenKIEZ;
		for (auto stat : labelStats) {
			if (abs(stat.Area - avgSize) < 200)
				TastenKIEZ.push_back(stat);

			if (!TastenKIEZ.empty()) {
				if (TastenKIEZ[0].Width < TastenKIEZ[0].Height) {
					sort(TastenKIEZ.begin(), TastenKIEZ.end(), statsSortProperty(VERTICAL_SORT));
				}
				else {
					sort(TastenKIEZ.begin(), TastenKIEZ.end(), statsSortProperty(HORIZONTAL_SORT));
				}

				if (!segmentationColourSet) {
					for (int x = 0; x < TastenKIEZ.size(); x++) {
						segmentationColours.push_back(Scalar(rand() % 256, rand() % 256, rand() % 256));
						segmentationColourSet = true;
					}
				}
				
				for (int x = 0; x < TastenKIEZ.size(); x++) {
					TastenKIEZ[x].Colour = segmentationColours[x]; // random, but consistent colours
				}
			}
		}


		Mat image = grayImage.clone();
		cvtColor(image, image, CV_GRAY2BGR);

		for (auto key : TastenKIEZ) {
			rectangle(image, Rect(key.X, key.Y, key.Width, key.Height), key.Colour, CV_FILLED);
		}

		imshow("SegmentedImage", image);
		waitKey(1);
	}

private:

	cv::Mat zImage, grayImage;
	cv::Mat cameraMatrix, distortionCoefficients;
	std::mutex flagMutex;
	// assignment 4 / 5
	VideoWriter zVideo;
	VideoWriter grayVideo;

	// assignment 6
	VideoCapture zStream;
	VideoCapture grayStream;

	// parameter for program
	// 1: Evaluation
	// 2: Record Video
	// 3: Play Video
	int mode;

	// L2A1
	int frameCounter;			// number of frames used to accumulate image, when 0 display image
	cv::Mat accGrayImage, accFrameGrayImage, avgGrayImage, medianGrayImage, bilateralGrayImage;

	// L2Segmentation
	bool segmentationColourSet;			// ugly solution for static segmentaton colours using a flag
	vector<Scalar> segmentationColours;	// contains segmentation colours sorted by keys
};

int main(int argc, char *argv[])
{
	MyListener listener;
	string prefix;

	// this represents the main camera device object
	std::unique_ptr<royale::ICameraDevice> cameraDevice;

	// the camera manager will query for a connected camera
	{
		royale::CameraManager manager;

		// try to open the first connected camera
		royale::Vector<royale::String> camlist(manager.getConnectedCameraList());
		std::cout << "Detected " << camlist.size() << " camera(s)." << std::endl;

		if (!camlist.empty())
		{
			cameraDevice = manager.createCamera(camlist[0]);
		}
		else
		{
			std::cerr << "No suitable camera device detected." << std::endl
				<< "Please make sure that a supported camera is plugged in, all drivers are "
				<< "installed, and you have proper USB permission" << std::endl;
			return 1;
		}

		camlist.clear();

	}
	// the camera device is now available and CameraManager can be deallocated here

	if (cameraDevice == nullptr)
	{
		// no cameraDevice available
		if (argc > 1)
		{
			std::cerr << "Could not open " << argv[1] << std::endl;
			return 1;
		}
		else
		{
			std::cerr << "Cannot create the camera device" << std::endl;
			return 1;
		}
	}

	// call the initialize method before working with the camera device
	auto status = cameraDevice->initialize();
	if (status != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Cannot initialize the camera device, error string : " << getErrorString(status) << std::endl;
		return 1;
	}

	// retrieve the lens parameters from Royale
	royale::LensParameters lensParameters;
	status = cameraDevice->getLensParameters(lensParameters);
	if (status != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Can't read out the lens parameters" << std::endl;
		return 1;
	}


	cameraDevice->setExposureMode(royale::ExposureMode::AUTOMATIC);
	listener.setLensParameters(lensParameters);

	// register a data listener
	if (cameraDevice->registerDataListener(&listener) != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error registering data listener" << std::endl;
		return 1;
	}

	namedWindow("zMeow", CV_WINDOW_AUTOSIZE);
	namedWindow("grayMeow", CV_WINDOW_AUTOSIZE);

	// start capture mode
	if (cameraDevice->startCapture() != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error starting the capturing" << std::endl;
		return 1;
	}

	uint16_t width, height, framerate;
	cameraDevice->getMaxSensorWidth(width);
	cameraDevice->getMaxSensorHeight(height);
	cameraDevice->getMaxFrameRate(framerate);

	Size size = Size(width, height);

	// assingment 3 /4 / 6
	// input via modified properties through argc and argv
	if (argc > 1) {
		// assignment 3
		switch (stoi(argv[1])) {
		case Evaluation: // assignemtn 3: if 1 then print message "data evaluation"
			cout << "Call of evaluation method" << endl;
			listener.setMode(Evaluation);
			break;
		case RECORDVIDEO: // assignment 3 / 4: if 2 then record video, read prefix of video file name or take as parameter
			if (argc == 3) {
				prefix = argv[2];
			}
			else {
				cout << "Please enter a name for the video file" << endl;
				cin >> prefix;
			}
			listener.setMode(RECORDVIDEO);
			listener.videoHandler(prefix, size, framerate, false);
			break;
		case PLAYVIDEO:	// assignment 6: same as assignment 4 and 5 but dispaly video as well 
			if (argc == 3) {
				prefix = argv[2];
			}
			else {
				cout << "Please enter a name for the video file" << endl;
				cin >> prefix;
			}
			listener.showCapture(prefix, framerate);
			break;
		default:
			cout << "DEBUG: No additional parameters passed" << endl;
		}
	}

	while (waitKey(0) != ENTER) {
		// nothing happens here because we are waiting for enter to be pressed
	}

	// stop capture mode
	if (cameraDevice->stopCapture() != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error stopping the capturing" << std::endl;
		return 1;
	}

	return 0;
}

