//============================================================================
// Name        : main.cpp
// Author      : 
// Version     :
// Copyright   : None
// Description : A sudoku solver for the Innovation lab
//============================================================================
#include <math.h>
#include <tesseract/baseapi.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <stdio.h>
using namespace cv;


/* a more evolved computer vision example, solves your daily sudoku puzzle for you */


Point2f computeIntersect(Vec2f line1, Vec2f line2);
vector<Point2f> lineToPointPair(Vec2f line);
void rotate(cv::Mat& src, double angle, cv::Mat& dst);

bool acceptLinePair(Vec2f line1, Vec2f line2, float minTheta);


void drawLine(Vec2f line, Mat &img, Scalar rgb = CV_RGB(0,0,255))
{
    if(line[1]!=0)
    {
        float m = -1/tan(line[1]);
        float c = line[0]/sin(line[1]);

        cv::line(img, Point(0, c), Point(img.size().width, m*img.size().width+c), rgb);
    }
    else
    {
        cv::line(img, Point(line[0], 0), Point(line[0], img.size().height), rgb);
    }
}






int main(int argc, char* argv[]) {

	//initialising opencv2
	CvSize size640x480 = cvSize(640, 480);			

	CvCapture* p_capWebcam;						

	Mat p_imgOriginal;			
	Mat p_imgProcessed;			
	Mat p_imgCopy;
	Mat p_imgCopy2;

	char charCheckForEscKey;			// Esc exits program

	p_capWebcam = cvCaptureFromCAM(0);	// 0 => first webcam on device

	if(p_capWebcam == NULL) {			// if capture failure, check webcam
		printf("error: capture is NULL \n");	
		getchar();								
		return(-1);								// kill program
	}

	cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);		// original image from webcam
	cvNamedWindow("Copy", CV_WINDOW_AUTOSIZE);		// will store a copy from webcam
	cvNamedWindow("Processed", CV_WINDOW_AUTOSIZE);		// the processed image we will use for further detection
	cvNamedWindow("Face", CV_WINDOW_AUTOSIZE);		// final
	p_imgProcessed = cvCreateImage(size640x480,			
								   IPL_DEPTH_8U,		// 8-bit color depth
								   1);					// 1 channel (grayscale), if this was a color image, use 3
	bool firstPass = true;
	while(1) {								// for each frame . . .
		p_imgOriginal = cvQueryFrame(p_capWebcam);		// get frame from webcam

		//p_imgProcessed = Mat(p_imgOriginal.size(), CV_8UC1);

		if(firstPass)      //make a copy of the original capture
		{
		cvtColor(p_imgOriginal,p_imgProcessed,CV_RGB2GRAY);
		cvtColor(p_imgOriginal,p_imgCopy,CV_RGB2GRAY);
		firstPass = false;
		}
		charCheckForEscKey = cvWaitKey(10);

		// delay (in ms), and get key press, if any

		if(charCheckForEscKey == 32)
		{
		  //p_imgCopy = p_imgOriginal;
		  p_imgCopy2 = p_imgOriginal;


		  cvtColor(p_imgOriginal,p_imgProcessed,CV_RGB2GRAY);  //we convert to greylevels, colors are unecessary
		  GaussianBlur(p_imgProcessed, p_imgProcessed, Size(11,11), 0); //blurring allows us to ignore unecessary details and removes artifacts.
		  adaptiveThreshold(p_imgProcessed, p_imgProcessed, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2); //scales colors for our darkest color to be (255,255,255), aka pure black
		  cv::bitwise_not(p_imgProcessed, p_imgProcessed); //inverse colors
		  Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
		  dilate(p_imgProcessed, p_imgProcessed, kernel); //less unecessary  will produce a more accurate result 

	    

		  cvtColor(p_imgOriginal,p_imgCopy,CV_RGB2GRAY);
		  GaussianBlur(p_imgCopy, p_imgCopy, Size(11,11), 0);
				adaptiveThreshold(p_imgCopy, p_imgCopy, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
		   cv::bitwise_not(p_imgCopy, p_imgCopy);
		  //Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
		  dilate(p_imgCopy, p_imgCopy, kernel);

		  //


		  //we try to find the biggest blob of colour (aka, all pixels of nearly the same color that are connected.
		  //In the case of a sudoku, the blob will give us the grid.
		     int max=-1;
		     Point maxPt;

		     for(int y=0;y<p_imgProcessed.size().height;y++)
		     {
		         uchar *row = p_imgProcessed.ptr(y);
		         for(int x=0;x<p_imgProcessed.size().width;x++)
		         {
		             if(row[x]>=128)
		             {
		                  int area = floodFill(p_imgProcessed, Point(x,y), CV_RGB(0,0,64));
		                  	  	  	  floodFill(p_imgCopy, maxPt, CV_RGB(0,0,64));
		                  if(area>max)
		                  {
		                      maxPt = Point(x,y);
		                      max = area;
		                  }
		             }
		         }
		     }

		     floodFill(p_imgProcessed, maxPt, CV_RGB(255,255,255)); // we only need the grid to detected where the relation between numbers.
		     floodFill(p_imgCopy, maxPt, CV_RGB(0,0,0));

		     for(int y=0;y<p_imgProcessed.size().height;y++)
		       {
		           uchar *row = p_imgProcessed.ptr(y);
		           for(int x=0;x<p_imgProcessed.size().width;x++)
		           {
		               if(row[x]==64 && x!=maxPt.x && y!=maxPt.y)
		               {
		            	  floodFill(p_imgProcessed, Point(x,y), CV_RGB(0,0,0));
		               }
		           }
		       }


		     erode(p_imgProcessed, p_imgProcessed, kernel); //blur our resulting blob/grid


		     vector<Vec2f> lines;
		      HoughLines(p_imgProcessed, lines, 1, CV_PI/180, 150, 0, 0 );

		    for(unsigned int i=0;i<lines.size();i++)
		      {
			  drawLine(lines[i],  p_imgProcessed, CV_RGB(0,0,128));
		      }

		    std::cout << "Detected " << lines.size() << " lines." << std::endl;

		      // compute the intersection from the lines detected...
		      vector<Point2f> intersections;
		      for( size_t i = 0; i < lines.size(); i++ )
		      {
			  for(size_t j = 0; j < lines.size(); j++)
			  {
			      Vec2f line1 = lines[i];
			      Vec2f line2 = lines[j];
			      if(acceptLinePair(line1, line2, CV_PI / 32))
			      {
				  Point2f intersection = computeIntersect(line1, line2);
				  intersections.push_back(intersection);
			      }
			  }

		      }
		      int sumX = 0;
		      int sumY = 0;
		    vector<Point2f> angles;
		      if(intersections.size() > 0)
		      {
			  int k = 0;
			  vector<Point2f>::iterator i;
			  vector<int> dist(lines.size(),0);

			  for(i = intersections.begin(); i != intersections.end(); ++i)
			  {
			      std::cout << "Intersection is " << i->x << ", " << i->y << std::endl;
			      circle(p_imgProcessed, *i, 1, Scalar(0, 255, 0), 3);
			      sumX += i->x;
			      sumY += i->y;

			      k++;

			  }
			  if (k != 0)
			  {
			  sumX = sumX/k;
			  sumY = sumY/k;

			circle(p_imgProcessed,cvPoint(sumX,sumY),5,CV_RGB(55,55,55),-1);
			k= 0;
			for(i = intersections.begin(); i != intersections.end(); ++i)
			  {
		  //      	dist.at(k) = sqrt((sumX-i->x)*(sumX-i->x)+(sumY-i->y)*(sumY-i->y));
				dist.insert(dist.begin()+k,  sqrt((sumX-i->x)*(sumX-i->x)+(sumY-i->y)*(sumY-i->y)) );
				k++;
			  }
			k = 0;
			Point2f topLeft = Point2f(0,0);
			Point2f topRight = Point2f(0,0);
			Point2f bottomLeft = Point2f(0,0);
			Point2f bottomRight = Point2f(0,0);
			int anglesDistance[4]  = {0};
			for(i = intersections.begin(); i != intersections.end(); ++i)
			  {
				float temp = sqrt((sumX-i->x)*(sumX-i->x)+(sumY-i->y)*(sumY-i->y)) ;

				if( (i->x - sumX) < 0 && (i->y -sumY)< 0)
				{
					if ( temp > anglesDistance[0]  )
					{
						anglesDistance[0] = temp;
						topLeft = Point2f(i->x, i->y);
					}
				}
				else if( (i->x - sumX) > 0 && (i->y -sumY)< 0)
				{
					if ( temp > anglesDistance[1]  )
					{
						anglesDistance[1] = temp;
						topRight = Point2f(i->x, i->y);
					}
				}
				else if( (i->x - sumX) < 0 && (i->y -sumY)> 0)
				{
					if ( temp > anglesDistance[2]  )
					{
						anglesDistance[2] = temp;
						bottomLeft = Point2f(i->x, i->y);
					}
				}
				else if( (i->x - sumX) > 0 && (i->y -sumY)> 0)
				{
					if ( temp > anglesDistance[3]  )
					{
						anglesDistance[3] = temp;
						bottomRight = Point2f(i->x, i->y);
					}
				}
		  //      	dist.at(k) = sqrt((sumX-i->x)*(sumX-i->x)+(sumY-i->y)*(sumY-i->y));
				k++;
			  }

		      Mat M = (Mat_<double>(2,3) << 1, 0, p_imgCopy.cols/2-sumX, 0, 1, p_imgCopy.rows/2-sumY);

			  warpAffine(p_imgCopy,p_imgCopy,M, cv::Size(p_imgCopy.cols, p_imgCopy.rows));



		      float angle1 =  atan2(topLeft.y - topRight.y, topLeft.x - topRight.x);
		      float angle2 =  atan2(topLeft.y - topRight.y, topLeft.x - topRight.x);
		      std::cout << "angle :" << angle1 << std::endl;
		      if(angle1 < angle2)
		      rotate(p_imgCopy, (180 +angle1* 180 / 3.14), p_imgCopy);
		      else
		      {
			  angle1 = angle2;
			  rotate(p_imgCopy, (180 +angle2* 180 / 3.14), p_imgCopy);

		      }
		  topLeft.x = p_imgCopy.cols/2 - abs(sumX - topLeft.x);
		    if(topLeft.x < 0)
			topLeft.x = 0;
		  topLeft.y = p_imgCopy.rows/2 - abs(sumY - topLeft.y);
						    if(topLeft.y < 0)
							topLeft.y = 0;

		bottomRight.x =  abs( bottomRight.x - sumX) + p_imgCopy.cols/2 ;
							if(bottomRight.x < 0)
								bottomRight.x = 0;
		bottomRight.y =  abs(bottomRight.y - sumY) + p_imgCopy.rows/2;
											if(bottomRight.y < 0)
												topLeft.y = 0;

		    p_imgCopy = p_imgCopy((cv::Rect(topLeft, bottomRight )));


		  //     if((sqrt((topLeft.x-bottomRight.x)*(topLeft.x-bottomRight.x)+(topLeft.y-bottomRight.y)*(topLeft.y-bottomRight.y))) > (sqrt((topRight.x-bottomLeft.x)*(topRight.x-bottomLeft.x)+(topRight.y-bottomLeft.y)*(topRight.y-bottomLeft.y))))

  int CellWidth =( p_imgCopy.cols)/9;
  int CellHeight = (p_imgCopy.rows)/9;
  //cvtColor(p_imgCopy,p_imgCopy,CV_RGB2GRAY);
  imshow("Copy", p_imgCopy);

			cv::Mat dst = p_imgCopy.clone();
  /*		    std::vector<std::vector<cv::Point> > contours;
			cv::findContours(dst.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			for (unsigned int m = 0; m < contours.size(); m++)
			{
			    if (cv::contourArea(contours[m]) < 100)
				cv::drawContours(dst, contours, m, cv::Scalar(0), -1);
			}


		  		    tesseract::TessBaseAPI tess;
		  		       tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
		  		       tess.SetVariable("tessedit_char_whitelist", "0123456789");
		  		       tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
		  		       tess.SetImage((uchar*)dst.data, dst.cols, dst.rows, 1, dst.cols);

		  		       char* out = tess.GetUTF8Text();
		  		       std::cout << out << std::endl;*/


vector<vector<Mat> > cells;

cells.resize(9);
  for (int f = 0; f < 9; ++f)
	  cells[f].resize(9);

  for(int j = 0; j < 9; ++j)
  {
	  for(int f = 0; f < 9; ++f)
	  {
		 // cells[f][j] = (Mat_<double>(2,3) << 1, 0, p_imgCopy.cols/2-sumX, 0, 1, p_imgCopy.rows/2-sumY);
		  	 // std::cout << "CellWidth * f " << CellWidth * f << " CellHeight * j " << CellHeight * j << " CellWidth * (f+1) " << CellWidth * (f+1) << " CellHeight * (j+1) " << CellHeight * (j+1) << std::endl;
		  cells[f][j] = p_imgCopy((cv::Rect(Point2f(CellWidth * f,CellHeight * j ), Point2f(CellWidth * (f+1),CellHeight * (j+1)) )));

		  cv::Mat dst =  cells[f][j].clone();
		  		  		    std::vector<std::vector<cv::Point> > contours;
		  		  			cv::findContours(dst.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		  		  			for (unsigned int m = 0; m < contours.size(); m++)
		  		  			{
		  		  			    if (cv::contourArea(contours[m]) < 100)
		  		  			        cv::drawContours(dst, contours, m, cv::Scalar(0), -1);
		  		  			}


		  		  		    tesseract::TessBaseAPI tess;
		  		  		       tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
		  		  		       tess.SetVariable("tessedit_char_whitelist", "0123456789");
		  		  		       tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
		  		  		       tess.SetImage((uchar*)dst.data, dst.cols, dst.rows, 1, dst.cols);

		  		  		       char* out = tess.GetUTF8Text();
		  		  		       std::cout << out << std::endl;
		  		  		   putText( p_imgCopy2, out,cvPoint(topLeft.x + CellWidth * f,topLeft.y + CellHeight * j), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1,8);
		  				 imshow("Copy", p_imgCopy2);

	  }
  }



		  		         for(i = angles.begin(); i != angles.end(); ++i)
		  		         {
	  		           //  std::cout << "firstPass" << i->x << ", " << i->y << std::endl;
		  		         }

		  		         }
		  		         }




		}

	    imshow("Original", p_imgOriginal);			// original image with detectec ball overlay
		    imshow("Processed", p_imgProcessed);		// image after processing

		if(charCheckForEscKey == 27) break;				// if Esc key (ASCII 27) was pressed, jump out of while loop

		//cvReleaseMemStorage(&p_strStorage);
	}

	// end while

	cvReleaseCapture(&p_capWebcam);					// release memory as applicable

	cvDestroyWindow("Original");
	cvDestroyWindow("Processed");

	return(0);
}



bool acceptLinePair(Vec2f line1, Vec2f line2, float minTheta)
{
    float theta1 = line1[1], theta2 = line2[1];

    if(theta1 < minTheta)
    {
        theta1 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    if(theta2 < minTheta)
    {
        theta2 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    return abs(theta1 - theta2) > minTheta;
}

// the long nasty wikipedia line-intersection equation...bleh...
Point2f computeIntersect(Vec2f line1, Vec2f line2)
{
    vector<Point2f> p1 = lineToPointPair(line1);
    vector<Point2f> p2 = lineToPointPair(line2);

    float denom = (p1[0].x - p1[1].x)*(p2[0].y - p2[1].y) - (p1[0].y - p1[1].y)*(p2[0].x - p2[1].x);
    Point2f intersect(((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].x - p2[1].x) -
                       (p1[0].x - p1[1].x)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom,
                      ((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].y - p2[1].y) -
                       (p1[0].y - p1[1].y)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom);

    return intersect;
}

vector<Point2f> lineToPointPair(Vec2f line)
{
    vector<Point2f> points;

    float r = line[0], t = line[1];
    double cos_t = cos(t), sin_t = sin(t);
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;

    points.push_back(Point2f(x0 + alpha*(-sin_t), y0 + alpha*cos_t));
    points.push_back(Point2f(x0 - alpha*(-sin_t), y0 - alpha*cos_t));

    return points;
}


void rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    //int len = std::max(src.cols, src.rows);
    cv::Point2f pt(src.cols/2, src.rows/2);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);

    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
}








