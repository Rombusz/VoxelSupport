#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

class ImplicitSurface{

    public:
        virtual float evaluate(const Point3f& point) const = 0;
        virtual float getBoundingBoxWidth() const = 0;
        virtual float getBoundingBoxHeight() const = 0;
        virtual float getBoundingBoxDepth() const = 0;
        virtual Point3f getBoundingBoxCorner() const = 0;

};

class ImplicitSphere : public ImplicitSurface{

    public:

        Point3f center;
        float radius;

        ImplicitSphere():center(Point3f(0,0,0)),radius(1){}

        ImplicitSphere(const Point3f& center, float radius):center(center),radius(radius)
        {

        }

        virtual float evaluate(const Point3f& point) const override{

            float x_member = center.x - point.x;
            float y_member = center.y - point.y;
            float z_member = center.z - point.z;

            return (x_member*x_member)+(y_member*y_member)+(z_member*z_member) - radius*radius;

        }

        virtual float getBoundingBoxWidth() const override{

            return this->radius*2;

        };

        virtual float getBoundingBoxHeight() const override{

            return this->radius*2;

        };

        virtual float getBoundingBoxDepth() const override{

            return this->radius*2;

        };

        virtual Point3f getBoundingBoxCorner() const override{

            return Point3f(this->center.x - this->radius, this->center.y - this->radius, this->center.z - this->radius);

        };

};

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

static void on_trackbar( int value, void* ptr)
{

}

Mat GrowingSwallow(const Mat& shadow, const Mat& part, const Mat& partUp, float selfSupportThreshold){

    Mat intersect = Mat::zeros(shadow.size(),shadow.type());

    bitwise_and(part, partUp, intersect);

    Mat dilation = intersect.clone();
    Mat substractResult = intersect.clone();
    Mat substractTempResult = intersect.clone();
    Mat closePoints = intersect.clone();
    Mat support = shadow.clone();
    Mat invertedPart = part.clone();

    bitwise_not(invertedPart,invertedPart);

    distanceTransform( invertedPart, closePoints, DIST_L2, CV_8U );
    
    imwrite("closePoints.jpg", closePoints);
    imwrite("partUp.jpg", partUp);
    imwrite("part.jpg", part);
    imwrite("shadow.jpg", support);

    threshold( closePoints, closePoints, selfSupportThreshold , 255.0, THRESH_BINARY);
    closePoints.convertTo(closePoints, CV_8U);

    Mat structuringElement = getStructuringElement( MORPH_RECT, Size( 3, 3 ) );

    int i=0;

    imwrite("closePointsThreshold.jpg", closePoints);

    while( sum(substractResult) != Scalar(0.0) ){

        dilate( intersect, dilation, structuringElement  );

        subtract( dilation, intersect, substractResult );
        bitwise_and(substractResult, closePoints, substractTempResult);

        imwrite(to_string(i)+"subtractResult1.jpg", substractResult);
        bitwise_and(substractTempResult, support, substractResult);
        imwrite(to_string(i)+"subtractResult2.jpg", substractResult);


        Mat tempIntersect = intersect.clone();

        bitwise_or( tempIntersect, substractResult, intersect );

        subtract( support, substractResult, support );

        imwrite(to_string(i)+"intersect.jpg", intersect);
        imwrite(to_string(i)+"dilation.jpg", dilation);
        imwrite(to_string(i)+"support.jpg", support);

        i++;

    }

    return support;

}

static int selectedSlice = 0;

int main(int argc, char const *argv[])
{
    
    ImplicitSphere sphere{Point3f(0,0,210),10};

    float filamentDiameter = 0.125f;

    int gridWidth = (int)sphere.getBoundingBoxWidth()/filamentDiameter;
    int gridHeight = (int)sphere.getBoundingBoxDepth()/filamentDiameter;
    int sliceNumber = (int)sphere.getBoundingBoxHeight()/filamentDiameter;

    Point3f boundingBoxCorner = sphere.getBoundingBoxCorner();

    Mat M[sliceNumber+1];

    cout << "Resolution is: " << gridWidth << " x " << gridHeight << " x " << sliceNumber << endl;
 
    for(int z=0;z<sliceNumber+1;z++){

        cout << "Doing slice #" << z << endl;
        M[z].create(gridWidth, gridHeight, CV_8UC(1));

        for(int x = 0; x < gridWidth; x++)
        {
            
            for(int y = 0; y < gridHeight; y++)
            {

                Point3f evaluationPoint{boundingBoxCorner.x + x*filamentDiameter, boundingBoxCorner.y + y*filamentDiameter, boundingBoxCorner.z + z*filamentDiameter};

                M[z].at<unsigned char>(x,y) = sphere.evaluate(evaluationPoint) >= 0.0f ? 0 : 255;

            }
            
        }

    }

    Mat support[sliceNumber+1];

    for(int z=0;z<sliceNumber;z++){

        support[z].create(gridWidth, gridHeight, CV_8UC(1));
        Mat shadow;
        subtract( M[z+1], M[z], shadow);
        support[z] = GrowingSwallow( shadow, M[z], M[z+1], 5.0f);

    }

   namedWindow("intersect_slice", WINDOW_AUTOSIZE);
   createTrackbar( "Slice number", "intersect_slice", &selectedSlice, sliceNumber, on_trackbar );

    bool exit = false;

    while(!exit)
    {
        imshow("intersect_slice",M[selectedSlice]);
        imshow("support_slice",support[selectedSlice]);
        int key = waitKey(10);

        exit = (key == 'q');

    }

    return 0;
}
