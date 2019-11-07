#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <PolyVoxCore/CubicSurfaceExtractorWithNormals.h>
#include <PolyVoxCore/MarchingCubesSurfaceExtractor.h>
#include <PolyVoxCore/SurfaceMesh.h>
#include <PolyVoxCore/SimpleVolume.h>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Utils/getopt.h>
#include <vector>

typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;

using namespace cv;
using namespace std;
using namespace PolyVox;

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

Mat RegionSubtraction(const Mat& part_i, const Mat& part_i_plus1, const Mat& support_i_plus1, float selfSupportThreshold){

    Mat support = Mat::zeros(part_i.size(),part_i.type());
    Mat shadow = Mat::zeros(part_i.size(),part_i.type());

    subtract(part_i_plus1, part_i, shadow);

    Mat support_candidate = GrowingSwallow(shadow, part_i, part_i_plus1, selfSupportThreshold);

    bitwise_or(support_candidate, support_i_plus1, support);
    subtract(support, part_i, support);

    return support;

}

//TODO: use centroidal voronoi tessellation to minimize anchor points
Mat GenerateAnchorMap(const Mat& support_i, float anchorRadius){

    Mat anchor_point_image = Mat::zeros(support_i.size(),support_i.type());
    Mat support_copy = support_i.clone();

    int grid_res_x = 20;
    int grid_res_y = 20;

    vector<Point2i> anchorMap;

    int grid_step_x = support_copy.size().width/grid_res_x;
    int grid_step_y = support_copy.size().height/grid_res_y;

    //First phase sample grid


    for(int x = 0;x<support_copy.size().width;x+=grid_step_x){

        for(int y = 0;y<support_copy.size().height;y+=grid_step_y){

            if( support_copy.at<unsigned char>(x,y) == 255 ){

                Point2i anchorPoint{x,y};

                anchorMap.push_back(anchorPoint);
                circle(anchor_point_image, anchorPoint, 1, Scalar(255), -1);

            }

        }   

    }

    support_copy = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);

    //Second phase scan along lines

    for(int y = 0;y<support_copy.size().height;y+=grid_step_y){
        
        bool intersection_line_detected = false;
        int intersection_begin = 0;

        for(int x = 0;x<support_copy.size().width;x++){

            if(support_copy.at<unsigned char>(x,y)==255 && !intersection_line_detected) {

                intersection_line_detected = true;
                intersection_begin = x;

            }
            
            if(support_copy.at<unsigned char>(x,y)!=255 && intersection_line_detected) {

                intersection_line_detected = false;

                Point2i anchorPoint{x + (x-intersection_begin)/2,y};

                anchorMap.push_back(anchorPoint);
                circle(anchor_point_image, anchorPoint, 1, Scalar(255), -1);
                support_copy = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);
                imwrite("testsupp.jpg",support_copy);
                x = intersection_begin-1;

            }

        }   

    }

    for(int x = 0;x<support_copy.size().width;x+=grid_step_x){

        bool intersection_line_detected = false;
        int intersection_begin = 0;

        for(int y = 0;y<support_copy.size().height;y++){

            if(support_copy.at<unsigned char>(x,y)==255 && !intersection_line_detected) {

                intersection_line_detected = true;
                intersection_begin = y;

            }
            
            if(support_copy.at<unsigned char>(x,y)!=255 && intersection_line_detected) {

                intersection_line_detected = false;

                Point2i anchorPoint{x, y + (y-intersection_begin)/2};

                anchorMap.push_back(anchorPoint);
                circle(anchor_point_image, anchorPoint, 1, Scalar(255), -1);
                support_copy = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);

                y = intersection_begin-1;

            }

        }   

    }

    //Third phase scan pixel by pixel
    for(int x = 0;x<support_copy.size().width;x++){

        for(int y = 0;y<support_copy.size().height;y++){

            if( support_copy.at<unsigned char>(x,y) == 255 ) {

                Point2i anchorPoint{ x, y };
                anchorMap.push_back(anchorPoint);

                circle(anchor_point_image, anchorPoint, 1, Scalar(255), -1);
                support_copy = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);

            }

        }   

    }

    return anchor_point_image;

}

Mat RegionSubtractionSLA(const Mat& part_i, const Mat& part_i_plus1, const Mat& anchor_support_i_plus1, float selfSupportThreshold, float anchorRadius){

    Mat anchorMap = Mat::zeros(part_i.size(),part_i.type());
    Mat shadow = Mat::zeros(part_i.size(),part_i.type());
    Mat PA_plus1 = Mat::zeros(part_i.size(),part_i.type());

    subtract(part_i_plus1, part_i, shadow);
    imwrite("shadowsla.jpg", shadow);
    subtract(anchor_support_i_plus1, part_i, PA_plus1);
    imwrite("paiplus.jpg", PA_plus1);

    Mat support_candidate = GrowingSwallow(shadow, part_i, part_i_plus1, selfSupportThreshold);
    Mat support_candidate2 = GrowingSwallow(support_candidate, PA_plus1, PA_plus1, anchorRadius);
    
    Mat anchor_candidate = GenerateAnchorMap(support_candidate2, anchorRadius);

    bitwise_or(anchor_candidate, PA_plus1, anchorMap);
    return anchorMap;

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

    Mat slice[sliceNumber+1];

    cout << "Resolution is: " << gridWidth << " x " << gridHeight << " x " << sliceNumber << endl;
 
    for(int z=0;z<sliceNumber+1;z++){

        cout << "Sampling slice #" << z << endl;
        slice[z].create(gridWidth, gridHeight, CV_8UC(1));

        for(int x = 0; x < gridWidth; x++)
        {
            
            for(int y = 0; y < gridHeight; y++)
            {

                Point3f evaluationPoint{boundingBoxCorner.x + x*filamentDiameter, boundingBoxCorner.y + y*filamentDiameter, boundingBoxCorner.z + z*filamentDiameter};

                slice[z].at<unsigned char>(x,y) = sphere.evaluate(evaluationPoint) >= 0.0f ? 0 : 255;

            }
            
        }

    }

    Mat support[sliceNumber+1];
    SimpleVolume<uint8_t> volData( Region(Vector3DInt32(0,0,0), Vector3DInt32(gridWidth, gridHeight, sliceNumber)));

    support[sliceNumber] = Mat::zeros(slice[0].size(), slice[0].type());
    for(int z=sliceNumber-1;z>=0;z--){

        cout << "Generating support for layer #" << z << endl;

        //support[z].create(gridWidth, gridHeight, CV_8UC(1));
        //support[z] = RegionSubtractionSLA(slice[z], slice[z+1], support[z+1], 100.0f, 3.0f);

        //Mat shadow;
        //subtract( slice[z+1], slice[z], shadow);
        //support[z] = GrowingSwallow( shadow, slice[z], slice[z+1], 3.0f);

        support[z] = RegionSubtraction( slice[z], slice[z+1], support[z+1], 1.4f);


    }

    for(int z=0;z<sliceNumber;z++){

        cout << "Creating voxel layer #" << z << endl;

        for(int x = 0; x < gridWidth; x++)
        {
            
            for(int y = 0; y < gridHeight; y++)
            {

              volData.setVoxelAt(x,y,z,support[z].at<uint8_t>(x,y));

            }
            
        }

    }

    SurfaceMesh<PositionMaterialNormal> surfaceMesh;
    CubicSurfaceExtractorWithNormals< SimpleVolume<uint8_t> > surfaceExtractor(&volData, volData.getEnclosingRegion(), &surfaceMesh);

    surfaceExtractor.execute();
    
    const vector<uint32_t>& vecIndices = surfaceMesh.getIndices();
    const vector<PositionMaterialNormal>& vecVertices = surfaceMesh.getVertices();
    
    vector<MyMesh::VertexHandle> handles;
    MyMesh om_mesh;

    for(int i=0; i < vecVertices.size();i++){

      Vector3DFloat pos = vecVertices.at(i).getPosition();

      handles.push_back(om_mesh.add_vertex( MyMesh::Point( pos.getX(), pos.getY(), pos.getZ() ) ));

    }

    for(int i=0; i < vecIndices.size()-2;i+=3){

      int index0 = vecIndices.at(i);
      int index1 = vecIndices.at(i+1);
      int index2 = vecIndices.at(i+2);

      om_mesh.add_face( { handles.at(index0) , handles.at(index1), handles.at(index2) } );

    }

    if (!OpenMesh::IO::write_mesh(om_mesh, "output.obj")) 
    {
      std::cerr << "write error\n";
      exit(1);
    }

    return 0;
}
