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
#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;
struct Ray
{

public:
    OpenMesh::Vec3f m_point;
    OpenMesh::Vec3f m_direction;

    //Getting intersection point by an algorithm found in ISBN: 1-55860-594-0 page 481
    OpenMesh::Vec3f faceIntersect(MyMesh &mesh, OpenMesh::FaceHandle &face) const
    {
        std::array<OpenMesh::DefaultTraits::Point, 3> triangleVertexes;

        int i = 0;
        OpenMesh::PolyConnectivity::ConstFaceVertexIter vertex = mesh.fv_iter(face);

        while (vertex.is_valid())
        {
            OpenMesh::Vec3f ver = mesh.point(vertex);
            triangleVertexes[i] = ver;
            ++vertex;
            ++i;
        }

        OpenMesh::DefaultTraits::Point v0 = triangleVertexes.at(0);
        OpenMesh::DefaultTraits::Point v1 = triangleVertexes.at(1);
        OpenMesh::DefaultTraits::Point v2 = triangleVertexes.at(2);

        //parameters t, u, v
        OpenMesh::Vec3f parameters = {

            OpenMesh::dot(OpenMesh::cross((m_point - v0), v1 - v0), v2 - v0),
            OpenMesh::dot(OpenMesh::cross(m_direction, v2 - v0), m_point - v0),
            OpenMesh::dot(OpenMesh::cross(m_point - v0, v1 - v0), m_direction)

        };

        float normalizeCoeffitient = (OpenMesh::dot(OpenMesh::cross(m_direction, v2 - v0), v1 - v0));

        parameters /= normalizeCoeffitient;

        float u = parameters[1];
        float v = parameters[2];

        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f || u + v > 1)
            return OpenMesh::Vec3f{-1, 0, 0};

        return parameters;
    }

    std::vector<float> intersect(MyMesh &t_mesh) const
    {
        // TODO divide space into cubes

        std::vector<float> intersectionParams;

        for (auto face : t_mesh.faces())
        {

            OpenMesh::Vec3f intersectionParameters = faceIntersect(t_mesh, face);

            if (intersectionParameters[0] >= 0.0f)
            {
                intersectionParams.push_back(intersectionParameters[0]);
            }
        }

        return intersectionParams;
    }

    OpenMesh::Vec3f getPointAt(float t) const
    {
        return m_point + m_direction * t;
    }
};

class ImplicitSurface
{

public:
    virtual float evaluate(const cv::Point3f &point) const = 0;
    virtual float getBoundingBoxWidth() const = 0;
    virtual float getBoundingBoxHeight() const = 0;
    virtual float getBoundingBoxDepth() const = 0;
    virtual cv::Point3f getBoundingBoxCorner() const = 0;
};

class VoxelizedMesh : public ImplicitSurface
{

private:
    MyMesh m_mesh;
    std::unique_ptr<PolyVox::SimpleVolume<uint8_t>> m_volume;
    std::array<OpenMesh::Vec3f, 8> m_boundingBox;

    void createBoundingBox()
    {

        float minX, minY, minZ;
        float maxX, maxY, maxZ;

        auto firstPoint = m_mesh.point(m_mesh.vertices_begin());

        minX = firstPoint[0];
        maxX = firstPoint[0];
        minY = firstPoint[1];
        maxY = firstPoint[1];
        minZ = firstPoint[2];
        maxZ = firstPoint[2];

        for (auto vertex : m_mesh.vertices())
        {
            auto point = m_mesh.point(vertex);

            if (minX > point[0])
                minX = point[0];

            if (minY > point[1])
                minY = point[1];

            if (minZ > point[2])
                minZ = point[2];

            if (maxX < point[0])
                maxX = point[0];

            if (maxY < point[1])
                maxY = point[1];

            if (maxZ < point[2])
                maxZ = point[2];
        }

        m_boundingBox.at(0) = OpenMesh::Vec3f(minX, minY, minZ); // bottom closer left
        m_boundingBox.at(1) = OpenMesh::Vec3f(maxX, minY, minZ); // bottom closer right
        m_boundingBox.at(2) = OpenMesh::Vec3f(maxX, minY, maxZ); // bottom further right
        m_boundingBox.at(3) = OpenMesh::Vec3f(minX, minY, maxZ); // bottom further left
        m_boundingBox.at(4) = OpenMesh::Vec3f(minX, maxY, minZ); // top closer left
        m_boundingBox.at(5) = OpenMesh::Vec3f(maxX, maxY, minZ); // top closer right
        m_boundingBox.at(6) = OpenMesh::Vec3f(maxX, maxY, maxZ); // top further right
        m_boundingBox.at(7) = OpenMesh::Vec3f(minX, maxY, maxZ); // top further left
    }

public:
    VoxelizedMesh(MyMesh t_mesh) : m_mesh{t_mesh}
    {
    }

    MyMesh getRaysAsMesh(const int t_resolutionX, const int t_resolutionY, const int t_resolutionZ)
    {

        MyMesh meshCopy;

        createBoundingBox();

        auto corner = getBoundingBoxCorner();

        float incrementX = getBoundingBoxWidth() / t_resolutionX;
        float incrementY = getBoundingBoxHeight() / t_resolutionY;
        float incrementZ = getBoundingBoxDepth() / t_resolutionZ;

        for (int y = 0; y < t_resolutionY; y++)
        {
            Ray r;
            OpenMesh::Vec3f rayStartPos{corner.x, corner.y + y * incrementY, corner.z};
            for (int x = 0; x < t_resolutionX; x++)
            {

                rayStartPos[0] = corner.x + x * incrementX;

                r.m_point = rayStartPos;
                r.m_direction = OpenMesh::Vec3f{0, 0, 1};
                //auto intersections = r.intersect(meshCopy);

                //hack, you cant create lines in openmesh, so this is going to be a narrow triangle
                auto vh1 = meshCopy.add_vertex(r.m_point);
                auto vh2 = meshCopy.add_vertex(r.m_point + OpenMesh::Vec3f(0, 0, 0.001));
                auto vh3 = meshCopy.add_vertex(r.getPointAt(10.0f));

                meshCopy.add_face(vh1, vh2, vh3);
            }
        }

        return meshCopy;
    }

    MyMesh getIntersections(const int t_resolutionX, const int t_resolutionY, const int t_resolutionZ)
    {

        MyMesh meshCopy;

        createBoundingBox();

        auto corner = getBoundingBoxCorner();

        float incrementX = getBoundingBoxWidth() / t_resolutionX;
        float incrementY = getBoundingBoxHeight() / t_resolutionY;
        float incrementZ = getBoundingBoxDepth() / t_resolutionZ;

        for (int y = 0; y < t_resolutionY; y++)
        {
            Ray r;
            OpenMesh::Vec3f rayStartPos{corner.x, corner.y + y * incrementY, corner.z};
            for (int x = 0; x < t_resolutionX; x++)
            {

                rayStartPos[0] = corner.x + x * incrementX;

                r.m_point = rayStartPos;
                r.m_direction = OpenMesh::Vec3f{0, 0, 1};
                auto intersections = r.intersect(m_mesh);

                std::sort(intersections.begin(), intersections.end(), [](float a, float b) { return a >= b; });

                //hack, you cant create lines in openmesh, so this is going to be a narrow triangle
                auto vh1 = meshCopy.add_vertex(r.m_point);
                auto vh2 = meshCopy.add_vertex(r.m_point + OpenMesh::Vec3f(0, 0, -0.1f));
                std::cout << "(" << x << "," << y << "):" << std::endl;
                for (auto t : intersections)
                {

                    auto vec = r.getPointAt(t);
                    auto vh3 = meshCopy.add_vertex(vec);
                    meshCopy.add_face(vh2, vh3, vh1);
                    std::cout << " Param " << t << " ( " << vec[0] << " " << vec[1] << " " << vec[2] << " ) , ";
                }

                if (intersections.empty())
                {

                    auto vh3 = meshCopy.add_vertex(r.getPointAt(20.0f));
                    meshCopy.add_face(vh2, vh3, vh1);
                    std::cout << "Not found.";
                }

                std::cout << std::endl;
            }
        }

        return meshCopy;
    }

    void voxelize(const int t_resolutionX, const int t_resolutionY, const int t_resolutionZ)
    {

        if (m_volume)
            m_volume.release();

        createBoundingBox();

        m_volume = std::make_unique<PolyVox::SimpleVolume<uint8_t>>(PolyVox::Region(PolyVox::Vector3DInt32(0, 0, 0), PolyVox::Vector3DInt32(t_resolutionX, t_resolutionY, t_resolutionZ)));

        auto corner = getBoundingBoxCorner();

        float incrementX = getBoundingBoxWidth() / t_resolutionX;
        float incrementY = getBoundingBoxHeight() / t_resolutionY;
        float incrementZ = getBoundingBoxDepth() / t_resolutionZ;

        std::tuple<Ray, std::vector<OpenMesh::FaceHandle>> rayContainer[t_resolutionY][t_resolutionX];

        for (int y = 0; y < t_resolutionY; y++)
        {
            OpenMesh::Vec3f rayStartPos{corner.x, corner.y + y * incrementY + incrementY / 2, corner.z};
            for (int x = 0; x < t_resolutionX; x++)
            {
                Ray r;

                rayStartPos[0] = corner.x + x * incrementX + incrementX / 2;
                r.m_point = rayStartPos;
                r.m_direction = OpenMesh::Vec3f{0, 0, 1};

                auto tuple = std::make_tuple(r, std::vector<OpenMesh::FaceHandle>());

                rayContainer[y][x] = tuple;
            }
        }

        for (auto face : m_mesh.faces())
        {

            std::array<OpenMesh::DefaultTraits::Point, 3> triangleVertexes;

            int i = 0;
            OpenMesh::PolyConnectivity::ConstFaceVertexIter vertex = m_mesh.fv_iter(face);

            while (vertex.is_valid())
            {
                OpenMesh::Vec3f ver = m_mesh.point(vertex);
                triangleVertexes[i] = ver;
                ++vertex;
                ++i;
            }

            float minX = std::min({(triangleVertexes[0][0] - corner.x), (triangleVertexes[1][0] - corner.x), (triangleVertexes[2][0] - corner.x)});
            float minY = std::min({(triangleVertexes[0][1] - corner.y), (triangleVertexes[1][1] - corner.y), (triangleVertexes[2][1] - corner.y)});
            float maxX = std::max({(triangleVertexes[0][0] - corner.x), (triangleVertexes[1][0] - corner.x), (triangleVertexes[2][0] - corner.x)});
            float maxY = std::max({(triangleVertexes[0][1] - corner.y), (triangleVertexes[1][1] - corner.y), (triangleVertexes[2][1] - corner.y)});

            size_t minXIndex = std::clamp((int)std::floor(minX / incrementX), 0, t_resolutionX - 1);
            size_t minYIndex = std::clamp((int)std::floor(minY / incrementY), 0, t_resolutionY - 1);
            size_t maxXIndex = std::clamp((int)std::floor(maxX / incrementX), 0, t_resolutionX - 1);
            size_t maxYIndex = std::clamp((int)std::floor(maxY / incrementY), 0, t_resolutionY - 1);

            for (int y = minYIndex; y <= maxYIndex; y++)
            {
                for (int x = minXIndex; x <= maxXIndex; x++)
                {
                    std::get<1>(rayContainer[y][x]).push_back(face);
                }
            }
        }

        for (int y = 0; y < t_resolutionY; y++)
        {
            for (int x = 0; x < t_resolutionX; x++)
            {
                auto ray = std::get<0>(rayContainer[y][x]);
                auto faces = std::get<1>(rayContainer[y][x]);

                std::vector<float> intersectionparams;

                for (auto face : faces)
                {

                    //t,u,v coords
                    auto intersection = ray.faceIntersect(m_mesh, face);

                    if (intersection[0] >= 0.0f)
                    {
                        intersectionparams.push_back(intersection[0]);
                    }
                }

                if (!intersectionparams.empty())
                {

                    //std::sort(intersectionparams.begin(), intersectionparams.end(), [](float a, float b) { return a >= b; });
                    std::sort(intersectionparams.begin(), intersectionparams.end());

                    auto intersectionIterator = intersectionparams.begin();
                    bool isInside = false;

                    for (int z = 0; z < t_resolutionZ; ++z)
                    {
                        if ((z * incrementZ + incrementZ / 2) >= *intersectionIterator)
                        {
                            isInside = !isInside;
                            ++intersectionIterator;
                        }

                        if (isInside)
                        {
                            m_volume->setVoxelAt(x, y, z, 255);
                        }
                        else
                        {
                            m_volume->setVoxelAt(x, y, z, 0);
                        }
                    }
                }
            }
        }
    }

    float evaluate(const cv::Point3f &point) const override
    {

        //TODO is this casting ok?
        return m_volume->getVoxelAt(point.x, point.y, point.z);
    };

    //X coordinate
    float getBoundingBoxWidth() const override
    {
        return (m_boundingBox[1] - m_boundingBox[0]).length();
    };

    //Y coordinate
    float getBoundingBoxHeight() const override
    {
        return (m_boundingBox[4] - m_boundingBox[0]).length();
    };

    //Z coordinate
    float getBoundingBoxDepth() const override
    {
        return (m_boundingBox[3] - m_boundingBox[0]).length();
    };

    cv::Point3f getBoundingBoxCorner() const override
    {

        OpenMesh::Vec3f corner = m_boundingBox.at(0);

        return cv::Point3f(corner[0], corner[1], corner[2]);
    };

    //Only for testing purposes
    MyMesh getVoxelAsMesh()
    {

        PolyVox::SurfaceMesh<PolyVox::PositionMaterialNormal> surfaceMesh;
        PolyVox::CubicSurfaceExtractorWithNormals<PolyVox::SimpleVolume<uint8_t>> surfaceExtractor(&(*m_volume), m_volume->getEnclosingRegion(), &surfaceMesh);

        surfaceExtractor.execute();

        const std::vector<uint32_t> &vecIndices = surfaceMesh.getIndices();
        const std::vector<PolyVox::PositionMaterialNormal> &vecVertices = surfaceMesh.getVertices();

        std::vector<MyMesh::VertexHandle> handles;
        MyMesh om_mesh;

        for (int i = 0; !vecVertices.empty() && i < vecVertices.size(); i++)
        {

            PolyVox::Vector3DFloat pos = vecVertices.at(i).getPosition();

            handles.push_back(om_mesh.add_vertex(MyMesh::Point(pos.getX(), pos.getY(), pos.getZ())));
        }

        for (int i = 0; !vecIndices.empty() && i < vecIndices.size() - 2; i += 3)
        {

            int index0 = vecIndices.at(i);
            int index1 = vecIndices.at(i + 1);
            int index2 = vecIndices.at(i + 2);

            om_mesh.add_face({handles.at(index0), handles.at(index1), handles.at(index2)});
        }

        return om_mesh;
    }
};

class ImplicitSphere : public ImplicitSurface
{

public:
    cv::Point3f center;
    float radius;

    ImplicitSphere() : center(cv::Point3f(0, 0, 0)), radius(1)
    {
    }

    ImplicitSphere(const cv::Point3f &center, float radius) : center(center), radius(radius)
    {
    }

    virtual float evaluate(const cv::Point3f &point) const override
    {
        float x_member = center.x - point.x;
        float y_member = center.y - point.y;
        float z_member = center.z - point.z;

        return (x_member * x_member) + (y_member * y_member) + (z_member * z_member) - radius * radius;
    };

    virtual float getBoundingBoxWidth() const override
    {
        return this->radius * 2;
    };

    virtual float getBoundingBoxHeight() const override
    {
        return this->radius * 2;
    };

    virtual float getBoundingBoxDepth() const override
    {
        return this->radius * 2;
    };

    virtual cv::Point3f getBoundingBoxCorner() const override
    {
        return cv::Point3f(this->center.x - this->radius, this->center.y - this->radius, this->center.z - this->radius);
    };
};

std::string type2str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

cv::Mat GrowingSwallow(const cv::Mat &shadow, const cv::Mat &part, const cv::Mat &partUp, float selfSupportThreshold)
{

    cv::Mat intersect = cv::Mat::zeros(shadow.size(), shadow.type());

    bitwise_and(part, partUp, intersect);

    cv::Mat dilation = intersect.clone();
    cv::Mat substractResult = intersect.clone();
    cv::Mat substractTempResult = intersect.clone();
    cv::Mat closePoints = intersect.clone();
    cv::Mat support = shadow.clone();
    cv::Mat invertedPart = part.clone();

    cv::bitwise_not(invertedPart, invertedPart);

    cv::distanceTransform(invertedPart, closePoints, cv::DIST_L2, CV_8U);

#ifdef DEBUG

    imwrite("closePoints.jpg", closePoints);
    imwrite("partUp.jpg", partUp);
    imwrite("part.jpg", part);
    imwrite("shadow.jpg", support);

#endif

    threshold(closePoints, closePoints, selfSupportThreshold, 255.0, cv::THRESH_BINARY);
    closePoints.convertTo(closePoints, CV_8U);

    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    int i = 0;
#ifdef DEBUG

    imwrite("closePointsThreshold.jpg", closePoints);

#endif

    while (sum(substractResult) != cv::Scalar(0.0))
    {

        dilate(intersect, dilation, structuringElement);

        subtract(dilation, intersect, substractResult);
        bitwise_and(substractResult, closePoints, substractTempResult);

#ifdef DEBUG
        imwrite(to_string(i) + "subtractResult1.jpg", substractResult);
#endif

        bitwise_and(substractTempResult, support, substractResult);

#ifdef DEBUG

        imwrite(to_string(i) + "subtractResult2.jpg", substractResult);
#endif

        bitwise_or(intersect, substractResult, intersect);

        subtract(support, substractResult, support);
#ifdef DEBUG

        imwrite(to_string(i) + "intersect.jpg", intersect);
        imwrite(to_string(i) + "dilation.jpg", dilation);
        imwrite(to_string(i) + "support.jpg", support);
#endif

        i++;
    }

    return support;
}

cv::Mat RegionSubtraction(const cv::Mat &part_i, const cv::Mat &part_i_plus1, const cv::Mat &support_i_plus1, float selfSupportThreshold)
{

    cv::Mat support = cv::Mat::zeros(part_i.size(), part_i.type());
    cv::Mat shadow = cv::Mat::zeros(part_i.size(), part_i.type());

    subtract(part_i_plus1, part_i, shadow);

    cv::Mat support_candidate = GrowingSwallow(shadow, part_i, part_i_plus1, selfSupportThreshold);

    bitwise_or(support_candidate, support_i_plus1, support);
    subtract(support, part_i, support);

    return support;
}

//TODO: use centroidal voronoi tessellation to minimize anchor points
//NOTE: Dynamic grid resolution?
cv::Mat GenerateAnchorMap(const cv::Mat &support_i, float anchorRadius)
{

    cv::Mat anchor_point_image = cv::Mat::zeros(support_i.size(), support_i.type());
    cv::Mat support_copy = support_i.clone();

    int grid_res_x = 20;
    int grid_res_y = 20;

    std::vector<cv::Point2i> anchorMap;

    int grid_step_x = support_copy.size().width / grid_res_x;
    int grid_step_y = support_copy.size().height / grid_res_y;

    //First phase sample grid
    for (int x = 0; x < support_copy.size().width; x += grid_step_x)
    {

        for (int y = 0; y < support_copy.size().height; y += grid_step_y)
        {

            if (support_copy.at<unsigned char>(x, y) == 255)
            {

                cv::Point2i anchorPoint{y, x};

                anchorMap.push_back(anchorPoint);
                cv::circle(anchor_point_image, anchorPoint, 1, cv::Scalar(255), -1);
            }
        }
    }

    cv::Mat sub = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);
    subtract(support_copy, sub, support_copy);
    //Second phase scan along lines

    for (int y = 0; y < support_copy.size().height; y += grid_step_y)
    {

        bool intersection_line_detected = false;
        int intersection_begin = 0;

        for (int x = 0; x < support_copy.size().width; x++)
        {
            unsigned char wat = support_copy.at<unsigned char>(x, y);
            if (support_copy.at<unsigned char>(x, y) == 255 && !intersection_line_detected)
            {

                intersection_line_detected = true;
                intersection_begin = x;
            }

            if (support_copy.at<unsigned char>(x, y) != 255 && intersection_line_detected)
            {

                intersection_line_detected = false;

                cv::Point2i anchorPoint{y, intersection_begin + (x - intersection_begin) / 2};

                anchorMap.push_back(anchorPoint);
                circle(anchor_point_image, anchorPoint, 1, cv::Scalar(255), -1);
                sub = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);
                subtract(support_copy, sub, support_copy);
                x = intersection_begin - 1;
                intersection_begin = 0;
            }
        }
    }

    for (int x = 0; x < support_copy.size().width; x += grid_step_x)
    {

        bool intersection_line_detected = false;
        int intersection_begin = 0;

        for (int y = 0; y < support_copy.size().height; y++)
        {

            if (support_copy.at<unsigned char>(x, y) == 255 && !intersection_line_detected)
            {

                intersection_line_detected = true;
                intersection_begin = y;
            }

            if (support_copy.at<unsigned char>(x, y) != 255 && intersection_line_detected)
            {

                intersection_line_detected = false;

                cv::Point2i anchorPoint{intersection_begin + (y - intersection_begin) / 2, x};

                anchorMap.push_back(anchorPoint);
                circle(anchor_point_image, anchorPoint, 1, cv::Scalar(255), -1);
                sub = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);
                subtract(support_copy, sub, support_copy);
                y = intersection_begin - 1;
            }
        }
    }

    //Third phase scan pixel by pixel
    for (int x = 0; x < support_copy.size().width; x++)
    {

        for (int y = 0; y < support_copy.size().height; y++)
        {

            if (support_copy.at<unsigned char>(x, y) == 255)
            {

                cv::Point2i anchorPoint{y, x};
                anchorMap.push_back(anchorPoint);

                circle(anchor_point_image, anchorPoint, 1, cv::Scalar(255), -1);
                sub = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);
                subtract(support_copy, sub, support_copy);
            }
        }
    }

    return anchor_point_image;
}

cv::Mat RegionSubtractionSLA(const cv::Mat &part_i, const cv::Mat &part_i_plus1, const cv::Mat &anchor_support_i_plus1, float selfSupportThreshold, float anchorRadius)
{

    cv::Mat anchorMap = cv::Mat::zeros(part_i.size(), part_i.type());
    cv::Mat shadow = cv::Mat::zeros(part_i.size(), part_i.type());
    cv::Mat PA_plus1 = cv::Mat::zeros(part_i.size(), part_i.type());

    subtract(part_i_plus1, part_i, shadow);
    subtract(anchor_support_i_plus1, part_i, PA_plus1);

    cv::Mat support_candidate = GrowingSwallow(shadow, part_i, part_i_plus1, selfSupportThreshold);
    cv::Mat support_candidate2 = GrowingSwallow(support_candidate, PA_plus1, PA_plus1, anchorRadius);
    imwrite("pap1.jpg", PA_plus1);
    cv::Mat anchor_candidate = GenerateAnchorMap(support_candidate2, anchorRadius);

    imwrite("anchor_cand.jpg", anchor_candidate);
    bitwise_or(anchor_candidate, PA_plus1, anchorMap);
    imwrite("anchor_cand2.jpg", anchor_candidate);
    imwrite("anchor_map.jpg", anchorMap);
    return anchorMap;
}

static int selectedSlice = 0;

int main(int argc, char const *argv[])
{

    MyMesh in_mesh;

    std::string filename;

    if (argc >= 2)
    {

        filename = argv[1];
        std::cout << "Filename as argument: " << filename << std::endl;
    }
    else
    {
        filename = "input.obj";
    }

    if (!OpenMesh::IO::read_mesh(in_mesh, filename))
    {
        std::cout << "Could not read mesh input.obj .\n";
        exit(1);
    }

    VoxelizedMesh voxel{in_mesh};
    //MyMesh mesh2 = voxel.getIntersections(10, 10, 10);
    voxel.voxelize(200, 200, 200);

    if (!OpenMesh::IO::write_mesh(voxel.getVoxelAsMesh(), "output2.obj"))
    {
        std::cerr << "write error\n";
        exit(1);
    }
}

/*int main(int argc, char const *argv[])
{
    float anchorSupport = 10.0f;
    float selfSupport = 5.0f;

    if (argc == 3)
    {
        anchorSupport = std::stof(argv[2]);
        selfSupport = std::stof(argv[1]);
    }

    if (argc == 2)
    {
        selfSupport = std::stof(argv[1]);
    }

    ImplicitSphere sphere{cv::Point3f(0, 0, 210), 10};

    float filamentDiameter = 0.125f;

    int gridWidth = (int)sphere.getBoundingBoxWidth() / filamentDiameter;
    int gridHeight = (int)sphere.getBoundingBoxDepth() / filamentDiameter;
    int sliceNumber = 1024;

    cv::Point3f boundingBoxCorner = sphere.getBoundingBoxCorner();

    cv::Mat slice[sliceNumber + 1];

    std::cout << "Resolution is: " << gridWidth << " x " << gridHeight << " x " << sliceNumber << std::endl;

    for (int z = 0; z < sliceNumber + 1; z++)
    {

        std::cout << "Sampling slice #" << z << std::endl;
        slice[z].create(gridWidth, gridHeight, CV_8UC(1));

        for (int x = 0; x < gridWidth; x++)
        {

            for (int y = 0; y < gridHeight; y++)
            {

                cv::Point3f evaluationPoint{boundingBoxCorner.x + x * filamentDiameter, boundingBoxCorner.y + y * filamentDiameter, boundingBoxCorner.z + z * filamentDiameter};

                slice[z].at<unsigned char>(x, y) = sphere.evaluate(evaluationPoint) >= 0.0f ? 0 : 255;
            }
        }
    }

    cv::Mat support[sliceNumber + 1];
    PolyVox::SimpleVolume<uint8_t> volData(PolyVox::Region(PolyVox::Vector3DInt32(0, 0, 0), PolyVox::Vector3DInt32(gridWidth, gridHeight, sliceNumber)));

    support[sliceNumber] = cv::Mat::zeros(slice[0].size(), slice[0].type());
    for (int z = sliceNumber - 1; z >= 0; z--)
    {

        std::cout << "Generating support for layer #" << z << std::endl;

        support[z].create(gridWidth, gridHeight, CV_8UC(1));
        support[z] = RegionSubtractionSLA(slice[z], slice[z + 1], support[z + 1], selfSupport, anchorSupport);
    }

    for (int z = 0; z < sliceNumber; z++)
    {

        std::cout << "Creating voxel layer #" << z << std::endl;

        for (int x = 0; x < gridWidth; x++)
        {

            for (int y = 0; y < gridHeight; y++)
            {

                volData.setVoxelAt(x, y, z, support[z].at<uint8_t>(x, y));
            }
        }
    }

    PolyVox::SurfaceMesh<PolyVox::PositionMaterialNormal> surfaceMesh;
    PolyVox::CubicSurfaceExtractorWithNormals<PolyVox::SimpleVolume<uint8_t>> surfaceExtractor(&volData, volData.getEnclosingRegion(), &surfaceMesh);

    surfaceExtractor.execute();

    const std::vector<uint32_t> &vecIndices = surfaceMesh.getIndices();
    const std::vector<PolyVox::PositionMaterialNormal> &vecVertices = surfaceMesh.getVertices();

    std::vector<MyMesh::VertexHandle> handles;
    MyMesh om_mesh;

    for (int i = 0; i < vecVertices.size(); i++)
    {

        PolyVox::Vector3DFloat pos = vecVertices.at(i).getPosition();

        handles.push_back(om_mesh.add_vertex(MyMesh::Point(pos.getX(), pos.getY(), pos.getZ())));
    }

    for (int i = 0; i < vecIndices.size() - 2; i += 3)
    {

        int index0 = vecIndices.at(i);
        int index1 = vecIndices.at(i + 1);
        int index2 = vecIndices.at(i + 2);

        om_mesh.add_face({handles.at(index0), handles.at(index1), handles.at(index2)});
    }
    if (!OpenMesh::IO::write_mesh(om_mesh, "output.obj"))
    {
        std::cerr << "write error\n";
        exit(1);
    }

    return 0;
}
*/