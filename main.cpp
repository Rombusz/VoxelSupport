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
#include <map>
#include <memory>
#include <algorithm>
#include <cmath>
#include "ldni/ldni.hh"
#include "libgeom/geometry.hh"
#include <string>

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

static int anchorId = 0;

static bool insidep(const LDNI &ldni, const std::array<size_t, 3> &index)
{
    int votes = 0;
    for (int c0 = 0; c0 < 3; ++c0)
    {
        int c1 = (c0 + 1) % 3, c2 = (c0 + 2) % 3;
        const auto &ray = ldni.rays[c0][index[c1] * (ldni.res[c2] + 1) + index[c2]];
        double distance = index[c0] * ldni.dirs[c0][c0];
        bool inside = false;
        for (const auto &dn : ray)
        {
            if (dn.d > distance)
                break;
            inside = !inside;
        }
        if (inside)
            ++votes;
    }
    return votes >= 2;
}

class ImplicitSurface
{

public:
    virtual float evaluate(const cv::Point3f &point) const = 0;
    virtual float getBoundingBoxWidth() const = 0;
    virtual float getBoundingBoxHeight() const = 0;
    virtual float getBoundingBoxDepth() const = 0;
    virtual cv::Point3f getBoundingBoxCorner() const = 0;
};
/*

class VoxelizedMesh : public ImplicitSurface
{

private:
    MyMesh m_mesh;
    std::unique_ptr<PolyVox::SimpleVolume<uint8_t>> m_volume;
    std::unique_ptr<PolyVox::SimpleVolume<uint8_t>> m_volumeX;
    std::unique_ptr<PolyVox::SimpleVolume<uint8_t>> m_volumeY;
    std::unique_ptr<PolyVox::SimpleVolume<uint8_t>> m_volumeZ;
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

        std::tuple<Ray, std::vector<OpenMesh::FaceHandle>> rayContainer[t_resolutionZ][t_resolutionY];

        for (int z = 0; z < t_resolutionZ; z++)
        {
            OpenMesh::Vec3f rayStartPos{corner.x, corner.y, corner.z + z * incrementZ + incrementZ / 2};
            for (int y = 0; y < t_resolutionY; y++)
            {
                Ray r;

                rayStartPos[1] = corner.y + y * incrementY + incrementY / 2;
                r.m_point = rayStartPos;
                r.m_direction = OpenMesh::Vec3f{1, 0, 0};

                auto tuple = std::make_tuple(r, std::vector<OpenMesh::FaceHandle>());

                rayContainer[z][y] = tuple;
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

            float minY = std::min({(triangleVertexes[0][1] - corner.y), (triangleVertexes[1][1] - corner.y), (triangleVertexes[2][1] - corner.y)});
            float minZ = std::min({(triangleVertexes[0][2] - corner.z), (triangleVertexes[1][2] - corner.z), (triangleVertexes[2][2] - corner.z)});
            float maxY = std::max({(triangleVertexes[0][1] - corner.y), (triangleVertexes[1][1] - corner.y), (triangleVertexes[2][1] - corner.y)});
            float maxZ = std::max({(triangleVertexes[0][2] - corner.z), (triangleVertexes[1][2] - corner.z), (triangleVertexes[2][2] - corner.z)});

            size_t minYIndex = std::clamp((int)std::floor(minY / incrementY) - 1, 0, t_resolutionY - 1);
            size_t minZIndex = std::clamp((int)std::floor(minZ / incrementZ) - 1, 0, t_resolutionZ - 1);
            size_t maxYIndex = std::clamp((int)std::ceil(maxY / incrementY) + 1, 0, t_resolutionY - 1);
            size_t maxZIndex = std::clamp((int)std::ceil(maxZ / incrementZ) + 1, 0, t_resolutionZ - 1);

            for (int z = minZIndex; z <= maxZIndex; z++)
            {
                for (int y = minYIndex; y <= maxYIndex; y++)
                {
                    std::get<1>(rayContainer[z][y]).push_back(face);
                }
            }
        }

        for (int z = 0; z < t_resolutionY; z++)
        {
            for (int y = 0; y < t_resolutionY; y++)
            {
                auto ray = std::get<0>(rayContainer[z][y]);
                auto faces = std::get<1>(rayContainer[z][y]);

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

                    std::sort(intersectionparams.begin(), intersectionparams.end());

                    auto intersectionIterator = intersectionparams.begin();
                    bool isInside = false;

                    for (int x = 0; x < t_resolutionX; ++x)
                    {

                        float t = *intersectionIterator;

                        if ((x * incrementX + incrementX / 2) >= t)
                        {
                            isInside = !isInside;
                            ++intersectionIterator;
                        }

                        if (isInside)
                        {
                            m_volumeX->setVoxelAt(x, y, z, 255);
                        }
                        else
                        {
                            m_volumeX->setVoxelAt(x, y, z, 0);
                        }
                    }
                }
            }
        }

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
                for (auto t : intersections)
                {

                    auto vec = r.getPointAt(t);
                    auto vh3 = meshCopy.add_vertex(vec);
                    meshCopy.add_face(vh2, vh3, vh1);
                }

                if (intersections.empty())
                {

                    auto vh3 = meshCopy.add_vertex(r.getPointAt(20.0f));
                    meshCopy.add_face(vh2, vh3, vh1);
                }
            }
        }

        return meshCopy;
    }

    void voxelize(const float unitPerVoxel)
    {

        if (m_volume)
            m_volume.release();

        createBoundingBox();

        const int t_resolutionX = std::ceil(getBoundingBoxWidth() / unitPerVoxel);
        const int t_resolutionY = std::ceil(getBoundingBoxHeight() / unitPerVoxel);
        const int t_resolutionZ = std::ceil(getBoundingBoxDepth() / unitPerVoxel);

        m_volume = std::make_unique<PolyVox::SimpleVolume<uint8_t>>(PolyVox::Region(PolyVox::Vector3DInt32(0, 0, 0), PolyVox::Vector3DInt32(t_resolutionX, t_resolutionY, t_resolutionZ)));

        voxelizeAlongX(t_resolutionX, t_resolutionY, t_resolutionZ);
        //voxelizeAlongY(t_resolutionX, t_resolutionY, t_resolutionZ);
        //voxelizeAlongZ(t_resolutionX, t_resolutionY, t_resolutionZ);

        for (int y = 0; y < t_resolutionY; y++)
        {
            for (int x = 0; x < t_resolutionX; x++)
            {
                for (int z = 0; z < t_resolutionZ; z++)
                {
                    int voxelValue = m_volumeX->getVoxelAt(x, y, z) + m_volumeY->getVoxelAt(x, y, z) + m_volumeZ->getVoxelAt(x, y, z);

                    if (voxelValue >= 2 * 255)
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

    MyMesh getVoxelXAsMesh()
    {

        PolyVox::SurfaceMesh<PolyVox::PositionMaterialNormal> surfaceMesh;
        PolyVox::CubicSurfaceExtractorWithNormals<PolyVox::SimpleVolume<uint8_t>> surfaceExtractor(&(*m_volumeX), m_volumeX->getEnclosingRegion(), &surfaceMesh);

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

    MyMesh getVoxelYAsMesh()
    {

        PolyVox::SurfaceMesh<PolyVox::PositionMaterialNormal> surfaceMesh;
        PolyVox::CubicSurfaceExtractorWithNormals<PolyVox::SimpleVolume<uint8_t>> surfaceExtractor(&(*m_volumeY), m_volumeY->getEnclosingRegion(), &surfaceMesh);

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

    MyMesh getVoxelZAsMesh()
    {

        PolyVox::SurfaceMesh<PolyVox::PositionMaterialNormal> surfaceMesh;
        PolyVox::CubicSurfaceExtractorWithNormals<PolyVox::SimpleVolume<uint8_t>> surfaceExtractor(&(*m_volumeZ), m_volumeZ->getEnclosingRegion(), &surfaceMesh);

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
*/

struct Anchor
{
    OpenMesh::Vec3i start, end;

    bool operator!=(const Anchor &other) const
    {
        return (start != other.start) && (end != other.end);
    }
};

struct ConnectionPointPair
{
    OpenMesh::Vec3i point1, point2;
};

class Connection
{
    std::array<Anchor, 3> m_anchors;
    bool isClosestValid;
    bool isSecondClosestValid;

    void connectToMainAnchor(const Anchor &other, std::vector<ConnectionPointPair> &pairs, float pillair_stability) const
    {
        int main_start_z = m_anchors[0].start[2];
        int main_end_z = m_anchors[0].end[2];
        int other_start_z = other.start[2];
        int other_end_z = other.end[2];

        OpenMesh::Vec2i main_proj{m_anchors[0].start[0], m_anchors[0].start[1]};
        OpenMesh::Vec2i other_proj{other.start[0], other.start[1]};

        int start_z = std::min(main_start_z, other_start_z);
        int end_z = std::max(main_end_z, other_end_z);

        //float estimated_connection_number = std::floor((start_z - end_z) / pillair_stability);
        float estimated_connection_number = start_z - end_z > pillair_stability ? 2 : 0;

        int connection_number = (int)estimated_connection_number;

        for (int i = 1; i <= connection_number; i++)
        {

            float connection_height = (start_z - end_z) / connection_number;
            auto pair = ConnectionPointPair{
                point1 : OpenMesh::Vec3i{main_proj[0], main_proj[1], std::floor(end_z + (i - 1) * connection_height)},
                point2 : OpenMesh::Vec3i{other_proj[0], other_proj[1], std::floor(end_z + i * connection_height)}
            };

            pairs.push_back(pair);
        }

        for (int i = 1; i <= connection_number; i++)
        {
            float connection_height = (start_z - end_z) / connection_number;
            auto pair = ConnectionPointPair{
                point1 : OpenMesh::Vec3i{other_proj[0], other_proj[1], std::floor(end_z + (i - 1) * connection_height)},
                point2 : OpenMesh::Vec3i{main_proj[0], main_proj[1], std::floor(end_z + i * connection_height)}
            };

            pairs.push_back(pair);
        }
    }

public:
    Connection(const Anchor &main) : isClosestValid{false}, isSecondClosestValid{false}
    {

        m_anchors.at(0) = main;
    }

    void setClosestPoint(const Anchor &closest)
    {

        this->isClosestValid = true;
        m_anchors.at(1) = closest;
    }

    void setSecondClosestPoint(const Anchor &second)
    {

        this->isSecondClosestValid = true;
        m_anchors.at(2) = second;
    }

    std::vector<ConnectionPointPair> createConnectionPoints(float pillairStability) const
    {
        std::vector<ConnectionPointPair> pairs;

        if (isClosestValid)
        {
            connectToMainAnchor(m_anchors[1], pairs, pillairStability);
        }
        if (isSecondClosestValid)
        {
            //connectToMainAnchor(m_anchors[2], pairs, pillairStability);
        }

        return pairs;
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
        imwrite(std::to_string(i) + "subtractResult1.jpg", substractResult);
#endif

        bitwise_and(substractTempResult, support, substractResult);

#ifdef DEBUG

        imwrite(std::to_string(i) + "subtractResult2.jpg", substractResult);
#endif

        bitwise_or(intersect, substractResult, intersect);

        subtract(support, substractResult, support);
#ifdef DEBUG

        imwrite(std::to_string(i) + "intersect.jpg", intersect);
        imwrite(std::to_string(i) + "dilation.jpg", dilation);
        imwrite(std::to_string(i) + "support.jpg", support);
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
cv::Mat GenerateAnchorMap(const cv::Mat &support_i, float anchorRadius, int layer_number, std::vector<OpenMesh::Vec3i> &anchorMap, int grid_res_x, int grid_res_y, int pillairRadius)
{

    cv::Mat anchor_point_image = cv::Mat::zeros(support_i.size(), support_i.type());
    cv::Mat support_copy = support_i.clone();

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

                anchorMap.push_back(OpenMesh::Vec3i{x, y, layer_number});
                cv::circle(anchor_point_image, anchorPoint, pillairRadius, cv::Scalar(255), -1);
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

                anchorMap.push_back(OpenMesh::Vec3i{intersection_begin + (x - intersection_begin), y, layer_number});
                cv::imwrite("before_draw.jpg", anchor_point_image);
                circle(anchor_point_image, anchorPoint, pillairRadius, cv::Scalar(255), -1);
                cv::imwrite("after_draw.jpg", anchor_point_image);
                sub = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);
                cv::imwrite("sub.jpg", sub);
                subtract(support_copy, sub, support_copy);
                cv::imwrite("support_copy.jpg", support_copy);
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

                anchorMap.push_back(OpenMesh::Vec3i{x, intersection_begin + (y - intersection_begin) / 2, layer_number});
                cv::imwrite("before_draw.jpg", anchor_point_image);
                circle(anchor_point_image, anchorPoint, pillairRadius, cv::Scalar(255), -1);
                cv::imwrite("after_draw.jpg", anchor_point_image);
                sub = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);
                cv::imwrite("sub.jpg", sub);
                subtract(support_copy, sub, support_copy);
                cv::imwrite("support_copy.jpg", support_copy);
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
                anchorMap.push_back(OpenMesh::Vec3i{x, y, layer_number});

                circle(anchor_point_image, anchorPoint, pillairRadius, cv::Scalar(255), -1);
                sub = GrowingSwallow(support_copy, anchor_point_image, anchor_point_image, anchorRadius);
                subtract(support_copy, sub, support_copy);
            }
        }
    }

    return anchor_point_image;
}

cv::Mat RegionSubtractionSLA(const cv::Mat &part_i, const cv::Mat &part_i_plus1, const cv::Mat &anchor_support_i_plus1, int layer_number, float selfSupportThreshold, float anchorRadius, std::vector<Anchor> &anchorList, int anchor_grid_res_x, int anchor_grid_res_y, int pillairRadius)
{

    cv::Mat anchorMap = cv::Mat::zeros(part_i.size(), part_i.type());
    cv::Mat shadow = cv::Mat::zeros(part_i.size(), part_i.type());
    cv::Mat PA_plus1 = cv::Mat::zeros(part_i.size(), part_i.type());

    subtract(part_i_plus1, part_i, shadow);
    subtract(anchor_support_i_plus1, part_i, PA_plus1);

    cv::Mat support_candidate = GrowingSwallow(shadow, part_i, part_i_plus1, selfSupportThreshold);
    cv::Mat support_candidate2 = GrowingSwallow(support_candidate, PA_plus1, PA_plus1, anchorRadius);

    std::vector<OpenMesh::Vec3i> anchorPointList;
    cv::Mat anchor_candidate = GenerateAnchorMap(support_candidate2, anchorRadius, layer_number, anchorPointList, anchor_grid_res_x, anchor_grid_res_y, pillairRadius);

    for (const auto &point : anchorPointList)
    {
        auto anchor_it = std::find_if(anchorList.begin(), anchorList.end(), [point](Anchor a) { return (a.start[0] == point[0] && a.start[1] == point[1] && a.end == OpenMesh::Vec3i{-1, -1, -1}); });

        if (anchor_it == anchorList.end())
        {

            Anchor new_anchor;
            new_anchor.start = point;
            new_anchor.end = OpenMesh::Vec3i{-1, -1, -1};

            anchorList.push_back(new_anchor);
        }
        else
        {
            anchor_it->end = point;
        }
    }

    bitwise_or(anchor_candidate, PA_plus1, anchorMap);

    return anchorMap;
}

static int selectedSlice = 0;

int main(int argc, char const *argv[])
{
    float anchorSupport = 10.0f;
    float selfSupport = 5.0f;

    int anchor_grid_res_x = 20;
    int anchor_grid_res_y = 20;
    int pillairRadius = 1;
    float pillair_stability = 10;

    if (argc == 9)
    {
        pillairRadius = std::stoi(argv[7]);
        anchorSupport = std::stof(argv[2]) + pillairRadius;
        selfSupport = std::stof(argv[1]);
        pillair_stability = std::stof(argv[8]);

        anchor_grid_res_x = std::stoi(argv[5]);
        anchor_grid_res_y = std::stoi(argv[6]);

        std::cout << "Parameters:" << std::endl;
        std::cout << "Self support: " << selfSupport << std::endl;
        std::cout << "Anchor support:" << anchorSupport << std::endl;
        std::cout << "Pillair radius:" << pillairRadius << std::endl;
        std::cout << "Anchor grid resolution: " << anchor_grid_res_x << " x " << anchor_grid_res_y << std::endl;

        LDNI ldni;
        std::chrono::steady_clock::time_point start, stop;

        size_t res = std::atoi(argv[4]);

        auto input_mesh = Geometry::TriMesh::readOBJ(argv[3]);
        std::cout << "File loaded." << std::endl;

        start = std::chrono::steady_clock::now();
        std::cout << res << std::endl;
        ldni = mesh2ldni(input_mesh, res);
        stop = std::chrono::steady_clock::now();
        std::cout << "LDNI generation: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << "ms" << std::endl;
        std::cout << "Resolution: "
                  << ldni.res[0] << "x" << ldni.res[1] << "x" << ldni.res[2] << std::endl;

        float filamentDiameter = 0.125f;

        int gridWidth = ldni.res[0];
        int gridHeight = ldni.res[1];
        int sliceNumber = ldni.res[2];

        cv::Mat slice[sliceNumber + 1];

        std::cout << "Resolution is: " << gridWidth << " x " << gridHeight << " x " << sliceNumber << std::endl;

        for (int z = 0; z < sliceNumber + 1; z++)
        {
            slice[z].create(gridWidth, gridHeight, CV_8UC(1));

            for (int x = 0; x < gridWidth; x++)
            {

                for (int y = 0; y < gridHeight; y++)
                {
                    slice[z].at<unsigned char>(x, y) = 255 * insidep(ldni, {x, y, z});
                }
            }
            //cv::imwrite( std::to_string(z) + "slice_z.jpg" , slice[z] );
        }

        cv::Mat support[sliceNumber + 1];
        PolyVox::SimpleVolume<uint8_t> volData(PolyVox::Region(PolyVox::Vector3DInt32(0, 0, 0), PolyVox::Vector3DInt32(gridWidth, gridHeight, sliceNumber)));

        std::vector<Anchor> anchorList;

        support[sliceNumber] = cv::Mat::zeros(slice[0].size(), slice[0].type());
        for (int z = sliceNumber - 1; z >= 0; z--)
        {
            std::cout << "Region subtraction on slice: " << z << std::endl;
            support[z].create(gridWidth, gridHeight, CV_8UC(1));
            support[z] = RegionSubtractionSLA(slice[z], slice[z + 1], support[z + 1], z, selfSupport, anchorSupport, anchorList, anchor_grid_res_x, anchor_grid_res_y, pillairRadius);
        }

        for (auto &anchor : anchorList)
        {

            if (anchor.end == OpenMesh::Vec3i{-1, -1, -1})
            {
                anchor.end = OpenMesh::Vec3i{anchor.start[0], anchor.start[1], 0};
            }
        }

        //determine which anchors should be connected
        std::vector<Connection> connections;

        for (const auto &anchor1 : anchorList)
        {
            Anchor closest, second_closest;
            //we say that the closest anchor is like if we put an anchor to the origo
            float closest_distance = MAXFLOAT;
            float second_closest_distance = closest_distance;

            int numberOfChange = 0;

            for (const auto &anchor2 : anchorList)
            {

                OpenMesh::Vec2i a1_proj = {anchor1.start[0], anchor1.start[1]};
                OpenMesh::Vec2i a2_proj = {anchor2.start[0], anchor2.start[1]};

                float distance = (a1_proj - a2_proj).length();

                int lower_start_z = std::min(anchor1.start[2], anchor2.start[2]);
                int higher_end_z = std::max(anchor1.end[2], anchor2.end[2]);

                if (anchor1 != anchor2 && a1_proj != a2_proj && distance < closest_distance && lower_start_z > higher_end_z)
                {
                    second_closest = closest;
                    closest = anchor2;
                    second_closest_distance = closest_distance;
                    closest_distance = distance;
                    numberOfChange++;
                }
            }

            Connection c{anchor1};

            if (numberOfChange >= 1)
                c.setClosestPoint(closest);
            if (numberOfChange >= 2)
                c.setSecondClosestPoint(second_closest);

            connections.push_back(c);
        }

        /*for (const auto &connection : connections)
        {
            auto pointPairs = connection.createConnectionPoints(pillair_stability);

            //fill voxel with pairs

            for (auto connectionPoints : pointPairs)
            {

                OpenMesh::Vec3i difference = (connectionPoints.point2 - connectionPoints.point1);

                float xz_inc = difference[0] / (float)difference[2];
                float yz_inc = difference[1] / (float)difference[2];
                float xy_inc = difference[0] / (float)difference[1];

                float x = connectionPoints.point1[0];
                float y = connectionPoints.point1[1];
                float z = connectionPoints.point1[2];

                for (int indexZ = connectionPoints.point1[2]; indexZ <= connectionPoints.point2[2]; indexZ++)
                {

                    int prevIndexX = (int)std::floor(x);
                    int prevIndexY = (int)std::floor(y);

                    volData.setVoxelAt(prevIndexX, prevIndexY, indexZ, 255);

                    x += xz_inc;
                    y += yz_inc;
                    int newIndexX = (int)std::floor(x);
                    int newIndexY = (int)std::floor(y);

                    for (int indexX = prevIndexX; indexX <= newIndexX && indexX >= 0 && newIndexY >=0 && indexZ >= 0; indexX++)
                    {
                        volData.setVoxelAt(indexX, newIndexY, indexZ, 255);
                    }

                    for (int indexX = prevIndexX; indexX >= newIndexX && indexX >= 0 && newIndexY >=0 && indexZ >= 0; indexX--)
                    {
                        volData.setVoxelAt(indexX, newIndexY, indexZ, 255);
                    }

                    for (int indexY = prevIndexY; indexY <= newIndexY && indexY >= 0 && newIndexX >=0 && indexZ >= 0; indexY++)
                    {
                        volData.setVoxelAt(newIndexX, indexY, indexZ, 255);
                    }

                    for (int indexY = prevIndexY; indexY >= newIndexY && indexY >= 0 && newIndexX >=0 && indexZ >= 0; indexY--)
                    {
                        volData.setVoxelAt(newIndexX, indexY, indexZ, 255);
                    }
                }
            }
        }*/

        for (int z = 0; z < sliceNumber; z++)
        {

            for (int x = 0; x < gridWidth; x++)
            {

                for (int y = 0; y < gridHeight; y++)
                {
                    if (/*support[z].at<uint8_t>(x, y) == 255 ||*/ slice[z].at<uint8_t>(x, y) == 255)
                        volData.setVoxelAt(x, y, z, 255);
                }
            }
        }

        std::cout << "Extracting surface." << std::endl;
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
    }
    else
    {
        std::cerr << "Usage:" << std::endl;
        std::cerr << "a) " << argv[0] << " <model.obj> <resolution>" << std::endl;
        std::cerr << "b) " << argv[0] << " <model.ldni>" << std::endl;
        return 1;
    }

    return 0;
}
