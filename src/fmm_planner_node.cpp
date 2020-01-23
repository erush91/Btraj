#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <fast_methods/gradientdescent/gradientdescent.hpp>
#include <fast_methods/ndgridmap/ndgridmap.hpp>
#include <fast_methods/fm/fmdata/fmcell.h>
#include <fast_methods/fm/fmm.hpp>
#include <fast_methods/fm/fmmstar.hpp>
#include <fast_methods/io/gridplotter.hpp>
#include <fast_methods/io/gridwriter.hpp>

#include "backward.hpp"

using namespace std;
using namespace Eigen;

namespace backward
{
    backward::SignalHandling sh;
}

// Typedefs
typedef nDGridMap<FMCell, 3> FMGrid3D;
typedef array<unsigned int, 3> Coord3D;
typedef typename std::vector< array<double, 3> > Path3D; 

// simulation param from launch file
double _vis_traj_width;
double _resolution, _inv_resolution;

// Global variables
nav_msgs::Odometry _odom;
bool _has_odom  = false;
bool _has_map   = false;
bool _has_target= false;
bool _has_traj  = false;

Vector3d _start_pt, _start_pt_rounding_eror;
Vector3d _map_origin;
double _x_size, _y_size, _z_size;
int _max_x_idx, _max_y_idx, _max_z_idx;

// ROS
ros::Subscriber _map_sub, _pts_sub, _odom_sub;
ros::Publisher _fm_path_vis_pub, _esdf_map_vis_pub, _fmm_map_vis_pub;

void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map);
void rcvOdometryCallbck(const nav_msgs::Odometry odom);
void visPath(vector<Vector3d> path);

void rcvOdometryCallbck(const nav_msgs::Odometry odom)
{
    _odom = odom;
    _has_odom = true;

    _start_pt(0)  = _odom.pose.pose.position.x;
    _start_pt(1)  = _odom.pose.pose.position.y;
    _start_pt(2)  = _odom.pose.pose.position.z;

    _start_pt_rounding_eror(0) = ((_start_pt(0) * _inv_resolution) - round(_start_pt(0) * _inv_resolution)) * _resolution;
    _start_pt_rounding_eror(1) = ((_start_pt(1) * _inv_resolution) - round(_start_pt(1) * _inv_resolution)) * _resolution;
    _start_pt_rounding_eror(2) = ((_start_pt(2) * _inv_resolution) - round(_start_pt(2) * _inv_resolution)) * _resolution;

    std::cout << "_start_pt_rounding_eror" << _start_pt_rounding_eror(0) << ", " << _start_pt_rounding_eror(1) << ", " << _start_pt_rounding_eror(2) << std::endl;

}

void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map)
{

    pcl::PointCloud<pcl::PointXYZI> cloud;  
    pcl::fromROSMsg(pointcloud_map, cloud);

    std::cout << "_start_pt: " << _start_pt(0) << ", " << _start_pt(1) << ", " << _start_pt(2) << std::endl;

    std::string frame;
    
    unsigned int x_size = (unsigned int)(_max_x_idx);
    unsigned int y_size = (unsigned int)(_max_y_idx);
    unsigned int z_size = (unsigned int)(_max_z_idx);

    ////////////////////
    // LOCAL ESDF MAP //
    ////////////////////
    
    Coord3D dimsize {x_size, y_size, z_size};
    FMGrid3D grid_fmm_3d(dimsize);
    
    grid_fmm_3d.clear();

    vector<unsigned int> obs;


    for(unsigned long int grid_idx = 0; grid_idx < x_size * y_size * z_size; grid_idx++)
    {
        grid_fmm_3d[grid_idx].setOccupancy(-1.0);
    }

    // Assign the ESDF
    for (unsigned long int pcl_idx = 0; pcl_idx < cloud.points.size(); pcl_idx++)
    {
        pcl::PointXYZI mk = cloud.points[pcl_idx];
        unsigned int  x_idx = ((mk.x - _start_pt(0) - _map_origin(0)) * _inv_resolution);
        unsigned int  y_idx = ((mk.y - _start_pt(1) - _map_origin(1)) * _inv_resolution);
        unsigned int  z_idx = ((mk.z - _start_pt(2) - _map_origin(2)) * _inv_resolution);
        unsigned long int grid_idx = x_idx + y_idx * x_size + z_idx * x_size * y_size;
        grid_fmm_3d[grid_idx].setOccupancy(mk.intensity / 5.0);
    }

    for(unsigned long int grid_idx = 0; grid_idx < x_size*y_size*z_size; grid_idx++)
    {
        if (grid_fmm_3d[grid_idx].isOccupied())
        {
            obs.push_back(grid_idx);
        }
    }

    grid_fmm_3d.setOccupiedCells(std::move(obs));
    grid_fmm_3d.setLeafSize(_resolution);

    GridWriter::saveGridValues("test_fm3d.txt", grid_fmm_3d);


    //////////////////////////////////
    // LOCAL FMM MAP (ARRIVAL TIME) //
    //////////////////////////////////

    Vector3d endIdx3d   = - _map_origin * _inv_resolution;

    Coord3D init_point = {(unsigned int)round(endIdx3d[0]), (unsigned int)round(endIdx3d[1]), (unsigned int)round(endIdx3d[2])};

    unsigned int startIdx;
    vector<unsigned int> startIndices;
    grid_fmm_3d.coord2idx(init_point, startIdx);

    startIndices.push_back(startIdx);

    Solver<FMGrid3D>* fm_solver = new FMMStar<FMGrid3D>("FMM*_Dist", TIME); // LSM, FMM

    fm_solver->setEnvironment(&grid_fmm_3d);
    fm_solver->setInitialPoints(startIndices);
    fm_solver->compute(1.0);

    ////////////////////
    // LOCAL ESDF PCL //
    ////////////////////

    pcl::PointCloud<pcl::PointXYZI> cloud_esdf;
    cloud_esdf.height = 1;
    cloud_esdf.is_dense = true;
    cloud_esdf.header.frame_id = "odom";

    long int cnt = 0;
    for(unsigned long int grid_idx = 0; grid_idx < x_size * y_size * z_size; grid_idx++)
    {
        int z_idx =  grid_idx / (x_size * y_size);
        int y_idx = (grid_idx - z_idx * x_size * y_size) / x_size;
        int x_idx =  grid_idx - z_idx * x_size * y_size - y_idx * x_size;
        pcl::PointXYZI esdf_pt;
        esdf_pt.x = (x_idx + 0.5) * _resolution + _start_pt(0) - _start_pt_rounding_eror(0) + _map_origin(0);
        esdf_pt.y = (y_idx + 0.5) * _resolution + _start_pt(1) - _start_pt_rounding_eror(1) + _map_origin(1);
        esdf_pt.z = (z_idx + 0.5) * _resolution + _start_pt(2) - _start_pt_rounding_eror(2) + _map_origin(2);
        esdf_pt.intensity = grid_fmm_3d[grid_idx].getOccupancy();
        if(esdf_pt.intensity > 0)
        {
            cnt++;
            cloud_esdf.push_back(esdf_pt);
        }
    }

    cloud_esdf.width = cnt;

    // DEBUGGING
    // std::cout << "cloud_esdf cnt: " << cnt << std::endl;
    // std::cout << "cloud_esdf.points.size: " << cloud_esdf.points.size() << std::endl;

    sensor_msgs::PointCloud2 esdfMap;
    pcl::toROSMsg(cloud_esdf, esdfMap);
    _esdf_map_vis_pub.publish(esdfMap);


    ///////////////////
    // LOCAL FMM PCL //
    ///////////////////

    pcl::PointCloud<pcl::PointXYZI> cloud_fmm;
    cloud_fmm.height = 1;
    cloud_fmm.is_dense = true;
    cloud_fmm.header.frame_id = "odom";

    cnt = 0;
    for(unsigned long int grid_idx = 0; grid_idx < x_size * y_size * z_size; grid_idx++)
    {
        int z_idx =  grid_idx / (x_size * y_size);
        int y_idx = (grid_idx - z_idx * x_size * y_size) / x_size;
        int x_idx =  grid_idx - z_idx * x_size * y_size - y_idx * x_size;
        pcl::PointXYZI fmm_pt;
        fmm_pt.x = (x_idx + 0.5) * _resolution + _start_pt(0) - _start_pt_rounding_eror(0) + _map_origin(0);
        fmm_pt.y = (y_idx + 0.5) * _resolution + _start_pt(1) - _start_pt_rounding_eror(1) + _map_origin(1);
        fmm_pt.z = (z_idx + 0.5) * _resolution + _start_pt(2) - _start_pt_rounding_eror(2) + _map_origin(2);
        fmm_pt.intensity = grid_fmm_3d[grid_idx].getOccupancy();
        if(fmm_pt.intensity >= 0 && fmm_pt.intensity < 99999)
        {
            cnt++;
            cloud_fmm.push_back(fmm_pt);
        }
    }

    cloud_fmm.width = cnt;

    // DEBUGGING
    // std::cout << "cloud_fmm cnt: " << cnt << std::endl;
    // std::cout << "cloud_fmm.points.size: " << cloud_fmm.points.size() << std::endl;

    sensor_msgs::PointCloud2 fmmMap;
    pcl::toROSMsg(cloud_fmm, fmmMap);
    _fmm_map_vis_pub.publish(fmmMap);






    /////////////////////////////
    // LOCAL PATH (FMM SOLVER) //
    /////////////////////////////

    Vector3d goal_offset = {3, 0, 0};
    Vector3d offset = {3, 0, 0};
    Vector3d startIdx3d = (goal_offset - _map_origin) * _inv_resolution - offset;
    Coord3D goal_point = {(unsigned int)round(startIdx3d[0]), (unsigned int)round(startIdx3d[1]), (unsigned int)round(startIdx3d[2])};
    unsigned int goalIdx;
    grid_fmm_3d.coord2idx(goal_point, goalIdx);

    Path3D path3D;
    vector<double> path_vels, time;
    GradientDescent< FMGrid3D > grad3D;
    grid_fmm_3d.coord2idx(goal_point, goalIdx);

    vector<Vector3d> path_coord;
    int cnt2 = 0;

    if(grad3D.gradient_descent(grid_fmm_3d, goalIdx, path3D, path_vels, time) == -1)
    {
        std::cout << "GRADIENT DESCENT FMM ERROR" << std::endl;
    }
    else
    {
        std::cout << "Path3D is :" << std::endl;
        for (auto pt: path3D)
        {
            for (int elem : pt)
            {
                std::cout << elem << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // path_coord.push_back(_start_pt);

        double coord_x, coord_y, coord_z;
  
        for(int path_idx = (int)path3D.size() - 1; path_idx >= 0; path_idx--)
        {
            cnt2++;
            coord_x = path3D[path_idx][0] * _resolution + _start_pt(0) + _map_origin(0);
            coord_y = path3D[path_idx][1] * _resolution + _start_pt(1) + _map_origin(1);
            coord_z = path3D[path_idx][2] * _resolution + _start_pt(2) + _map_origin(2);

            Vector3d pt(coord_x, coord_y, coord_z);
            path_coord.push_back(pt);
            std::cout << "path[" << path_idx << "]: " << pt(0) - _start_pt(0) << ", " << pt(1) - _start_pt(1) << ", " << pt(2) - _start_pt(2) << std::endl;
        }
        visPath(path_coord);


    }

    std::cout << "path3D.size(): " << path3D.size() << std::endl;
    std::cout << "cnt2: " << cnt2 << std::endl;


    /////////////////////////
    // GOAL POINT SELCTION //
    /////////////////////////

    for(int path_idx = 0; path_idx < int(path_coord.size()); path_idx++)
    {
        Vector3d path_pt = path_coord[path_idx] - _start_pt;

        std::cout << "_check_pt: " << path_pt(0) << ", " << path_pt(1) << ", " << path_pt(2) << std::endl;

        float path_pt_distance = sqrt(path_pt(0)* path_pt(0) + path_pt(1)*path_pt(1) + path_pt(2)*path_pt(2));
        
        std::cout << "path_pt_distance: " << path_pt_distance << std::endl;

        Vector3d path_dir = path_pt / path_pt_distance;

        std::cout << "path_dir: " << path_dir(0) << ", " << path_dir(1) << ", " << path_dir(2) << std::endl;

        int num_interpolated_pts_to_check = ceil(2 * _inv_resolution * (float)path_pt_distance);

        std::cout << "num_interpolated_pts_to_check: " << num_interpolated_pts_to_check << std::endl;

        for(int interp_idx = 1; interp_idx < num_interpolated_pts_to_check + 1; interp_idx++)
        {
            Vector3d _check_pt = path_pt / num_interpolated_pts_to_check * interp_idx ;
            std::cout << "_check_pt: " << _check_pt(0) << ", " << _check_pt(1) << ", " << _check_pt(2) << std::endl;
        }
            







            
        Vector3d goal_pt = path_coord[path_idx] - _start_pt - _map_origin;
        std::cout << "goal_pt[" << path_idx << "]:" << goal_pt(0) << ", " << goal_pt(1) << ", " << goal_pt(2) << std::endl;
        
        Vector3d goalPtIdx3d = goal_pt * _inv_resolution;
        
        Coord3D goal_point = {(unsigned int)round(goalPtIdx3d[0]), (unsigned int)round(goalPtIdx3d[1]), (unsigned int)round(goalPtIdx3d[2])};
        std::cout << "goalPtIdx3d[" << path_idx << "]:" << goal_point[0] << ", " << goal_point[1] << ", " << goal_point[2] << std::endl;

        unsigned int pathPtIdx;
        grid_fmm_3d.coord2idx(goal_point, pathPtIdx);
        float occ = grid_fmm_3d[pathPtIdx].getOccupancy();
        std::cout << "occ[" << path_idx << "]:" << occ << std::endl;
        std::cout << std::endl;

    }

    std::cout << std::endl;
    std::cout << std::endl;
    
    // Vector3d test_pt_offset = {0, 4, 0};
    // Vector3d test_pt =  path_coord[0] + test_pt_offset - _start_pt - _map_origin;
    // std::cout << "test_pt" << test_pt(0) << ", " << test_pt(1) << ", " << test_pt(2) << std::endl;
    
    // Vector3d testPtIdx3d = test_pt * _inv_resolution;
    
    // Coord3D test_point = {(unsigned int)testPtIdx3d[0], (unsigned int)testPtIdx3d[1], (unsigned int)testPtIdx3d[2]};
    // std::cout << "testPtIdx3d:" << test_point[0] << ", " << test_point[1] << ", " << test_point[2] << std::endl;

    // unsigned int testPtIdx;
    // grid_fmm_3d.coord2idx(test_point, testPtIdx);
    // float occ_test = grid_fmm_3d[testPtIdx].getOccupancy();
    // std::cout << "occ_test:" << occ_test << std::endl;










}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "b_traj_node");
    ros::NodeHandle nh("~");

    _map_sub  = nh.subscribe( "map",       1, rcvPointCloudCallBack );
    _odom_sub = nh.subscribe( "odometry",  1, rcvOdometryCallbck);

    _esdf_map_vis_pub   = nh.advertise<sensor_msgs::PointCloud2>("map/esdf", 1);
    _fmm_map_vis_pub    = nh.advertise<sensor_msgs::PointCloud2>("map/fmm_arrival_time", 1);
    _fm_path_vis_pub    = nh.advertise<visualization_msgs::MarkerArray>("goal/path/viz", 1);

    nh.param("map/resolution",  _resolution, 0.2);
    nh.param("map/x_size",      _x_size, 50.0);
    nh.param("map/y_size",      _y_size, 50.0);
    nh.param("map/z_size",      _z_size, 50.0);

    // Origin is located in the middle of the map
    _map_origin = {- _x_size / 2.0, - _y_size / 2.0 , - _z_size / 2.0};

    // Inverse resolution
    _inv_resolution = 1.0 / _resolution;

    // This is the maximum indeces in the map
    _max_x_idx = (int)(_x_size * _inv_resolution) ;
    _max_y_idx = (int)(_y_size * _inv_resolution) ;
    _max_z_idx = (int)(_z_size * _inv_resolution) ;

    ros::Rate rate(100);
    bool status = ros::ok();
    while(status)
    {
        ros::spinOnce();           
        status = ros::ok();
        rate.sleep();
    }

    return 0;
}

visualization_msgs::MarkerArray path_vis;
void visPath(vector<Vector3d> path)
{
    for(auto & mk: path_vis.markers)
        mk.action = visualization_msgs::Marker::DELETE;

    _fm_path_vis_pub.publish(path_vis);
    path_vis.markers.clear();

    visualization_msgs::Marker mk;
    mk.header.frame_id = "odom";
    mk.header.stamp = ros::Time::now();
    mk.ns = "b_traj/fast_marching_path";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.a = 0.6;
    mk.color.r = 1.0;
    mk.color.g = 0.647;
    mk.color.b = 0.0;

    for(int path_idx = 0; path_idx < int(path.size()); path_idx++)
    {
        mk.id = path_idx;

        mk.pose.position.x = path[path_idx](0);
        mk.pose.position.y = path[path_idx](1);
        mk.pose.position.z = path[path_idx](2);

        mk.scale.x = _resolution;
        mk.scale.y = _resolution;
        mk.scale.z = _resolution;

        path_vis.markers.push_back(mk);
    }

    _fm_path_vis_pub.publish(path_vis);
}
