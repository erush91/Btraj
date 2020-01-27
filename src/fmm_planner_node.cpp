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

#include <functional> // For multiplying vector by scalar

#define _USE_MATH_DEFINES
#include <cmath> // For quaternion to Euler angle conversion

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
Vector3d _desired_pt;
Vector3d max_reward_pt;
Vector3d _map_origin;
double _x_size, _y_size, _z_size;
double _look_ahead_distance;
int _max_x_idx, _max_y_idx, _max_z_idx;
unsigned long int _max_grid_idx;
int x_size, y_size, z_size;

// ROS
ros::Subscriber _map_sub, _pts_sub, _odom_sub;
ros::Publisher _fm_path_vis_pub;
ros::Publisher _map_fmm_vel_vis_pub;
ros::Publisher _map_fmm_time_vis_pub;
ros::Publisher _map_fmm_reward_vis_pub;
ros::Publisher _line_of_sight_path_vis_pub;
ros::Publisher _line_of_sight_vector_vis_pub;
ros::Publisher _collision_vector_vis_pub;
ros::Publisher _goal_point_uncropped_pub;
ros::Publisher _goal_pose_uncropped_pub;
ros::Publisher _goal_point_pub;
ros::Publisher _goal_pose_pub;
ros::Publisher _max_reward_point_pub;
ros::Publisher _max_reward_pose_pub;

visualization_msgs::MarkerArray line_of_sight_vectors_msg;
    
void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map);
void rcvOdometryCallbck(const nav_msgs::Odometry odom);
void publishVisPath(vector<Vector3d> path, ros::Publisher _path_vis_pub);
double velMapping(double d);

vector<long int> ptPcl2idx(pcl::PointXYZI pt);
vector<long int> ptVect2idx(vector<float> pt);
long int ptVect2gridIdx(vector<float> pt);
pcl::PointXYZI idx2pt(long int grid_idx);
vector<int> grid2xyz(long int grid_idx);
pcl::PointXYZI xyz2ptPCl(vector<int> xyz_idx);
Vector3d xyz2ptVect(vector<int> xyz_idx);

int main(int argc, char** argv)
{
    ros::init(argc, argv, "b_traj_node");
    ros::NodeHandle nh("~");

    _map_sub  = nh.subscribe( "map",       1, rcvPointCloudCallBack );
    _odom_sub = nh.subscribe( "odometry",  1, rcvOdometryCallbck);

    _map_fmm_vel_vis_pub            = nh.advertise<sensor_msgs::PointCloud2>("viz/map/fmm/velocity", 1);
    _map_fmm_time_vis_pub           = nh.advertise<sensor_msgs::PointCloud2>("viz/map/fmm/arrival_time", 1);
    _map_fmm_reward_vis_pub         = nh.advertise<sensor_msgs::PointCloud2>("viz/map/fmm/reward", 1);
    _fm_path_vis_pub                = nh.advertise<visualization_msgs::MarkerArray>("viz/goal/fmm_path", 1);
    _line_of_sight_path_vis_pub     = nh.advertise<visualization_msgs::MarkerArray>("viz/goal/line_of_sight_path", 1);
    _line_of_sight_vector_vis_pub   = nh.advertise<visualization_msgs::MarkerArray>("viz/goal/line_of_sight/vectors", 1);
    _collision_vector_vis_pub       = nh.advertise<visualization_msgs::MarkerArray>("viz/goal/collision/vectors", 1);
    _goal_pose_uncropped_pub        = nh.advertise<geometry_msgs::PoseStamped>("viz/goal/pose_uncropped", 1);
    _goal_point_uncropped_pub       = nh.advertise<geometry_msgs::PointStamped>("viz/goal/point_uncropped", 1);
    _goal_pose_pub                  = nh.advertise<geometry_msgs::PoseStamped>("out/goal/pose", 1);
    _goal_point_pub                 = nh.advertise<geometry_msgs::PointStamped>("out/goal/point", 1);
    _max_reward_pose_pub            = nh.advertise<geometry_msgs::PoseStamped>("out/goal/max_reward_pose", 1);
    _max_reward_point_pub           = nh.advertise<geometry_msgs::PointStamped>("out/goal/max_reward_point", 1);

    nh.param("map/resolution",              _resolution, 0.2);
    nh.param("map/x_size",                  _x_size, 50.0);
    nh.param("map/y_size",                  _y_size, 50.0);
    nh.param("map/z_size",                  _z_size, 50.0);
    nh.param("goal/look_ahead_distance",   _look_ahead_distance, 2.5);

    // Origin is located in the middle of the map
    _map_origin = {- _x_size / 2.0, - _y_size / 2.0 , - _z_size / 2.0};

    // Inverse resolution
    _inv_resolution = 1.0 / _resolution;

    // This is the maximum indeces in the map
    _max_x_idx = (int)(_x_size * _inv_resolution);
    _max_y_idx = (int)(_y_size * _inv_resolution);
    _max_z_idx = (int)(_z_size * _inv_resolution);
    _max_grid_idx = _max_x_idx * _max_y_idx * _max_z_idx;

    x_size = (int)(_max_x_idx);
    y_size = (int)(_max_y_idx);
    z_size = (int)(_max_z_idx);

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

    tf::Quaternion quat(_odom.pose.pose.orientation.x,
                        _odom.pose.pose.orientation.y,
                        _odom.pose.pose.orientation.z,
                        _odom.pose.pose.orientation.w);
        
    tf::Matrix3x3 m_odom(quat);
    double roll, pitch, yaw;
    m_odom.getRPY(roll, pitch, yaw);

    // DEBUGGING
    // std::cout << "yaw: " << yaw << std::endl;
    
    _desired_pt(0)  = _start_pt(0) + 5*cos(-yaw);
    _desired_pt(1)  = _start_pt(1) - 5*sin(-yaw);
    _desired_pt(2)  = _start_pt(2);


    // DEBUGGING
    // std::cout << "_start_pt_rounding_eror" << _start_pt_rounding_eror(0) << ", " << _start_pt_rounding_eror(1) << ", " << _start_pt_rounding_eror(2) << std::endl;
}
vector<long int> pt2idx(pcl::PointXYZI pt)
{
    long int  x_idx = ((pt.x - _start_pt(0) - _map_origin(0)) * _inv_resolution);
    long int  y_idx = ((pt.y - _start_pt(1) - _map_origin(1)) * _inv_resolution);
    long int  z_idx = ((pt.z - _start_pt(2) - _map_origin(2)) * _inv_resolution);
    long int grid_idx = x_idx + y_idx * x_size + z_idx * x_size * y_size;
    vector<long int> idx = {x_idx, y_idx, z_idx, grid_idx};
    return idx;
}

vector<long int> ptPcl2idx(pcl::PointXYZI pt)
{
    long int  x_idx = ((pt.x - _start_pt(0) - _map_origin(0)) * _inv_resolution);
    long int  y_idx = ((pt.y - _start_pt(1) - _map_origin(1)) * _inv_resolution);
    long int  z_idx = ((pt.z - _start_pt(2) - _map_origin(2)) * _inv_resolution);
    long int grid_idx = x_idx + y_idx * x_size + z_idx * x_size * y_size;
    vector<long int> idx = {x_idx, y_idx, z_idx, grid_idx};
    return idx;
}

vector<long int> ptVect2idx(vector<float> pt)
{
    long int  x_idx = ((pt[0] - _start_pt(0) - _map_origin(0)) * _inv_resolution);
    long int  y_idx = ((pt[1] - _start_pt(1) - _map_origin(1)) * _inv_resolution);
    long int  z_idx = ((pt[2] - _start_pt(2) - _map_origin(2)) * _inv_resolution);
    long int grid_idx = x_idx + y_idx * x_size + z_idx * x_size * y_size;
    vector<long int> idx = {x_idx, y_idx, z_idx, grid_idx};
    return idx;
}

long int ptVect2gridIdx(vector<float> pt)
{
    long int  x_idx = ((pt[0] - _start_pt(0) - _map_origin(0)) * _inv_resolution);
    long int  y_idx = ((pt[1] - _start_pt(1) - _map_origin(1)) * _inv_resolution);
    long int  z_idx = ((pt[2] - _start_pt(2) - _map_origin(2)) * _inv_resolution);
    long int grid_idx = x_idx + y_idx * x_size + z_idx * x_size * y_size;
    return grid_idx;
}

pcl::PointXYZI idx2pt(long int grid_idx)
{
    vector<int> xyz_idx = grid2xyz(grid_idx);
    pcl::PointXYZI pt = xyz2ptPCl(xyz_idx);
    return pt;
}

vector<int> grid2xyz(long int grid_idx)
{
    int z_idx =  grid_idx / (x_size * y_size);
    int y_idx = (grid_idx - z_idx * x_size * y_size) / x_size;
    int x_idx =  grid_idx - z_idx * x_size * y_size - y_idx * x_size;
    vector<int> xyz_int = {x_idx, y_idx, z_idx};
    return xyz_int;
}


pcl::PointXYZI xyz2ptPCl(vector<int> xyz_idx)
{
    pcl::PointXYZI pt;
    pt.x = (xyz_idx[0] + 0.5) * _resolution + _start_pt(0) - _start_pt_rounding_eror(0) + _map_origin(0);
    pt.y = (xyz_idx[1] + 0.5) * _resolution + _start_pt(1) - _start_pt_rounding_eror(1) + _map_origin(1);
    pt.z = (xyz_idx[2] + 0.5) * _resolution + _start_pt(2) - _start_pt_rounding_eror(2) + _map_origin(2);
    return pt;
}

Vector3d xyz2ptVect(vector<int> xyz_idx)
{
    Vector3d pt;
    pt(0) = (xyz_idx[0] + 0.5) * _resolution + _start_pt(0) - _start_pt_rounding_eror(0) + _map_origin(0);
    pt(1) = (xyz_idx[1] + 0.5) * _resolution + _start_pt(1) - _start_pt_rounding_eror(1) + _map_origin(1);
    pt(2) = (xyz_idx[2] + 0.5) * _resolution + _start_pt(2) - _start_pt_rounding_eror(2) + _map_origin(2);
    return pt;
}

void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map)
{

    pcl::PointCloud<pcl::PointXYZI> cloud;  
    pcl::fromROSMsg(pointcloud_map, cloud);

    // DEBUGGING
    // std::cout << "_start_pt: " << _start_pt(0) << ", " << _start_pt(1) << ", " << _start_pt(2) << std::endl;
    
    //////////////////////////////
    // LOCAL ESDF MAP (DISTANCE //
    //////////////////////////////
    
    Coord3D dimsize {x_size, y_size, z_size};
    FMGrid3D grid_fmm_3d(dimsize);
    
    grid_fmm_3d.clear();

    vector<unsigned int> obs;

    for(long int grid_idx = 0; grid_idx < _max_grid_idx; grid_idx++)
    {
        grid_fmm_3d[grid_idx].setOccupancy(0.0);
    }

    // Assign the ESDF
    for (unsigned long int pcl_idx = 0; pcl_idx < cloud.points.size(); pcl_idx++)
    {
        pcl::PointXYZI fmm_vel_pt = cloud.points[pcl_idx];
        vector<long int> fmm_vel_idx = ptPcl2idx(fmm_vel_pt);

        if(fmm_vel_idx[0] >= 0 && fmm_vel_idx[0] < _max_x_idx)
        {
            if(fmm_vel_idx[1] >= 0 && fmm_vel_idx[1] < _max_y_idx)
            {
                if(fmm_vel_idx[2] >= 0 && fmm_vel_idx[2] < _max_z_idx)
                {
                    grid_fmm_3d[fmm_vel_idx[3]].setOccupancy(velMapping(cloud.points[pcl_idx].intensity));
                }
            }
        }
    }

    long int cntt = 0;
    for(unsigned long int grid_idx = 0; grid_idx < _max_grid_idx; grid_idx++)
    {
        if (grid_fmm_3d[grid_idx].isOccupied())
        {
            cntt++;
            obs.push_back(grid_idx);
        }
    }
    grid_fmm_3d.setOccupiedCells(std::move(obs));
    grid_fmm_3d.setLeafSize(_resolution);

    std::cout << cntt << std::endl;

    ////////////////////
    // LOCAL ESDF PCL //
    ////////////////////

    pcl::PointCloud<pcl::PointXYZI> cloud_fmm_vel;
    cloud_fmm_vel.height = 1;
    cloud_fmm_vel.is_dense = true;
    cloud_fmm_vel.header.frame_id = "odom";

    long int cnt = 0;
    for(long int grid_idx = 0; grid_idx < _max_grid_idx; grid_idx++)
    {
        pcl::PointXYZI fmm_vel_pt = idx2pt(grid_idx);
        fmm_vel_pt.intensity = grid_fmm_3d[grid_idx].getOccupancy();
        if(fmm_vel_pt.intensity > 0)
        {
            cnt++;
            cloud_fmm_vel.push_back(fmm_vel_pt);
        }
    }

    cloud_fmm_vel.width = cnt;

    // DEBUGGING
    // std::cout << "cloud_fmm_vel cnt: " << cnt << std::endl;
    // std::cout << "cloud_fmm_vel.points.size: " << cloud_fmm_vel.points.size() << std::endl;

    sensor_msgs::PointCloud2 velMap;
    pcl::toROSMsg(cloud_fmm_vel, velMap);
    _map_fmm_vel_vis_pub.publish(velMap);

    //////////////////////////////////
    // LOCAL FMM MAP (ARRIVAL TIME) //
    //////////////////////////////////
    
    vector<float> fmm_init_pt = {_start_pt(0), _start_pt(1), _start_pt(2)};
    vector<long int> fmm_init_idx = ptVect2idx(fmm_init_pt);

    std::cout << "grid_fmm_3d[fmm_init_idx[3]].isOccupied(): " << grid_fmm_3d[fmm_init_idx[3]].isOccupied() << std::endl;
    if(grid_fmm_3d[fmm_init_idx[3]].isOccupied())
    {
        fmm_init_pt = {_start_pt(0) + _resolution, _start_pt(1), _start_pt(2)};
        vector<long int> fmm_init_idx = ptVect2idx(fmm_init_pt);
        std::cout << "grid_fmm_3d[fmm_init_idx[3]].isOccupied() inside if: " << grid_fmm_3d[fmm_init_idx[3]].isOccupied() << std::endl;
    }

    vector<unsigned int> startIndices;
    startIndices.push_back(fmm_init_idx[3]);



    std::cout << "start pt occ: " << grid_fmm_3d[fmm_init_idx[3]].getOccupancy() << std::endl;

    Solver<FMGrid3D>* fm_solver = new FMMStar<FMGrid3D>("FMM*_Dist", TIME); // LSM, FMM

    fm_solver->setEnvironment(&grid_fmm_3d);
    fm_solver->setInitialPoints(startIndices);
    fm_solver->setup();
    fm_solver->compute(1.0);

    // Preventing memory leaks.
    delete fm_solver;

    ///////////////////
    // LOCAL FMM PCL //
    ///////////////////

    pcl::PointCloud<pcl::PointXYZI> cloud_fmm_time;
    cloud_fmm_time.height = 1;
    cloud_fmm_time.is_dense = true;
    cloud_fmm_time.header.frame_id = "odom";

    cnt = 0;
    for(long int grid_idx = 0; grid_idx < _max_grid_idx; grid_idx++)
    {
        pcl::PointXYZI fmm_time_pt = idx2pt(grid_idx);
        fmm_time_pt.intensity = grid_fmm_3d[grid_idx].getArrivalTime(); // ARRIVAL TIME ()
        if(fmm_time_pt.intensity > 0 && fmm_time_pt.intensity < 99999)
        {
            cnt++;
            cloud_fmm_time.push_back(fmm_time_pt);
        }
    }

    cloud_fmm_time.width = cnt;

    // DEBUGGING
    // std::cout << "cloud_fmm_time cnt: " << cnt << std::endl;
    // std::cout << "cloud_fmm_time.points.size: " << cloud_fmm_time.points.size() << std::endl;

    sensor_msgs::PointCloud2 timeMap;
    pcl::toROSMsg(cloud_fmm_time, timeMap);
    _map_fmm_time_vis_pub.publish(timeMap);

    //////////////////////
    // LOCAL REWARD PCL //
    //////////////////////

    pcl::PointCloud<pcl::PointXYZI> cloud_fmm_reward;
    cloud_fmm_reward.height = 1;
    cloud_fmm_reward.is_dense = true;
    cloud_fmm_reward.header.frame_id = "odom";

    float reward;
    cnt = 0;
    long int min_idx = 0;
    float min_dist = 10.0;

    for(long int grid_idx = 0; grid_idx < _max_grid_idx; grid_idx++)
    {
        // Compute distance to desired point
        pcl::PointXYZI fmm_reward_pt = idx2pt(grid_idx);
        reward = sqrt((fmm_reward_pt.x - _desired_pt(0))*(fmm_reward_pt.x - _desired_pt(0))
                    + (fmm_reward_pt.y - _desired_pt(1))*(fmm_reward_pt.y - _desired_pt(1))); // DISTANCE FROM GOAL POINT IN FRONHT OF ROBOT
                    
        fmm_reward_pt.intensity = reward;

        // Push traversable points to pcl
        if(grid_fmm_3d[grid_idx].getArrivalTime() > 0 && grid_fmm_3d[grid_idx].getArrivalTime() < 99999)
        {
            cnt++;
            cloud_fmm_reward.push_back(fmm_reward_pt);

            // Find point farthest away
            if (reward < min_dist)
            {
                min_idx = cnt - 1;
                min_dist = reward;
            }
            
        }


    }
    cloud_fmm_reward.width = cnt;

    // DEBUGGING
    // std::cout << "cloud_fmm_reward cnt: " << cnt << std::endl;
    // std::cout << "cloud_fmm_reward.points.size: " << cloud_fmm_time.points.size() << std::endl;

    sensor_msgs::PointCloud2 rewardMap;
    pcl::toROSMsg(cloud_fmm_reward, rewardMap);
    _map_fmm_reward_vis_pub.publish(rewardMap);

    std::cout << "min_idx: " << min_idx << std::endl;

    //////////////////////////
    // GOAL POINT SELECTION //
    //////////////////////////


    pcl::PointXYZI fmm_max_reward_pt;
    vector<long int> fmm_max_reward_idx;

    if(cloud_fmm_reward.width > 0)
    {
        fmm_max_reward_pt = cloud_fmm_reward.points[min_idx];
                                  
        // DEBUGGING
        // std::cout << "inside if: cloud_fmm_reward.width: " << cloud_fmm_reward.width << std::endl;
        // std::cout << "inside if: min_idx: " << min_idx << std::endl;
        // std::cout << "inside if: min_dist: " << min_dist << std::endl;
        // std::cout << "inside if: cloud_fmm_reward.points[min_idx].x: " << cloud_fmm_reward.points[min_idx].x << std::endl;
        // std::cout << "inside if: cloud_fmm_reward.points[min_idx].x: " << cloud_fmm_reward.points[min_idx].x << std::endl;
        // std::cout << "inside if: cloud_fmm_reward.points[min_idx].y: " << cloud_fmm_reward.points[min_idx].y << std::endl;
        // std::cout << "inside if: cloud_fmm_reward.points[min_idx].z: " << cloud_fmm_reward.points[min_idx].z << std::endl;

        geometry_msgs::PointStamped max_reward_point;
        max_reward_point.header.stamp = ros::Time::now();
        max_reward_point.header.frame_id = "odom";
        max_reward_point.point.x = fmm_max_reward_pt.x;
        max_reward_point.point.y = fmm_max_reward_pt.y;
        max_reward_point.point.z = fmm_max_reward_pt.z;
            
        geometry_msgs::PoseStamped max_reward_pose;
        max_reward_pose.header.stamp = ros::Time::now();
        max_reward_pose.header.frame_id = "odom";
        max_reward_pose.pose.position.x = fmm_max_reward_pt.x;
        max_reward_pose.pose.position.y = fmm_max_reward_pt.y;
        max_reward_pose.pose.position.z = fmm_max_reward_pt.z;
        max_reward_pose.pose.orientation.x = 0.0;
        max_reward_pose.pose.orientation.y = 0.0;
        max_reward_pose.pose.orientation.z = 0.0;
        max_reward_pose.pose.orientation.w = 1.0;

        _max_reward_point_pub.publish(max_reward_point);
        _max_reward_pose_pub.publish(max_reward_pose);
    }
    else
    {
        fmm_max_reward_pt.x = _start_pt(0);
        fmm_max_reward_pt.x = _start_pt(1);
        fmm_max_reward_pt.x = _start_pt(2);
    }
    fmm_max_reward_idx = pt2idx(fmm_max_reward_pt);

    /////////////////////////////
    // LOCAL PATH (FMM SOLVER) //
    /////////////////////////////

    // DEBUGGING: FIXED GOAL POINT TESTING
    // Vector3d start_offset = {0.0, 0.0, 0.0};
    // Vector3d startIdx3d = (start_offset - _map_origin) * _inv_resolution;
    // Coord3D end_point = {(unsigned int)round(startIdx3d[0]), (unsigned int)round(startIdx3d[1]), (unsigned int)round(startIdx3d[2])};
    // unsigned int goalIdx;
    // grid_fmm_3d.coord2idx(end_point, goalIdx);

    Path3D path3D;
    vector<double> path_vels, time;
    GradientDescent< FMGrid3D > grad3D;

    vector<Vector3d> path_coord;
    int cnt2 = 0;

    unsigned int fmm_max_reward_grid_idx = (unsigned int)fmm_max_reward_idx[3];
    if(grad3D.gradient_descent(grid_fmm_3d, fmm_max_reward_grid_idx, path3D, path_vels, time) == -1)
    {
        std::cout << "GRADIENT DESCENT FMM ERROR" << std::endl;
    }
    else
    {
        // DEBUGGING
        // std::cout << "Path3D is :" << std::endl;
        // for (auto pt: path3D)
        // {
        //     for (int elem : pt)
        //     {
        //         std::cout << elem << ' ';
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
  
        for(int path_idx = (int)path3D.size() - 1; path_idx >= 0; path_idx--)
        {
            cnt2++;
            vector<int> path3Dxyz = {path3D[path_idx][0], path3D[path_idx][1], path3D[path_idx][2]};
            Vector3d pt = xyz2ptVect(path3Dxyz);
            path_coord.push_back(pt);

            // DEBUGGING
            // std::cout << "path[" << path_idx << "]: " << pt(0) - _start_pt(0) << ", " << pt(1) - _start_pt(1) << ", " << pt(2) - _start_pt(2) << std::endl;
        }
        publishVisPath(path_coord, _fm_path_vis_pub);
    }

    // DEBUGGING
    // GridWriter::saveGridValues("test_fm3d.txt", grid_fmm_3d);
    
    //DEBUGGING
    // std::cout << "path3D.size(): " << path3D.size() << std::endl;
    // std::cout << "cnt2: " << cnt2 << std::endl;

    /////////////////////////
    // GOAL POINT SELCTION //
    /////////////////////////

    vector<Vector3d> path_line_of_sight;
    int line_of_sight_idx = 0;
    int collision_idx = 0;
    Vector3d line_of_sight_pt;
    Vector3d collision_pt;
    Vector3d goal_pt;
    float check_vect_dist = 0;
    float check_vect_dist_max = 0;

    visualization_msgs::MarkerArray line_of_sight_vectors_msg;
    visualization_msgs::MarkerArray collision_vectors_msg;

    // Find xyz coordinate of robot point 
    vector<float> robot_pt_xyz = {0.5 * _resolution + _start_pt(0) - _start_pt_rounding_eror(0),
                                  0.5 * _resolution + _start_pt(1) - _start_pt_rounding_eror(1),
                                  0.5 * _resolution + _start_pt(2) - _start_pt_rounding_eror(2)};
                                    
    // Find xyz indeces of robot point in FMM map
    vector<long int> robot_pt_idx = ptVect2idx(robot_pt_xyz);
    vector<int> robot_xyz_idx = {robot_pt_idx[0], robot_pt_idx[1], robot_pt_idx[2]};

    // Find xyz coordinate of robot point (from FMM map)
    Vector3d robot_pt = xyz2ptVect(robot_xyz_idx);

    geometry_msgs::PointStamped goal_point_uncropped;
    geometry_msgs::PoseStamped goal_pose_uncropped;
    geometry_msgs::PointStamped goal_point;
    geometry_msgs::PoseStamped goal_pose;

    for(int path_idx = 0; path_idx < int(path_coord.size()); path_idx++)
    {
        // Find xyz coordinate of path point[path_idx]
        vector<float> path_pt_xyz = {path_coord[path_idx](0),
                                     path_coord[path_idx](1),
                                     path_coord[path_idx](2)};

        float path_pt_distance = sqrt(path_pt_xyz[0]*path_pt_xyz[0] + path_pt_xyz[1]*path_pt_xyz[1] + path_pt_xyz[2]*path_pt_xyz[2]);
        int num_interpolated_pts_to_check = ceil(2 * _inv_resolution * (float)path_pt_distance);
        bool collision_detected = 0;

        for(int interp_idx = 1; interp_idx < num_interpolated_pts_to_check + 1; interp_idx++)
        {
            float percent_dist = (float)interp_idx / (float)num_interpolated_pts_to_check;
            vector<float> check_vect_xyz = {percent_dist * (path_pt_xyz[0] - robot_pt_xyz[0]),
                                            percent_dist * (path_pt_xyz[1] - robot_pt_xyz[1]),
                                            percent_dist * (path_pt_xyz[2] - robot_pt_xyz[2])};
            check_vect_dist = sqrt(check_vect_xyz[0]*check_vect_xyz[0] + check_vect_xyz[1]*check_vect_xyz[1] + check_vect_xyz[2]*check_vect_xyz[2]);
            vector<float> check_pt_xyz = {check_vect_xyz[0] + robot_pt_xyz[0],
                                          check_vect_xyz[1] + robot_pt_xyz[1],
                                          check_vect_xyz[2] + robot_pt_xyz[2]};
            vector<long int> check_pt_idx = ptVect2idx(check_pt_xyz);
            float d = grid_fmm_3d[check_pt_idx[3]].getOccupancy();
            if((d <= 0) || collision_detected)
            {
                collision_detected = 1;
            }
            if(!collision_detected)
            {
                visualization_msgs::Marker line_of_sight_vector_msg;
                line_of_sight_vector_msg.header.stamp = ros::Time();
                line_of_sight_vector_msg.header.frame_id = "odom";
                line_of_sight_vector_msg.id = line_of_sight_idx;
                line_of_sight_vector_msg.type = visualization_msgs::Marker::ARROW;
                line_of_sight_vector_msg.action = visualization_msgs::Marker::ADD;
                line_of_sight_vector_msg.pose.position.x = 0;
                line_of_sight_vector_msg.pose.position.y = 0;
                line_of_sight_vector_msg.pose.position.z = 0;
                line_of_sight_vector_msg.pose.orientation.x = 0.0;
                line_of_sight_vector_msg.pose.orientation.y = 0.0;
                line_of_sight_vector_msg.pose.orientation.z = 0.0;
                line_of_sight_vector_msg.pose.orientation.w = 1.0;
                line_of_sight_vector_msg.scale.x = 0.001;
                line_of_sight_vector_msg.scale.y = 0.03;
                line_of_sight_vector_msg.scale.z = 0.03;
                line_of_sight_vector_msg.color.a = 1.0; // Don't forget to set the alpha!
                line_of_sight_vector_msg.color.r = 0.0;
                line_of_sight_vector_msg.color.g = 1.0;
                line_of_sight_vector_msg.color.b = 0.0;
                geometry_msgs::Point q;
                q.x = robot_pt(0);
                q.y = robot_pt(1);
                q.z = robot_pt(2);
                line_of_sight_vector_msg.points.push_back(q);
                geometry_msgs::Point p;
                vector<long int> light_of_sight_pt_idx = check_pt_idx;
                vector<int> line_of_sight_xyz_idx = {light_of_sight_pt_idx[0], light_of_sight_pt_idx[1], light_of_sight_pt_idx[2]};
                line_of_sight_pt = xyz2ptVect(line_of_sight_xyz_idx);
                p.x = line_of_sight_pt(0);
                p.y = line_of_sight_pt(1);
                p.z = line_of_sight_pt(2);
                line_of_sight_vector_msg.points.push_back(p);
                line_of_sight_vectors_msg.markers.push_back(line_of_sight_vector_msg);
                if(check_vect_dist > check_vect_dist_max)
                {
                    goal_pt(0) = line_of_sight_pt(0);
                    goal_pt(1) = line_of_sight_pt(1);
                    goal_pt(2) = line_of_sight_pt(2);
                    check_vect_dist_max = check_vect_dist;
                }
                line_of_sight_idx++;
            }
            else
            {
                visualization_msgs::Marker collision_vector_msg;      
                collision_vector_msg.header.stamp = ros::Time();
                collision_vector_msg.header.frame_id = "odom";
                collision_vector_msg.id = collision_idx;
                collision_vector_msg.type = visualization_msgs::Marker::ARROW;
                collision_vector_msg.action = visualization_msgs::Marker::ADD;
                collision_vector_msg.pose.position.x = 0;
                collision_vector_msg.pose.position.y = 0;
                collision_vector_msg.pose.position.z = 0;
                collision_vector_msg.pose.orientation.x = 0.0;
                collision_vector_msg.pose.orientation.y = 0.0;
                collision_vector_msg.pose.orientation.z = 0.0;
                collision_vector_msg.pose.orientation.w = 1.0;
                collision_vector_msg.scale.x = 0.001;
                collision_vector_msg.scale.y = 0.03;
                collision_vector_msg.scale.z = 0.03;
                collision_vector_msg.color.a = 1.0; // Don't forget to set the alpha!
                collision_vector_msg.color.r = 1.0;
                collision_vector_msg.color.g = 0.0;
                collision_vector_msg.color.b = 0.0;
                geometry_msgs::Point q;
                q.x = robot_pt(0);
                q.y = robot_pt(1);
                q.z = robot_pt(2);
                collision_vector_msg.points.push_back(q);
                geometry_msgs::Point p;
                vector<long int> collision_pt_idx = check_pt_idx;
                vector<int> collision_xyz_idx = {collision_pt_idx[0], collision_pt_idx[1], collision_pt_idx[2]};
                collision_pt = xyz2ptVect(collision_xyz_idx);
                p.x = collision_pt(0);
                p.y = collision_pt(1);
                p.z = collision_pt(2);
                collision_vector_msg.points.push_back(p);
                collision_vectors_msg.markers.push_back(collision_vector_msg);
                collision_idx++;
                collision_detected = 1;
            }
        }
        
        goal_point_uncropped.header.stamp = ros::Time::now();
        goal_point_uncropped.header.frame_id = "odom";
        goal_point_uncropped.point.x = goal_pt(0);
        goal_point_uncropped.point.y = goal_pt(1);
        goal_point_uncropped.point.z = goal_pt(2);
            
        goal_pose_uncropped.header.stamp = ros::Time::now();
        goal_pose_uncropped.header.frame_id = "odom";
        goal_pose_uncropped.pose.position.x = goal_pt(0);
        goal_pose_uncropped.pose.position.y = goal_pt(1);
        goal_pose_uncropped.pose.position.z = goal_pt(2);
        goal_pose_uncropped.pose.orientation.x = 0.0;
        goal_pose_uncropped.pose.orientation.y = 0.0;
        goal_pose_uncropped.pose.orientation.z = 0.0;
        goal_pose_uncropped.pose.orientation.w = 1.0;

        float goal_point_uncropped_dist = sqrt((goal_pt(0) - robot_pt(0)) * (goal_pt(0) - robot_pt(0))
                                             + (goal_pt(1) - robot_pt(1)) * (goal_pt(1) - robot_pt(1))
                                             + (goal_pt(2) - robot_pt(2)) * (goal_pt(2) - robot_pt(2)));

        goal_point.header.stamp = ros::Time::now();
        goal_point.header.frame_id = "odom";
        goal_point.point.x = robot_pt(0) + (goal_pt(0) - robot_pt(0)) * _look_ahead_distance / goal_point_uncropped_dist;
        goal_point.point.y = robot_pt(1) + (goal_pt(1) - robot_pt(1)) * _look_ahead_distance / goal_point_uncropped_dist;
        goal_point.point.z = robot_pt(2) + (goal_pt(2) - robot_pt(2)) * _look_ahead_distance / goal_point_uncropped_dist;
            
        goal_pose.header.stamp = ros::Time::now();
        goal_pose.header.frame_id = "odom";
        goal_pose.pose.position.x = robot_pt(0) + (goal_pt(0) - robot_pt(0)) * _look_ahead_distance / goal_point_uncropped_dist;
        goal_pose.pose.position.y = robot_pt(1) + (goal_pt(1) - robot_pt(1)) * _look_ahead_distance / goal_point_uncropped_dist;
        goal_pose.pose.position.z = robot_pt(2) + (goal_pt(2) - robot_pt(2)) * _look_ahead_distance / goal_point_uncropped_dist;
        goal_pose.pose.orientation.x = 0.0;
        goal_pose.pose.orientation.y = 0.0;
        goal_pose.pose.orientation.z = 0.0;
        goal_pose.pose.orientation.w = 1.0;
    }

    _goal_point_uncropped_pub.publish(goal_point_uncropped);
    _goal_pose_uncropped_pub.publish(goal_pose_uncropped);
    _goal_point_pub.publish(goal_point);
    _goal_pose_pub.publish(goal_pose);

    // DEBUGGING
    // std::cout << sqrt(goal_point.point.x*goal_point.point.x + goal_point.point.y*goal_point.point.y + goal_point.point.z*goal_point.point.z) << std::endl;
    std::cout << "goal_point.point: " << goal_point.point.x << ", " << goal_point.point.y << ", " << goal_point.point.z << std::endl;
    
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    // DEBUGGING
    // _line_of_sight_vector_vis_pub.publish(line_of_sight_vectors_msg);
    // _collision_vector_vis_pub.publish(collision_vectors_msg);

}

void publishVisPath(vector<Vector3d> path, ros::Publisher _path_vis_pub)
{

    visualization_msgs::MarkerArray marker_array_msg;
    visualization_msgs::Marker marker_msg;
    for(auto & marker_msg: marker_array_msg.markers)
    {
        marker_msg.action = visualization_msgs::Marker::DELETE;
    }
    
    _path_vis_pub.publish(marker_array_msg);

    marker_array_msg.markers.clear();

    marker_msg.header.frame_id = "odom";
    marker_msg.header.stamp = ros::Time::now();
    marker_msg.ns = "b_traj/fast_marching_path";
    marker_msg.type = visualization_msgs::Marker::CUBE;
    marker_msg.action = visualization_msgs::Marker::ADD;

    marker_msg.pose.orientation.x = 0.0;
    marker_msg.pose.orientation.y = 0.0;
    marker_msg.pose.orientation.z = 0.0;
    marker_msg.pose.orientation.w = 1.0;
    marker_msg.color.a = 0.6;
    marker_msg.color.r = 1.0;
    marker_msg.color.g = 0.647;
    marker_msg.color.b = 0.0;

    for(int path_idx = 0; path_idx < int(path.size()); path_idx++)
    {
        marker_msg.id = path_idx;

        marker_msg.pose.position.x = path[path_idx](0);
        marker_msg.pose.position.y = path[path_idx](1);
        marker_msg.pose.position.z = path[path_idx](2);

        marker_msg.scale.x = _resolution;
        marker_msg.scale.y = _resolution;
        marker_msg.scale.z = _resolution;

        marker_array_msg.markers.push_back(marker_msg);
    }

    _path_vis_pub.publish(marker_array_msg);
}

double velMapping(double d)
{
    double vel;
    if( d <= 0.7)
    {
        vel = 0.2;
    }
    else if( d <= 1.4)
    {
        vel = 0.50632911392 * (d - 0.7) * (d - 0.7) + 0.2;
        // 0.6 = a * (1.4 - 0.7) * (1.4 - 0.7)  * (1.4 - 0.7) + 0.2
    }
    else
    {
        vel = 0.6;
    }
    return vel;
}
