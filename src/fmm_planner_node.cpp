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
unsigned long int _max_grid_idx;

// ROS
ros::Subscriber _map_sub, _pts_sub, _odom_sub;
ros::Publisher _fm_path_vis_pub;
ros::Publisher _map_fmm_vel_vis_pub;
ros::Publisher _map_fmm_timel_vis_pub;
ros::Publisher _line_of_sight_path_vis_pub;
ros::Publisher _line_of_sight_vector_vis_pub;


visualization_msgs::MarkerArray line_of_sight_vectors_msg;
    
void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map);
void rcvOdometryCallbck(const nav_msgs::Odometry odom);
void publishVisPath(vector<Vector3d> path, ros::Publisher _path_vis_pub);
double velMapping(double d);

int main(int argc, char** argv)
{
    ros::init(argc, argv, "b_traj_node");
    ros::NodeHandle nh("~");

    _map_sub  = nh.subscribe( "map",       1, rcvPointCloudCallBack );
    _odom_sub = nh.subscribe( "odometry",  1, rcvOdometryCallbck);

    _map_fmm_vel_vis_pub   = nh.advertise<sensor_msgs::PointCloud2>("map/fmm/velocity", 1);
    _map_fmm_timel_vis_pub    = nh.advertise<sensor_msgs::PointCloud2>("map/fmm/arrival_time", 1);
    _fm_path_vis_pub    = nh.advertise<visualization_msgs::MarkerArray>("goal/fmm_path/viz", 1);
    _line_of_sight_path_vis_pub    = nh.advertise<visualization_msgs::MarkerArray>("goal/line_of_sight_path/viz", 1);
    _line_of_sight_vector_vis_pub    = nh.advertise<visualization_msgs::MarkerArray>("goal/line_of_sight/vectors/viz", 1);

    nh.param("map/resolution",  _resolution, 0.2);
    nh.param("map/x_size",      _x_size, 50.0);
    nh.param("map/y_size",      _y_size, 50.0);
    nh.param("map/z_size",      _z_size, 50.0);

    // Origin is located in the middle of the map
    _map_origin = {- _x_size / 2.0, - _y_size / 2.0 , - _z_size / 2.0};

    // Inverse resolution
    _inv_resolution = 1.0 / _resolution;

    // This is the maximum indeces in the map
    _max_x_idx = (int)(_x_size * _inv_resolution);
    _max_y_idx = (int)(_y_size * _inv_resolution);
    _max_z_idx = (int)(_z_size * _inv_resolution);
    _max_grid_idx = _max_x_idx * _max_y_idx * _max_z_idx;

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

    // DEBUGGING
    // std::cout << "_start_pt_rounding_eror" << _start_pt_rounding_eror(0) << ", " << _start_pt_rounding_eror(1) << ", " << _start_pt_rounding_eror(2) << std::endl;

}

void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map)
{

    pcl::PointCloud<pcl::PointXYZI> cloud;  
    pcl::fromROSMsg(pointcloud_map, cloud);

    std::cout << "_start_pt: " << _start_pt(0) << ", " << _start_pt(1) << ", " << _start_pt(2) << std::endl;

    std::string frame;
    
    int x_size = (int)(_max_x_idx);
    int y_size = (int)(_max_y_idx);
    int z_size = (int)(_max_z_idx);
    

    ////////////////////
    // LOCAL ESDF MAP //
    ////////////////////
    
    Coord3D dimsize {x_size, y_size, z_size};
    FMGrid3D grid_fmm_3d(dimsize);
    
    grid_fmm_3d.clear();

    vector<unsigned int> obs;

    for(long int grid_idx = 0; grid_idx < _max_grid_idx; grid_idx++)
    {
        grid_fmm_3d[grid_idx].setOccupancy(0.0);
    }

    long int cntt2 = 0;
    // Assign the ESDF
    for (unsigned long int pcl_idx = 0; pcl_idx < cloud.points.size(); pcl_idx++)
    {
        cntt2++;
        pcl::PointXYZI fmm_vel_pt = cloud.points[pcl_idx];
        int  x_idx = ((fmm_vel_pt.x - _start_pt(0) - _map_origin(0)) * _inv_resolution);
        int  y_idx = ((fmm_vel_pt.y - _start_pt(1) - _map_origin(1)) * _inv_resolution);
        int  z_idx = ((fmm_vel_pt.z - _start_pt(2) - _map_origin(2)) * _inv_resolution);
        unsigned long int grid_idx = x_idx + y_idx * x_size + z_idx * x_size * y_size;

        if(x_idx >= 0 && x_idx < _max_x_idx)
        {
            if(y_idx >= 0 && y_idx < _max_y_idx)
            {
                if(z_idx >= 0 && z_idx < _max_z_idx)
                {
                    grid_fmm_3d[grid_idx].setOccupancy(velMapping(fmm_vel_pt.intensity));
                }
            }
        }
    }
    std::cout << cntt2 << std::endl;

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

    //////////////////////////////////
    // LOCAL FMM MAP (ARRIVAL TIME) //
    //////////////////////////////////

    Vector3d endIdx3d = - _map_origin * _inv_resolution;

    Coord3D init_point = {(unsigned int)round(endIdx3d[0]), (unsigned int)round(endIdx3d[1]), (unsigned int)round(endIdx3d[2])};

    unsigned int startIdx;
    vector<unsigned int> startIndices;
    grid_fmm_3d.coord2idx(init_point, startIdx);

    startIndices.push_back(startIdx);

    Solver<FMGrid3D>* fm_solver = new FMMStar<FMGrid3D>("FMM*_Dist", TIME); // LSM, FMM

    fm_solver->setEnvironment(&grid_fmm_3d);
    fm_solver->setInitialPoints(startIndices);
    fm_solver->setup();
    fm_solver->compute(1.0);

    // Preventing memory leaks.
    delete fm_solver;

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
        int z_idx =  grid_idx / (x_size * y_size);
        int y_idx = (grid_idx - z_idx * x_size * y_size) / x_size;
        int x_idx =  grid_idx - z_idx * x_size * y_size - y_idx * x_size;
        pcl::PointXYZI fmm_vel_pt;
        fmm_vel_pt.x = (x_idx + 0.5) * _resolution + _start_pt(0) - _start_pt_rounding_eror(0) + _map_origin(0);
        fmm_vel_pt.y = (y_idx + 0.5) * _resolution + _start_pt(1) - _start_pt_rounding_eror(1) + _map_origin(1);
        fmm_vel_pt.z = (z_idx + 0.5) * _resolution + _start_pt(2) - _start_pt_rounding_eror(2) + _map_origin(2);
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

    sensor_msgs::PointCloud2 esdfMap;
    pcl::toROSMsg(cloud_fmm_vel, esdfMap);
    _map_fmm_vel_vis_pub.publish(esdfMap);


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
        int z_idx =  grid_idx / (x_size * y_size);
        int y_idx = (grid_idx - z_idx * x_size * y_size) / x_size;
        int x_idx =  grid_idx - z_idx * x_size * y_size - y_idx * x_size;
        pcl::PointXYZI fmm_time_pt;
        fmm_time_pt.x = (x_idx + 0.5) * _resolution + _start_pt(0) - _start_pt_rounding_eror(0) + _map_origin(0);
        fmm_time_pt.y = (y_idx + 0.5) * _resolution + _start_pt(1) - _start_pt_rounding_eror(1) + _map_origin(1);
        fmm_time_pt.z = (z_idx + 0.5) * _resolution + _start_pt(2) - _start_pt_rounding_eror(2) + _map_origin(2);
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

    sensor_msgs::PointCloud2 fmmMap;
    pcl::toROSMsg(cloud_fmm_time, fmmMap);
    _map_fmm_timel_vis_pub.publish(fmmMap);

    /////////////////////////////
    // LOCAL PATH (FMM SOLVER) //
    /////////////////////////////

    Vector3d goal_offset = {3, 0, 0};
    Vector3d startIdx3d = (goal_offset - _map_origin) * _inv_resolution;
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
        publishVisPath(path_coord, _fm_path_vis_pub);


    }


    GridWriter::saveGridValues("test_fm3d.txt", grid_fmm_3d);
    
    std::cout << "path3D.size(): " << path3D.size() << std::endl;
    std::cout << "cnt2: " << cnt2 << std::endl;


    /////////////////////////
    // GOAL POINT SELCTION //
    /////////////////////////


    vector<Vector3d> path_line_of_sight;
    int vec_idx = 0;

    // visualization_msgs::MarkerArray line_of_sight_vectors_msg;

    visualization_msgs::MarkerArray line_of_sight_vectors_msg;

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

            int  x_idx = ((path_coord[path_idx] - _start_pt(0) - _map_origin(0)) * _inv_resolution);
            int  y_idx = ((path_coord[path_idx] - _start_pt(1) - _map_origin(1)) * _inv_resolution);
            int  z_idx = ((path_coord[path_idx] - _start_pt(2) - _map_origin(2)) * _inv_resolution);
            unsigned long int grid_idx = x_idx + y_idx * x_size + z_idx * x_size * y_size;
            float d = grid_fmm_3d[grid_idx].getOccupancy();
            if(d > 0)
            {

                visualization_msgs::Marker line_of_sight_vector_msg;
                std::cout << "got to here 1" << vec_idx <<  std::endl;
                line_of_sight_vector_msg.header.frame_id = "odom";
                std::cout << "got to here 1.5" <<  std::endl;                
                line_of_sight_vector_msg.header.stamp = ros::Time();
                // line_of_sight_vector_msg.ns = "my_namespace";
                std::cout << "got to here 2" <<  std::endl;
                line_of_sight_vector_msg.id = vec_idx;
                line_of_sight_vector_msg.type = visualization_msgs::Marker::ARROW;
                line_of_sight_vector_msg.action = visualization_msgs::Marker::ADD;
                line_of_sight_vector_msg.pose.position.x = 0;
                line_of_sight_vector_msg.pose.position.y = 0;
                line_of_sight_vector_msg.pose.position.z = 0;
                line_of_sight_vector_msg.pose.orientation.x = 0.0;
                line_of_sight_vector_msg.pose.orientation.y = 0.0;
                line_of_sight_vector_msg.pose.orientation.z = 0.0;
                line_of_sight_vector_msg.pose.orientation.w = 1.0;
                line_of_sight_vector_msg.scale.x = 0.02;
                line_of_sight_vector_msg.scale.y = 0.05;
                line_of_sight_vector_msg.scale.z = 0.05;
                line_of_sight_vector_msg.color.a = 1.0; // Don't forget to set the alpha!
                line_of_sight_vector_msg.color.r = 0.0;
                line_of_sight_vector_msg.color.g = 1.0;
                line_of_sight_vector_msg.color.b = 0.0;
                std::cout << "got to here 3" <<  std::endl;
                geometry_msgs::Point q;
                q.x = _start_pt(0);
                q.y = _start_pt(1);
                q.z = _start_pt(2);             
                line_of_sight_vector_msg.points.push_back(q);
                std::cout << "got to here 4" <<  std::endl;
                geometry_msgs::Point p;
                p.x = (x_idx + 0.5) * _resolution + _start_pt(0) - _start_pt_rounding_eror(0) + _map_origin(0);
                p.y = (y_idx + 0.5) * _resolution + _start_pt(1) - _start_pt_rounding_eror(1) + _map_origin(1);
                p.z = (z_idx + 0.5) * _resolution + _start_pt(2) - _start_pt_rounding_eror(2) + _map_origin(2);
                line_of_sight_vector_msg.points.push_back(p);
                std::cout << "got to here 5" <<  std::endl;

                line_of_sight_vectors_msg.markers.push_back(line_of_sight_vector_msg);

                vec_idx++;
            }
            else
            {
                std::cout << "ERRRRRRROOOOOOORRRRRR:" << d << std::endl;
            }

        std::cout << "got to here 6" <<  std::endl; 
        _line_of_sight_vector_vis_pub.publish(line_of_sight_vectors_msg);
        std::cout << "got to here 7" <<  std::endl;

            
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

        if(path_idx == int(path_coord.size()) - 1)
        {
            for(int interp_idx = 1; interp_idx < num_interpolated_pts_to_check + 1; interp_idx++)
            {
                // DEBUGGING
                // std::cout << "num_interpolated_pts_to_check: " << num_interpolated_pts_to_check << std::endl;
                Vector3d line_of_sight_pt = path_pt / num_interpolated_pts_to_check * interp_idx + _start_pt;
                path_line_of_sight.push_back(line_of_sight_pt);
            }            
        publishVisPath(path_line_of_sight, _line_of_sight_path_vis_pub);
        }
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
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
