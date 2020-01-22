#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

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

#include "trajectory_generator.h"
#include "bezier_base.h"
#include "data_type.h"
#include "utils.h"
#include "a_star.h"
#include "backward.hpp"

#include "quadrotor_msgs/PositionCommand.h"
#include "quadrotor_msgs/PolynomialTrajectory.h"






#include <fast_methods/ndgridmap/fmcell.h>
#include <fast_methods/ndgridmap/ndgridmap.hpp>

#include <fast_methods/fm/fmm.hpp>
#include <fast_methods/fm/sfmm.hpp>
#include <fast_methods/fm/sfmmstar.hpp>
#include <fast_methods/fm/fmmstar.hpp>
#include <fast_methods/datastructures/fmfibheap.hpp>
#include <fast_methods/datastructures/fmpriorityqueue.hpp>
#include <fast_methods/fm/fim.hpp>
#include <fast_methods/fm/gmm.hpp>
#include <fast_methods/fm/ufmm.hpp>
#include <fast_methods/fm/fsm.hpp>
#include <fast_methods/fm/lsm.hpp>
#include <fast_methods/fm/ddqm.hpp>



#include <fast_methods/io/gridplotter.hpp>
#include <fast_methods/io/gridwriter.hpp>



using namespace std;
using namespace Eigen;
using namespace sdf_tools;

namespace backward {
backward::SignalHandling sh;
}

// simulation param from launch file
double _vis_traj_width;
double _resolution, _inv_resolution;
double _cloud_margin, _cube_margin, _check_horizon, _stop_horizon;
double _x_size, _y_size, _z_size, _x_local_size, _y_local_size, _z_local_size;
double _MAX_Vel, _MAX_Acc;
bool   _is_use_fm, _is_proj_cube, _is_limit_vel, _is_limit_acc;
int    _step_length, _max_inflate_iter, _traj_order;
double _minimize_order;

// useful global variables
nav_msgs::Odometry _odom;
bool _has_odom  = false;
bool _has_map   = false;
bool _has_target= false;
bool _has_traj  = false;
bool _is_emerg  = false;
bool _is_init   = true;

Vector3d _start_pt, _end_pt;
double _init_x, _init_y, _init_z;
Vector3d _map_origin;
double _pt_max_x, _pt_min_x, _pt_max_y, _pt_min_y, _pt_max_z, _pt_min_z;
int _max_x_id, _max_y_id, _max_z_id, _max_local_x_id, _max_local_y_id, _max_local_z_id;
int _traj_id = 1;
COLLISION_CELL _free_cell(0.0);
COLLISION_CELL _obst_cell(1.0);
// ros related
ros::Subscriber _map_sub, _pts_sub, _odom_sub;
ros::Publisher _fm_path_vis_pub, _local_map_vis_pub, _esdf_map_vis_pub, _fmm_map_vis_pub, _inf_map_vis_pub, _corridor_vis_pub, _traj_vis_pub, _grid_path_vis_pub, _nodes_vis_pub, _traj_pub, _checkTraj_vis_pub, _stopTraj_vis_pub;

// trajectory related
int _seg_num;
VectorXd _seg_time;
MatrixXd _bezier_coeff;

// bezier basis constant
MatrixXd _MQM, _FM;
VectorXd _C, _Cv, _Ca, _Cj;

// useful object
quadrotor_msgs::PolynomialTrajectory _traj;
ros::Time _start_time = ros::TIME_MAX;
TrajectoryGenerator _trajectoryGenerator;
CollisionMapGrid * collision_map       = new CollisionMapGrid();
CollisionMapGrid * collision_map_local = new CollisionMapGrid();
gridPathFinder * path_finder           = new gridPathFinder();

void rcvWaypointsCallback(const nav_msgs::Path & wp);
void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map);
void rcvOdometryCallbck(const nav_msgs::Odometry odom);

bool checkExecTraj();
bool checkCoordObs(Vector3d checkPt);
vector<pcl::PointXYZ> pointInflate( pcl::PointXYZ pt);

void visPath(vector<Vector3d> path);
void visCorridor(vector<Cube> corridor);
void visGridPath( vector<Vector3d> grid_path);
void visExpNode( vector<GridNodePtr> nodes);
void visBezierTrajectory(MatrixXd polyCoeff, VectorXd time);

pair<Cube, bool> inflateCube(Cube cube, Cube lstcube);
Cube generateCube( Vector3d pt) ;
bool isContains(Cube cube1, Cube cube2);
void corridorSimplify(vector<Cube> & cubicList);
vector<Cube> corridorGeneration(vector<Vector3d> path_coord, vector<double> time);
vector<Cube> corridorGeneration(vector<Vector3d> path_coord);
void sortPath(vector<Vector3d> & path_coord, vector<double> & time);
void timeAllocation(vector<Cube> & corridor, vector<double> time);
void timeAllocation(vector<Cube> & corridor);

VectorXd getStateFromBezier(const MatrixXd & polyCoeff, double t_now, int seg_now );
Vector3d getPosFromBezier(const MatrixXd & polyCoeff, double t_now, int seg_now );
quadrotor_msgs::PolynomialTrajectory getBezierTraj();
quadrotor_msgs::PolynomialTrajectory getTraj();


void rcvOdometryCallbck(const nav_msgs::Odometry odom)
{
    // if (odom.header.frame_id != "uav") 
        // return ;

    _odom = odom;
    _has_odom = true;

    _start_pt(0)  = _odom.pose.pose.position.x;
    _start_pt(1)  = _odom.pose.pose.position.y;
    _start_pt(2)  = _odom.pose.pose.position.z;
}

void rcvWaypointsCallback(const nav_msgs::Path & wp)
{     
    if(wp.poses[0].pose.position.z < 0.0)
        return;

    _is_init = false;
    _end_pt << wp.poses[0].pose.position.x,
               wp.poses[0].pose.position.y,
               wp.poses[0].pose.position.z;

    _has_target = true;
    _is_emerg   = true;

    ROS_INFO("[Fast Marching Node] receive the way-points");

}

void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map)
{

    pcl::PointCloud<pcl::PointXYZI> cloud;  
    pcl::fromROSMsg(pointcloud_map, cloud);

    std::cout << "_start_pt: " << _start_pt(0) << ", " << _start_pt(1) << ", " << _start_pt(2) << std::endl;

    Eigen::Affine3d origin_transform;
    std::string frame;
    double resolution = 0.2;
    double x_size = 50;
    double y_size = 50;
    double z_size = 50;
    
    unsigned int size_x = (unsigned int)(_max_x_id);
    unsigned int size_y = (unsigned int)(_max_y_id);
    unsigned int size_z = (unsigned int)(_max_z_id);

    Coord3D dimsize {size_x, size_y, size_z};
    FMGrid3D grid_fmm_3d(dimsize);
    
    grid_fmm_3d.clear();

    double max_vel = _MAX_Vel * 0.75;
    vector<unsigned int> obs;

    unsigned long int idx;
    unsigned long int index;

    for(unsigned long int i = 0; i < size_x*size_y*size_z; i++)
    {
        grid_fmm_3d[i].setOccupancy(0.0);
    }

    // Assign the ESDF
    for (unsigned long int idx = 0; idx < cloud.points.size(); idx++)
    {
        pcl::PointXYZI mk = cloud.points[idx];
        unsigned int  i = mk.x - _start_pt(0) - _map_origin(0) * _inv_resolution;
        unsigned int  j = mk.y - _start_pt(1) - _map_origin(1) * _inv_resolution;
        unsigned int  k = mk.z - _start_pt(2) - _map_origin(2) * _inv_resolution;
        index = i + j * size_x + k * size_x * size_y;
        grid_fmm_3d[index].setOccupancy(mk.intensity);
    }

    for(unsigned long int i = 0; i < size_x*size_y*size_z; i++)
    {
        if (grid_fmm_3d[i].isOccupied())
        {
            obs.push_back(idx);
        }
    }

    grid_fmm_3d.setOccupiedCells(std::move(obs));
    grid_fmm_3d.setLeafSize(_resolution);

    GridWriter::saveGridValues("test_fm3d.txt", grid_fmm_3d);

    // Vector3d startIdx3d = - _map_origin * _inv_resolution;
    // Coord3D goal_point = {(unsigned int)startIdx3d[0], (unsigned int)startIdx3d[1], (unsigned int)startIdx3d[2]};
    // unsigned int goalIdx;
    // grid_fmm_3d.coord2idx(goal_point, goalIdx);
    // grid_fmm_3d[goalIdx].setOccupancy(max_vel);
    // vector<unsigned int> goalIndices;
    // goalIndices.push_back(goalIdx);

    // std::cout << "startIdx3d: " << startIdx3d[0] << ", " << startIdx3d[1] << ", " << startIdx3d[2] << ", " << std::endl;
    
    


    // Coord3D init_point = {35, 25, 25};
    // Coord3D goal_point = {25, 25, 25};
    // unsigned int goal;
    // grid_fmm_3d.coord2idx(goal_point, goal);
    // unsigned int init;
    // grid_fmm_3d.coord2idx(goal_point, init);
    // vector<unsigned int> goalIndices;
    // goalIndices.push_back(goal);

    // // Solvers declaration.
    // std::vector<Solver<FMGrid3D>*> solvers;
    // solvers.push_back(new FMMStar<FMGrid3D>("FMM*_Dist", DISTANCE));
    // for (Solver<FMGrid3D>* s : solvers)
    // {
    //     s->setEnvironment(&grid_fmm_3d);
    //     s->setInitialAndGoalPoints(init, goalIndices);
    //     s->setup(); // Not mandatory, but in FMMstar we want to precompute distances out of compute().
    //     s->compute(max_vel);
    //     cout << "\tElapsed "<< s->getName() <<" time: " << s->getTime() << " ms" << '\n';
    //     // GridPlotter::plotArrivalTimes(grid_fmm, s->getName());
    // }
    Vector3d offset = {1, 0, 0};
    Vector3d startIdx3d = (- _map_origin) * _inv_resolution;
    Vector3d endIdx3d   = (offset - _map_origin) * _inv_resolution;

    Coord3D goal_point = {(unsigned int)startIdx3d[0], (unsigned int)startIdx3d[1], (unsigned int)startIdx3d[2]};
    Coord3D init_point = {(unsigned int)endIdx3d[0],   (unsigned int)endIdx3d[1],   (unsigned int)endIdx3d[2]};

    unsigned int startIdx;
    vector<unsigned int> startIndices;
    grid_fmm_3d.coord2idx(init_point, startIdx);

    startIndices.push_back(startIdx);

    unsigned int goalIdx;
    grid_fmm_3d.coord2idx(goal_point, goalIdx);
    grid_fmm_3d[goalIdx].setOccupancy(max_vel);

    Solver<FMGrid3D>* fm_solver = new FMMStar<FMGrid3D>("FMM*_Dist", TIME); // LSM, FMM

    fm_solver->setEnvironment(&grid_fmm_3d);
    fm_solver->setInitialAndGoalPoints(startIndices, goalIdx);
    fm_solver->compute(max_vel);






    // Solver<FMGrid3D>* fm_solver = new FMMStar<FMGrid3D>("FMM*_Dist", TIME); // LSM, FMM
    // fm_solver->setEnvironment(&grid_fmm_3d);
    // fm_solver->setInitialPoints(goalIndices);

    // ros::Time time_bef_fm = ros::Time::now();
    // if(fm_solver->compute(max_vel) == -1)
    // {
    //     ROS_WARN("[Fast Marching Node] No path can be found");
    //     _traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_WARN_IMPOSSIBLE;
    //     _traj_pub.publish(_traj);
    //     _has_traj = false;

    //     return;
    // }
    // ros::Time time_aft_fm = ros::Time::now();
    // ROS_WARN("[Fast Marching Node] Time in Fast Marching computing is %f", (time_aft_fm - time_bef_fm).toSec() );

    // Path3D path3D;
    // vector<double> path_vels, time;
    // GradientDescent< FMGrid3D > grad3D;

    // if(grad3D.gradient_descent(grid_fmm_3d, goalIndices, path3D, path_vels, time) == -1)
    // {
    //     ROS_WARN("[Fast Marching Node] FMM failed, valid path not exists");
    //     if(_has_traj && _is_emerg)
    //     {
    //         _traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_WARN_IMPOSSIBLE;
    //         _traj_pub.publish(_traj);
    //         _has_traj = false;
    //     }
    //     return;
    // }
    
    // for (int i = 0; i < path3D.size(); i++)
    // {
    //     std::cout << "(";
    //     for (int j = 0; j < path3D[i].size(); j++)
    //     {
    //         std::cout << path3D[i][j];
    //         if(j != path3D[i].size() - 1)
    //         {
    //             std::cout << ", ";
    //         }
    //     }
    //     std::cout << ")" << std::endl;
    // }





    pcl::PointCloud<pcl::PointXYZI> cloud_esdf;
    cloud_esdf.height = 1;
    cloud_esdf.is_dense = true;
    cloud_esdf.header.frame_id = "odom";

    long int cnt = 0;
    for(unsigned long int idx = 0; idx < size_x*size_y*size_z; idx++)
    {
        // std::cout << "hello" << idx << std::endl;
        int k = idx / (size_x * size_y);
        int j = (idx - k * size_x * size_y) / size_x;
        int i = idx - k * size_x * size_y - j * size_x;
        pcl::PointXYZI esdf_pt;
        esdf_pt.x = i * _resolution + _start_pt(0) + _map_origin(0);
        esdf_pt.y = j * _resolution + _start_pt(1) + _map_origin(1);
        esdf_pt.z = k * _resolution + _start_pt(2) + _map_origin(2);
        esdf_pt.intensity = grid_fmm_3d[idx].getOccupancy();
        // std::cout << fmm_pt.x << ", " << fmm_pt.y << ", " << fmm_pt.z << ", " << fmm_pt.intensity << std::endl;
        if(esdf_pt.intensity > 0)
        {
            cnt++;
            cloud_esdf.push_back(esdf_pt);
        }
    }

    std::cout << "_resolution" << _resolution << std::endl;

    cloud_esdf.width = cnt;
    std::cout << cnt << std::endl;
    std::cout << cloud_esdf.points.size() << std::endl;

    sensor_msgs::PointCloud2 esdfMap;
    pcl::toROSMsg(cloud_esdf, esdfMap);
    _esdf_map_vis_pub.publish(esdfMap);


    pcl::PointCloud<pcl::PointXYZI> cloud_fmm;
    cloud_fmm.height = 1;
    cloud_fmm.is_dense = true;
    cloud_fmm.header.frame_id = "odom";

    cnt = 0;
    for(unsigned long int idx = 0; idx < size_x*size_y*size_z; idx++)
    {
        // std::cout << "hello" << idx << std::endl;
        int k = idx / (size_x * size_y);
        int j = (idx - k * size_x * size_y) / size_x;
        int i = idx - k * size_x * size_y - j * size_x;
        pcl::PointXYZI fmm_pt;
        fmm_pt.x = (i + 0.5) * _resolution + _start_pt(0) + _map_origin(0);
        fmm_pt.y = (j + 0.5) * _resolution + _start_pt(1) + _map_origin(1);
        fmm_pt.z = (k + 0.5) * _resolution + _start_pt(2) + _map_origin(2);
        fmm_pt.intensity = grid_fmm_3d[idx].getArrivalTime();
        // std::cout << fmm_pt.x << ", " << fmm_pt.y << ", " << fmm_pt.z << ", " << fmm_pt.intensity << std::endl;
        if(fmm_pt.intensity < 99999)
        {
            cnt++;
            cloud_fmm.push_back(fmm_pt);
        }
    }

    cloud_fmm.width = cnt;
    std::cout << cnt << std::endl;
    std::cout << cloud_fmm.points.size() << std::endl;

    sensor_msgs::PointCloud2 fmmMap;
    pcl::toROSMsg(cloud_fmm, fmmMap);
    _fmm_map_vis_pub.publish(fmmMap);

}

quadrotor_msgs::PolynomialTrajectory getTraj()
{
      quadrotor_msgs::PolynomialTrajectory traj;
      traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_ADD;
      traj.num_segment = _seg_num;

      int order = _traj_order;
      int poly_num1d = order + 1;
      int polyTotalNum = _seg_num * (order + 1);

      traj.coef_x.resize(polyTotalNum);
      traj.coef_y.resize(polyTotalNum);
      traj.coef_z.resize(polyTotalNum);

      int idx = 0;
      for(int i = 0; i < _seg_num; i++ )
      {
          for(int j =0; j < poly_num1d; j++)
          {
              traj.coef_x[idx] = _bezier_coeff(i,                  j);
              traj.coef_y[idx] = _bezier_coeff(i,     poly_num1d + j);
              traj.coef_z[idx] = _bezier_coeff(i, 2 * poly_num1d + j);
              idx++;
          }
      }

      traj.header.frame_id = "/bernstein";
      traj.header.stamp = _odom.header.stamp;
      _start_time = traj.header.stamp;

      traj.time.resize(_seg_num);
      traj.order.resize(_seg_num);

      traj.mag_coeff = 1.0;
      for (int idx = 0; idx < _seg_num; ++idx){
          traj.time[idx] = _seg_time(idx);
          traj.order[idx] = _traj_order;
      }

      traj.start_yaw = 0.0;
      traj.final_yaw = 0.0;

      traj.trajectory_id = _traj_id;
      traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_ADD;

      return traj;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "b_traj_node");
    ros::NodeHandle nh("~");

    _map_sub  = nh.subscribe( "map",       1, rcvPointCloudCallBack );
    _odom_sub = nh.subscribe( "odometry",  1, rcvOdometryCallbck);
    _pts_sub  = nh.subscribe( "waypoints", 1, rcvWaypointsCallback );

    _inf_map_vis_pub   = nh.advertise<sensor_msgs::PointCloud2>("vis_map_inflate", 1);
    _local_map_vis_pub = nh.advertise<sensor_msgs::PointCloud2>("vis_map_local", 1);
    _esdf_map_vis_pub   = nh.advertise<sensor_msgs::PointCloud2>("vis_map_esdf", 1);
    _fmm_map_vis_pub   = nh.advertise<sensor_msgs::PointCloud2>("vis_map_fmm", 1);
    _traj_vis_pub      = nh.advertise<visualization_msgs::Marker>("trajectory_vis", 1);
    _corridor_vis_pub  = nh.advertise<visualization_msgs::MarkerArray>("corridor_vis", 1);
    _fm_path_vis_pub   = nh.advertise<visualization_msgs::MarkerArray>("path_vis", 1);
    _grid_path_vis_pub = nh.advertise<visualization_msgs::MarkerArray>("grid_path_vis", 1);
    _nodes_vis_pub     = nh.advertise<visualization_msgs::Marker>("expanded_nodes_vis", 1);
    _checkTraj_vis_pub = nh.advertise<visualization_msgs::Marker>("check_trajectory", 1);
    _stopTraj_vis_pub  = nh.advertise<visualization_msgs::Marker>("stop_trajectory", 1);

    _traj_pub = nh.advertise<quadrotor_msgs::PolynomialTrajectory>("trajectory", 10);

    nh.param("map/margin",     _cloud_margin, 0.25);
    nh.param("map/resolution", _resolution, 0.2);

    nh.param("map/x_size",       _x_size, 50.0);
    nh.param("map/y_size",       _y_size, 50.0);
    nh.param("map/z_size",       _z_size, 5.0 );

    nh.param("map/x_local_size", _x_local_size, 20.0);
    nh.param("map/y_local_size", _y_local_size, 20.0);
    nh.param("map/z_local_size", _z_local_size, 5.0 );

    nh.param("planning/init_x",       _init_x,  0.0);
    nh.param("planning/init_y",       _init_y,  0.0);
    nh.param("planning/init_z",       _init_z,  0.0);

    nh.param("planning/max_vel",       _MAX_Vel,  1.0);
    nh.param("planning/max_acc",       _MAX_Acc,  1.0);
    nh.param("planning/max_inflate",   _max_inflate_iter, 100);
    nh.param("planning/step_length",   _step_length,     2);
    nh.param("planning/cube_margin",   _cube_margin,   0.2);
    nh.param("planning/check_horizon", _check_horizon,10.0);
    nh.param("planning/stop_horizon",  _stop_horizon,  5.0);
    nh.param("planning/is_limit_vel",  _is_limit_vel,  false);
    nh.param("planning/is_limit_acc",  _is_limit_acc,  false);
    nh.param("planning/is_use_fm",     _is_use_fm,  true);

    nh.param("optimization/min_order",  _minimize_order, 3.0);
    nh.param("optimization/poly_order", _traj_order,    10);

    nh.param("vis/vis_traj_width", _vis_traj_width, 0.15);
    nh.param("vis/is_proj_cube",   _is_proj_cube, true);

    Bernstein _bernstein;
    if(_bernstein.setParam(3, 12, _minimize_order) == -1)
        ROS_ERROR(" The trajectory order is set beyond the library's scope, please re-set ");

    _MQM = _bernstein.getMQM()[_traj_order];
    _FM  = _bernstein.getFM()[_traj_order];
    _C   = _bernstein.getC()[_traj_order];
    _Cv  = _bernstein.getC_v()[_traj_order];
    _Ca  = _bernstein.getC_a()[_traj_order];
    _Cj  = _bernstein.getC_j()[_traj_order];

    _map_origin << -_x_size/2.0, -_y_size/2.0, -_z_size/2.0;
    _pt_max_x = + _x_size / 2.0;
    _pt_min_x = - _x_size / 2.0;
    _pt_max_y = + _y_size / 2.0;
    _pt_min_y = - _y_size / 2.0;
    _pt_max_z = + _z_size;
    _pt_min_z = 0.0;

    _inv_resolution = 1.0 / _resolution;
    _max_x_id = (int)(_x_size * _inv_resolution);
    _max_y_id = (int)(_y_size * _inv_resolution);
    _max_z_id = (int)(_z_size * _inv_resolution);
    _max_local_x_id = (int)(_x_local_size * _inv_resolution);
    _max_local_y_id = (int)(_y_local_size * _inv_resolution);
    _max_local_z_id = (int)(_z_local_size * _inv_resolution);

    Vector3i GLSIZE(_max_x_id, _max_y_id, _max_z_id);
    Vector3i LOSIZE(_max_local_x_id, _max_local_y_id, _max_local_z_id);

    path_finder = new gridPathFinder(GLSIZE, LOSIZE);
    path_finder->initGridNodeMap(_resolution, _map_origin);

    Translation3d origin_translation( _map_origin(0), _map_origin(1), 0.0);
    Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
    Affine3d origin_transform = origin_translation * origin_rotation;
    collision_map = new CollisionMapGrid(origin_transform, "odom", _resolution, _x_size, _y_size, _z_size, _free_cell);

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

quadrotor_msgs::PolynomialTrajectory getBezierTraj()
{
      quadrotor_msgs::PolynomialTrajectory traj;
      traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_ADD;
      traj.num_segment = _seg_num;

      int order = _traj_order;
      int poly_num1d = order + 1;
      int polyTotalNum = _seg_num * (order + 1);

      traj.coef_x.resize(polyTotalNum);
      traj.coef_y.resize(polyTotalNum);
      traj.coef_z.resize(polyTotalNum);

      int idx = 0;
      for(int i = 0; i < _seg_num; i++ )
      {
          for(int j =0; j < poly_num1d; j++)
          {
              traj.coef_x[idx] = _bezier_coeff(i,                  j);
              traj.coef_y[idx] = _bezier_coeff(i,     poly_num1d + j);
              traj.coef_z[idx] = _bezier_coeff(i, 2 * poly_num1d + j);
              idx++;
          }
      }

      traj.header.frame_id = "/bernstein";
      traj.header.stamp = _odom.header.stamp;
      _start_time = traj.header.stamp;

      traj.time.resize(_seg_num);
      traj.order.resize(_seg_num);

      traj.mag_coeff = 1.0;
      for (int idx = 0; idx < _seg_num; ++idx){
          traj.time[idx] = _seg_time(idx);
          traj.order[idx] = _traj_order;
      }

      traj.start_yaw = 0.0;
      traj.final_yaw = 0.0;

      traj.trajectory_id = _traj_id;
      traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_ADD;

      return traj;
}

Vector3d getPosFromBezier(const MatrixXd & polyCoeff, double t_now, int seg_now )
{
    Vector3d ret = VectorXd::Zero(3);
    VectorXd ctrl_now = polyCoeff.row(seg_now);
    int ctrl_num1D = polyCoeff.cols() / 3;

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < ctrl_num1D; j++)
            ret(i) += _C(j) * ctrl_now(i * ctrl_num1D + j) * pow(t_now, j) * pow((1 - t_now), (_traj_order - j) );

    return ret;
}

VectorXd getStateFromBezier(const MatrixXd & polyCoeff, double t_now, int seg_now )
{
    VectorXd ret = VectorXd::Zero(12);

    VectorXd ctrl_now = polyCoeff.row(seg_now);
    int ctrl_num1D = polyCoeff.cols() / 3;

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < ctrl_num1D; j++){
            ret[i] += _C(j) * ctrl_now(i * ctrl_num1D + j) * pow(t_now, j) * pow((1 - t_now), (_traj_order - j) );

            if(j < ctrl_num1D - 1 )
                ret[i+3] += _Cv(j) * _traj_order
                      * ( ctrl_now(i * ctrl_num1D + j + 1) - ctrl_now(i * ctrl_num1D + j))
                      * pow(t_now, j) * pow((1 - t_now), (_traj_order - j - 1) );

            if(j < ctrl_num1D - 2 )
                ret[i+6] += _Ca(j) * _traj_order * (_traj_order - 1)
                      * ( ctrl_now(i * ctrl_num1D + j + 2) - 2 * ctrl_now(i * ctrl_num1D + j + 1) + ctrl_now(i * ctrl_num1D + j))
                      * pow(t_now, j) * pow((1 - t_now), (_traj_order - j - 2) );

            if(j < ctrl_num1D - 3 )
                ret[i+9] += _Cj(j) * _traj_order * (_traj_order - 1) * (_traj_order - 2)
                      * ( ctrl_now(i * ctrl_num1D + j + 3) - 3 * ctrl_now(i * ctrl_num1D + j + 2) + 3 * ctrl_now(i * ctrl_num1D + j + 1) - ctrl_now(i * ctrl_num1D + j))
                      * pow(t_now, j) * pow((1 - t_now), (_traj_order - j - 3) );
        }
    }

    return ret;
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
    mk.color.g = 1.0;
    mk.color.b = 1.0;

    int idx = 0;
    for(int i = 0; i < int(path.size()); i++)
    {
        mk.id = idx;

        mk.pose.position.x = path[i](0);
        mk.pose.position.y = path[i](1);
        mk.pose.position.z = path[i](2);

        mk.scale.x = _resolution;
        mk.scale.y = _resolution;
        mk.scale.z = _resolution;

        idx ++;
        path_vis.markers.push_back(mk);
    }

    _fm_path_vis_pub.publish(path_vis);
}

visualization_msgs::MarkerArray cube_vis;
void visCorridor(vector<Cube> corridor)
{
    for(auto & mk: cube_vis.markers)
        mk.action = visualization_msgs::Marker::DELETE;

    _corridor_vis_pub.publish(cube_vis);

    cube_vis.markers.clear();

    visualization_msgs::Marker mk;
    mk.header.frame_id = "odom";
    mk.header.stamp = ros::Time::now();
    mk.ns = "corridor";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;

    mk.color.a = 0.4;
    mk.color.r = 1.0;
    mk.color.g = 1.0;
    mk.color.b = 1.0;

    int idx = 0;
    for(int i = 0; i < int(corridor.size()); i++)
    {
        mk.id = idx;

        mk.pose.position.x = (corridor[i].vertex(0, 0) + corridor[i].vertex(3, 0) ) / 2.0;
        mk.pose.position.y = (corridor[i].vertex(0, 1) + corridor[i].vertex(1, 1) ) / 2.0;

        if(_is_proj_cube)
            mk.pose.position.z = 0.0;
        else
            mk.pose.position.z = (corridor[i].vertex(0, 2) + corridor[i].vertex(4, 2) ) / 2.0;

        mk.scale.x = (corridor[i].vertex(0, 0) - corridor[i].vertex(3, 0) );
        mk.scale.y = (corridor[i].vertex(1, 1) - corridor[i].vertex(0, 1) );

        if(_is_proj_cube)
            mk.scale.z = 0.05;
        else
            mk.scale.z = (corridor[i].vertex(0, 2) - corridor[i].vertex(4, 2) );

        idx ++;
        cube_vis.markers.push_back(mk);
    }

    _corridor_vis_pub.publish(cube_vis);
}

void visBezierTrajectory(MatrixXd polyCoeff, VectorXd time)
{
    visualization_msgs::Marker traj_vis;

    traj_vis.header.stamp       = ros::Time::now();
    traj_vis.header.frame_id    = "odom";

    traj_vis.ns = "trajectory/trajectory";
    traj_vis.id = 0;
    traj_vis.type = visualization_msgs::Marker::SPHERE_LIST;

    traj_vis.action = visualization_msgs::Marker::DELETE;
    _checkTraj_vis_pub.publish(traj_vis);
    _stopTraj_vis_pub.publish(traj_vis);

    traj_vis.action = visualization_msgs::Marker::ADD;
    traj_vis.scale.x = _vis_traj_width;
    traj_vis.scale.y = _vis_traj_width;
    traj_vis.scale.z = _vis_traj_width;
    traj_vis.pose.orientation.x = 0.0;
    traj_vis.pose.orientation.y = 0.0;
    traj_vis.pose.orientation.z = 0.0;
    traj_vis.pose.orientation.w = 1.0;
    traj_vis.color.r = 1.0;
    traj_vis.color.g = 0.0;
    traj_vis.color.b = 0.0;
    traj_vis.color.a = 0.6;

    double traj_len = 0.0;
    int count = 0;
    Vector3d cur, pre;
    cur.setZero();
    pre.setZero();
    
    traj_vis.points.clear();

    Vector3d state;
    geometry_msgs::Point pt;

    int segment_num  = polyCoeff.rows();
    for(int i = 0; i < segment_num; i++ ){
        for (double t = 0.0; t < 1.0; t += 0.05 / time(i), count += 1){
            state = getPosFromBezier( polyCoeff, t, i );
            cur(0) = pt.x = time(i) * state(0);
            cur(1) = pt.y = time(i) * state(1);
            cur(2) = pt.z = time(i) * state(2);
            traj_vis.points.push_back(pt);

            if (count) traj_len += (pre - cur).norm();
            pre = cur;
        }
    }

    ROS_INFO("[GENERATOR] The length of the trajectory; %.3lfm.", traj_len);
    _traj_vis_pub.publish(traj_vis);
}

visualization_msgs::MarkerArray grid_vis; 
void visGridPath( vector<Vector3d> grid_path )
{
    for(auto & mk: grid_vis.markers)
        mk.action = visualization_msgs::Marker::DELETE;

    _grid_path_vis_pub.publish(grid_vis);
    grid_vis.markers.clear();

    visualization_msgs::Marker mk;
    mk.header.frame_id = "odom";
    mk.header.stamp = ros::Time::now();
    mk.ns = "b_traj/grid_path";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.a = 1.0;
    mk.color.r = 1.0;
    mk.color.g = 0.0;
    mk.color.b = 0.0;

    int idx = 0;
    for(int i = 0; i < int(grid_path.size()); i++)
    {
        mk.id = idx;

        mk.pose.position.x = grid_path[i](0);
        mk.pose.position.y = grid_path[i](1);
        mk.pose.position.z = grid_path[i](2);

        mk.scale.x = _resolution;
        mk.scale.y = _resolution;
        mk.scale.z = _resolution;

        idx ++;
        grid_vis.markers.push_back(mk);
    }

    _grid_path_vis_pub.publish(grid_vis);
}

void visExpNode( vector<GridNodePtr> nodes )
{
    visualization_msgs::Marker node_vis;
    node_vis.header.frame_id = "odom";
    node_vis.header.stamp = ros::Time::now();
    node_vis.ns = "b_traj/visited_nodes";
    node_vis.type = visualization_msgs::Marker::CUBE_LIST;
    node_vis.action = visualization_msgs::Marker::ADD;
    node_vis.id = 0;

    node_vis.pose.orientation.x = 0.0;
    node_vis.pose.orientation.y = 0.0;
    node_vis.pose.orientation.z = 0.0;
    node_vis.pose.orientation.w = 1.0;
    node_vis.color.a = 0.3;
    node_vis.color.r = 0.0;
    node_vis.color.g = 1.0;
    node_vis.color.b = 0.0;

    node_vis.scale.x = _resolution;
    node_vis.scale.y = _resolution;
    node_vis.scale.z = _resolution;

    geometry_msgs::Point pt;
    for(int i = 0; i < int(nodes.size()); i++)
    {
        Vector3d coord = nodes[i]->coord;
        pt.x = coord(0);
        pt.y = coord(1);
        pt.z = coord(2);

        node_vis.points.push_back(pt);
    }

    _nodes_vis_pub.publish(node_vis);
}
