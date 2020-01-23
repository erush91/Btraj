cmake_minimum_required(VERSION 2.8.3)
project(fmm_planner)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/sdf_tools)

find_package(catkin REQUIRED COMPONENTS 
    roscpp 
    std_msgs
    nav_msgs
    visualization_msgs 
    quadrotor_msgs
    tf
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

catkin_package(
  INCLUDE_DIRS include
)


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/sdf_tools/include)

include_directories(
    include 
    SYSTEM 
    third_party
    ${catkin_INCLUDE_DIRS} 
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -O3 -Wall") # -Wextra -Werror

add_executable(
    fmm_planner_node 
    src/fmm_planner_node.cpp
    third_party/fast_methods/console/console.cpp
    third_party/fast_methods/fm/fmdata/fmcell.cpp
    third_party/fast_methods/ndgridmap/cell.cpp
)

target_link_libraries(fmm_planner_node ${catkin_LIBRARIES} sdf_tools ${PCL_LIBRARIES})


