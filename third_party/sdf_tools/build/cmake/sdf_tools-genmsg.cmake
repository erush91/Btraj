# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "sdf_tools: 2 messages, 0 services")

set(MSG_I_FLAGS "-Isdf_tools:/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg;-Igeometry_msgs:/opt/ros/melodic/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(sdf_tools_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg" NAME_WE)
add_custom_target(_sdf_tools_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "sdf_tools" "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg" "geometry_msgs/Vector3:geometry_msgs/Transform:geometry_msgs/Quaternion:std_msgs/Header"
)

get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg" NAME_WE)
add_custom_target(_sdf_tools_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "sdf_tools" "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg" "geometry_msgs/Vector3:geometry_msgs/Transform:geometry_msgs/Quaternion:std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sdf_tools
)
_generate_msg_cpp(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sdf_tools
)

### Generating Services

### Generating Module File
_generate_module_cpp(sdf_tools
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sdf_tools
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(sdf_tools_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(sdf_tools_generate_messages sdf_tools_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_cpp _sdf_tools_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_cpp _sdf_tools_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sdf_tools_gencpp)
add_dependencies(sdf_tools_gencpp sdf_tools_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sdf_tools_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sdf_tools
)
_generate_msg_eus(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sdf_tools
)

### Generating Services

### Generating Module File
_generate_module_eus(sdf_tools
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sdf_tools
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(sdf_tools_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(sdf_tools_generate_messages sdf_tools_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_eus _sdf_tools_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_eus _sdf_tools_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sdf_tools_geneus)
add_dependencies(sdf_tools_geneus sdf_tools_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sdf_tools_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sdf_tools
)
_generate_msg_lisp(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sdf_tools
)

### Generating Services

### Generating Module File
_generate_module_lisp(sdf_tools
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sdf_tools
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(sdf_tools_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(sdf_tools_generate_messages sdf_tools_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_lisp _sdf_tools_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_lisp _sdf_tools_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sdf_tools_genlisp)
add_dependencies(sdf_tools_genlisp sdf_tools_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sdf_tools_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sdf_tools
)
_generate_msg_nodejs(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sdf_tools
)

### Generating Services

### Generating Module File
_generate_module_nodejs(sdf_tools
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sdf_tools
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(sdf_tools_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(sdf_tools_generate_messages sdf_tools_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_nodejs _sdf_tools_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_nodejs _sdf_tools_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sdf_tools_gennodejs)
add_dependencies(sdf_tools_gennodejs sdf_tools_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sdf_tools_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sdf_tools
)
_generate_msg_py(sdf_tools
  "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/melodic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sdf_tools
)

### Generating Services

### Generating Module File
_generate_module_py(sdf_tools
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sdf_tools
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(sdf_tools_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(sdf_tools_generate_messages sdf_tools_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/SDF.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_py _sdf_tools_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg/CollisionMap.msg" NAME_WE)
add_dependencies(sdf_tools_generate_messages_py _sdf_tools_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sdf_tools_genpy)
add_dependencies(sdf_tools_genpy sdf_tools_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sdf_tools_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sdf_tools)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sdf_tools
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(sdf_tools_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(sdf_tools_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sdf_tools)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sdf_tools
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(sdf_tools_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(sdf_tools_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sdf_tools)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sdf_tools
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(sdf_tools_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(sdf_tools_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sdf_tools)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sdf_tools
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(sdf_tools_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(sdf_tools_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sdf_tools)
  install(CODE "execute_process(COMMAND \"/usr/bin/python2\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sdf_tools\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sdf_tools
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(sdf_tools_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(sdf_tools_generate_messages_py std_msgs_generate_messages_py)
endif()
