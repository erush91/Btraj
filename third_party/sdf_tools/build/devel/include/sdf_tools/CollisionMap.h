// Generated by gencpp from file sdf_tools/CollisionMap.msg
// DO NOT EDIT!


#ifndef SDF_TOOLS_MESSAGE_COLLISIONMAP_H
#define SDF_TOOLS_MESSAGE_COLLISIONMAP_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/Vector3.h>

namespace sdf_tools
{
template <class ContainerAllocator>
struct CollisionMap_
{
  typedef CollisionMap_<ContainerAllocator> Type;

  CollisionMap_()
    : header()
    , origin_transform()
    , dimensions()
    , cell_size(0.0)
    , OOB_occupancy_value(0.0)
    , OOB_component_value(0)
    , number_of_components(0)
    , components_valid(false)
    , initialized(false)
    , data()  {
    }
  CollisionMap_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , origin_transform(_alloc)
    , dimensions(_alloc)
    , cell_size(0.0)
    , OOB_occupancy_value(0.0)
    , OOB_component_value(0)
    , number_of_components(0)
    , components_valid(false)
    , initialized(false)
    , data(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef  ::geometry_msgs::Transform_<ContainerAllocator>  _origin_transform_type;
  _origin_transform_type origin_transform;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _dimensions_type;
  _dimensions_type dimensions;

   typedef double _cell_size_type;
  _cell_size_type cell_size;

   typedef float _OOB_occupancy_value_type;
  _OOB_occupancy_value_type OOB_occupancy_value;

   typedef uint32_t _OOB_component_value_type;
  _OOB_component_value_type OOB_component_value;

   typedef uint32_t _number_of_components_type;
  _number_of_components_type number_of_components;

   typedef uint8_t _components_valid_type;
  _components_valid_type components_valid;

   typedef uint8_t _initialized_type;
  _initialized_type initialized;

   typedef std::vector<uint8_t, typename ContainerAllocator::template rebind<uint8_t>::other >  _data_type;
  _data_type data;





  typedef boost::shared_ptr< ::sdf_tools::CollisionMap_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::sdf_tools::CollisionMap_<ContainerAllocator> const> ConstPtr;

}; // struct CollisionMap_

typedef ::sdf_tools::CollisionMap_<std::allocator<void> > CollisionMap;

typedef boost::shared_ptr< ::sdf_tools::CollisionMap > CollisionMapPtr;
typedef boost::shared_ptr< ::sdf_tools::CollisionMap const> CollisionMapConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::sdf_tools::CollisionMap_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::sdf_tools::CollisionMap_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace sdf_tools

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'std_msgs': ['/opt/ros/melodic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/melodic/share/geometry_msgs/cmake/../msg'], 'sdf_tools': ['/home/erush91/catkin_ws/src/Btraj/third_party/sdf_tools/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::sdf_tools::CollisionMap_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::sdf_tools::CollisionMap_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::sdf_tools::CollisionMap_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::sdf_tools::CollisionMap_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::sdf_tools::CollisionMap_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::sdf_tools::CollisionMap_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::sdf_tools::CollisionMap_<ContainerAllocator> >
{
  static const char* value()
  {
    return "69b7e5097be57c5900575a10000bd373";
  }

  static const char* value(const ::sdf_tools::CollisionMap_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x69b7e5097be57c59ULL;
  static const uint64_t static_value2 = 0x00575a10000bd373ULL;
};

template<class ContainerAllocator>
struct DataType< ::sdf_tools::CollisionMap_<ContainerAllocator> >
{
  static const char* value()
  {
    return "sdf_tools/CollisionMap";
  }

  static const char* value(const ::sdf_tools::CollisionMap_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::sdf_tools::CollisionMap_<ContainerAllocator> >
{
  static const char* value()
  {
    return "std_msgs/Header header\n"
"geometry_msgs/Transform origin_transform\n"
"geometry_msgs/Vector3 dimensions\n"
"float64 cell_size\n"
"float32 OOB_occupancy_value\n"
"uint32 OOB_component_value\n"
"uint32 number_of_components\n"
"bool components_valid\n"
"bool initialized\n"
"uint8[] data\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Transform\n"
"# This represents the transform between two coordinate frames in free space.\n"
"\n"
"Vector3 translation\n"
"Quaternion rotation\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Vector3\n"
"# This represents a vector in free space. \n"
"# It is only meant to represent a direction. Therefore, it does not\n"
"# make sense to apply a translation to it (e.g., when applying a \n"
"# generic rigid transformation to a Vector3, tf2 will only apply the\n"
"# rotation). If you want your data to be translatable too, use the\n"
"# geometry_msgs/Point message instead.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
;
  }

  static const char* value(const ::sdf_tools::CollisionMap_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::sdf_tools::CollisionMap_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.origin_transform);
      stream.next(m.dimensions);
      stream.next(m.cell_size);
      stream.next(m.OOB_occupancy_value);
      stream.next(m.OOB_component_value);
      stream.next(m.number_of_components);
      stream.next(m.components_valid);
      stream.next(m.initialized);
      stream.next(m.data);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct CollisionMap_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::sdf_tools::CollisionMap_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::sdf_tools::CollisionMap_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "origin_transform: ";
    s << std::endl;
    Printer< ::geometry_msgs::Transform_<ContainerAllocator> >::stream(s, indent + "  ", v.origin_transform);
    s << indent << "dimensions: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.dimensions);
    s << indent << "cell_size: ";
    Printer<double>::stream(s, indent + "  ", v.cell_size);
    s << indent << "OOB_occupancy_value: ";
    Printer<float>::stream(s, indent + "  ", v.OOB_occupancy_value);
    s << indent << "OOB_component_value: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.OOB_component_value);
    s << indent << "number_of_components: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.number_of_components);
    s << indent << "components_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.components_valid);
    s << indent << "initialized: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.initialized);
    s << indent << "data[]" << std::endl;
    for (size_t i = 0; i < v.data.size(); ++i)
    {
      s << indent << "  data[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.data[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // SDF_TOOLS_MESSAGE_COLLISIONMAP_H
