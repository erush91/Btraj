// Auto-generated. Do not edit!

// (in-package sdf_tools.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class SDF {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.origin_transform = null;
      this.dimensions = null;
      this.sdf_cell_size = null;
      this.OOB_value = null;
      this.initialized = null;
      this.locked = null;
      this.data = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('origin_transform')) {
        this.origin_transform = initObj.origin_transform
      }
      else {
        this.origin_transform = new geometry_msgs.msg.Transform();
      }
      if (initObj.hasOwnProperty('dimensions')) {
        this.dimensions = initObj.dimensions
      }
      else {
        this.dimensions = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('sdf_cell_size')) {
        this.sdf_cell_size = initObj.sdf_cell_size
      }
      else {
        this.sdf_cell_size = 0.0;
      }
      if (initObj.hasOwnProperty('OOB_value')) {
        this.OOB_value = initObj.OOB_value
      }
      else {
        this.OOB_value = 0.0;
      }
      if (initObj.hasOwnProperty('initialized')) {
        this.initialized = initObj.initialized
      }
      else {
        this.initialized = false;
      }
      if (initObj.hasOwnProperty('locked')) {
        this.locked = initObj.locked
      }
      else {
        this.locked = false;
      }
      if (initObj.hasOwnProperty('data')) {
        this.data = initObj.data
      }
      else {
        this.data = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SDF
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [origin_transform]
    bufferOffset = geometry_msgs.msg.Transform.serialize(obj.origin_transform, buffer, bufferOffset);
    // Serialize message field [dimensions]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.dimensions, buffer, bufferOffset);
    // Serialize message field [sdf_cell_size]
    bufferOffset = _serializer.float64(obj.sdf_cell_size, buffer, bufferOffset);
    // Serialize message field [OOB_value]
    bufferOffset = _serializer.float32(obj.OOB_value, buffer, bufferOffset);
    // Serialize message field [initialized]
    bufferOffset = _serializer.bool(obj.initialized, buffer, bufferOffset);
    // Serialize message field [locked]
    bufferOffset = _serializer.bool(obj.locked, buffer, bufferOffset);
    // Serialize message field [data]
    bufferOffset = _arraySerializer.uint8(obj.data, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SDF
    let len;
    let data = new SDF(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [origin_transform]
    data.origin_transform = geometry_msgs.msg.Transform.deserialize(buffer, bufferOffset);
    // Deserialize message field [dimensions]
    data.dimensions = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [sdf_cell_size]
    data.sdf_cell_size = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [OOB_value]
    data.OOB_value = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [initialized]
    data.initialized = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [locked]
    data.locked = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [data]
    data.data = _arrayDeserializer.uint8(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += object.data.length;
    return length + 98;
  }

  static datatype() {
    // Returns string type for a message object
    return 'sdf_tools/SDF';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a7c1a3fc5ebce39f4d97049e22031ffc';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Header header
    geometry_msgs/Transform origin_transform
    geometry_msgs/Vector3 dimensions
    float64 sdf_cell_size
    float32 OOB_value
    bool initialized
    bool locked
    uint8[] data
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: geometry_msgs/Transform
    # This represents the transform between two coordinate frames in free space.
    
    Vector3 translation
    Quaternion rotation
    
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
    float64 x
    float64 y
    float64 z
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SDF(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.origin_transform !== undefined) {
      resolved.origin_transform = geometry_msgs.msg.Transform.Resolve(msg.origin_transform)
    }
    else {
      resolved.origin_transform = new geometry_msgs.msg.Transform()
    }

    if (msg.dimensions !== undefined) {
      resolved.dimensions = geometry_msgs.msg.Vector3.Resolve(msg.dimensions)
    }
    else {
      resolved.dimensions = new geometry_msgs.msg.Vector3()
    }

    if (msg.sdf_cell_size !== undefined) {
      resolved.sdf_cell_size = msg.sdf_cell_size;
    }
    else {
      resolved.sdf_cell_size = 0.0
    }

    if (msg.OOB_value !== undefined) {
      resolved.OOB_value = msg.OOB_value;
    }
    else {
      resolved.OOB_value = 0.0
    }

    if (msg.initialized !== undefined) {
      resolved.initialized = msg.initialized;
    }
    else {
      resolved.initialized = false
    }

    if (msg.locked !== undefined) {
      resolved.locked = msg.locked;
    }
    else {
      resolved.locked = false
    }

    if (msg.data !== undefined) {
      resolved.data = msg.data;
    }
    else {
      resolved.data = []
    }

    return resolved;
    }
};

module.exports = SDF;
