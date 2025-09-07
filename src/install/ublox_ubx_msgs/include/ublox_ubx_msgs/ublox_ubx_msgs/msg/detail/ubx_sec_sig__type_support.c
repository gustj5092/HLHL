// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ublox_ubx_msgs:msg/UBXSecSig.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ublox_ubx_msgs/msg/detail/ubx_sec_sig__rosidl_typesupport_introspection_c.h"
#include "ublox_ubx_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ublox_ubx_msgs/msg/detail/ubx_sec_sig__functions.h"
#include "ublox_ubx_msgs/msg/detail/ubx_sec_sig__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `jam_state_cent_freqs`
#include "ublox_ubx_msgs/msg/jam_state_cent_freq.h"
// Member `jam_state_cent_freqs`
#include "ublox_ubx_msgs/msg/detail/jam_state_cent_freq__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ublox_ubx_msgs__msg__UBXSecSig__init(message_memory);
}

void ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_fini_function(void * message_memory)
{
  ublox_ubx_msgs__msg__UBXSecSig__fini(message_memory);
}

size_t ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__size_function__UBXSecSig__jam_state_cent_freqs(
  const void * untyped_member)
{
  const ublox_ubx_msgs__msg__JamStateCentFreq__Sequence * member =
    (const ublox_ubx_msgs__msg__JamStateCentFreq__Sequence *)(untyped_member);
  return member->size;
}

const void * ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__get_const_function__UBXSecSig__jam_state_cent_freqs(
  const void * untyped_member, size_t index)
{
  const ublox_ubx_msgs__msg__JamStateCentFreq__Sequence * member =
    (const ublox_ubx_msgs__msg__JamStateCentFreq__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__get_function__UBXSecSig__jam_state_cent_freqs(
  void * untyped_member, size_t index)
{
  ublox_ubx_msgs__msg__JamStateCentFreq__Sequence * member =
    (ublox_ubx_msgs__msg__JamStateCentFreq__Sequence *)(untyped_member);
  return &member->data[index];
}

void ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__fetch_function__UBXSecSig__jam_state_cent_freqs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const ublox_ubx_msgs__msg__JamStateCentFreq * item =
    ((const ublox_ubx_msgs__msg__JamStateCentFreq *)
    ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__get_const_function__UBXSecSig__jam_state_cent_freqs(untyped_member, index));
  ublox_ubx_msgs__msg__JamStateCentFreq * value =
    (ublox_ubx_msgs__msg__JamStateCentFreq *)(untyped_value);
  *value = *item;
}

void ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__assign_function__UBXSecSig__jam_state_cent_freqs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  ublox_ubx_msgs__msg__JamStateCentFreq * item =
    ((ublox_ubx_msgs__msg__JamStateCentFreq *)
    ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__get_function__UBXSecSig__jam_state_cent_freqs(untyped_member, index));
  const ublox_ubx_msgs__msg__JamStateCentFreq * value =
    (const ublox_ubx_msgs__msg__JamStateCentFreq *)(untyped_value);
  *item = *value;
}

bool ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__resize_function__UBXSecSig__jam_state_cent_freqs(
  void * untyped_member, size_t size)
{
  ublox_ubx_msgs__msg__JamStateCentFreq__Sequence * member =
    (ublox_ubx_msgs__msg__JamStateCentFreq__Sequence *)(untyped_member);
  ublox_ubx_msgs__msg__JamStateCentFreq__Sequence__fini(member);
  return ublox_ubx_msgs__msg__JamStateCentFreq__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_member_array[8] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs__msg__UBXSecSig, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "version",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs__msg__UBXSecSig, version),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "jam_det_enabled",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs__msg__UBXSecSig, jam_det_enabled),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "jamming_state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs__msg__UBXSecSig, jamming_state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "spf_det_enabled",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs__msg__UBXSecSig, spf_det_enabled),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "spoofing_state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs__msg__UBXSecSig, spoofing_state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "jam_num_cent_freqs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs__msg__UBXSecSig, jam_num_cent_freqs),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "jam_state_cent_freqs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs__msg__UBXSecSig, jam_state_cent_freqs),  // bytes offset in struct
    NULL,  // default value
    ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__size_function__UBXSecSig__jam_state_cent_freqs,  // size() function pointer
    ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__get_const_function__UBXSecSig__jam_state_cent_freqs,  // get_const(index) function pointer
    ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__get_function__UBXSecSig__jam_state_cent_freqs,  // get(index) function pointer
    ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__fetch_function__UBXSecSig__jam_state_cent_freqs,  // fetch(index, &value) function pointer
    ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__assign_function__UBXSecSig__jam_state_cent_freqs,  // assign(index, value) function pointer
    ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__resize_function__UBXSecSig__jam_state_cent_freqs  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_members = {
  "ublox_ubx_msgs__msg",  // message namespace
  "UBXSecSig",  // message name
  8,  // number of fields
  sizeof(ublox_ubx_msgs__msg__UBXSecSig),
  ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_member_array,  // message members
  ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_init_function,  // function to initialize message memory (memory has to be allocated)
  ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_type_support_handle = {
  0,
  &ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ublox_ubx_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ublox_ubx_msgs, msg, UBXSecSig)() {
  ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_member_array[7].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ublox_ubx_msgs, msg, JamStateCentFreq)();
  if (!ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_type_support_handle.typesupport_identifier) {
    ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ublox_ubx_msgs__msg__UBXSecSig__rosidl_typesupport_introspection_c__UBXSecSig_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
