// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from ublox_ubx_msgs:msg/UBXSecSig.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "ublox_ubx_msgs/msg/detail/ubx_sec_sig__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace ublox_ubx_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void UBXSecSig_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) ublox_ubx_msgs::msg::UBXSecSig(_init);
}

void UBXSecSig_fini_function(void * message_memory)
{
  auto typed_message = static_cast<ublox_ubx_msgs::msg::UBXSecSig *>(message_memory);
  typed_message->~UBXSecSig();
}

size_t size_function__UBXSecSig__jam_state_cent_freqs(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ublox_ubx_msgs::msg::JamStateCentFreq> *>(untyped_member);
  return member->size();
}

const void * get_const_function__UBXSecSig__jam_state_cent_freqs(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ublox_ubx_msgs::msg::JamStateCentFreq> *>(untyped_member);
  return &member[index];
}

void * get_function__UBXSecSig__jam_state_cent_freqs(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ublox_ubx_msgs::msg::JamStateCentFreq> *>(untyped_member);
  return &member[index];
}

void fetch_function__UBXSecSig__jam_state_cent_freqs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ublox_ubx_msgs::msg::JamStateCentFreq *>(
    get_const_function__UBXSecSig__jam_state_cent_freqs(untyped_member, index));
  auto & value = *reinterpret_cast<ublox_ubx_msgs::msg::JamStateCentFreq *>(untyped_value);
  value = item;
}

void assign_function__UBXSecSig__jam_state_cent_freqs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ublox_ubx_msgs::msg::JamStateCentFreq *>(
    get_function__UBXSecSig__jam_state_cent_freqs(untyped_member, index));
  const auto & value = *reinterpret_cast<const ublox_ubx_msgs::msg::JamStateCentFreq *>(untyped_value);
  item = value;
}

void resize_function__UBXSecSig__jam_state_cent_freqs(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ublox_ubx_msgs::msg::JamStateCentFreq> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember UBXSecSig_message_member_array[8] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXSecSig, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "version",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXSecSig, version),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "jam_det_enabled",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXSecSig, jam_det_enabled),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "jamming_state",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXSecSig, jamming_state),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "spf_det_enabled",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXSecSig, spf_det_enabled),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "spoofing_state",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXSecSig, spoofing_state),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "jam_num_cent_freqs",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXSecSig, jam_num_cent_freqs),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "jam_state_cent_freqs",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ublox_ubx_msgs::msg::JamStateCentFreq>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXSecSig, jam_state_cent_freqs),  // bytes offset in struct
    nullptr,  // default value
    size_function__UBXSecSig__jam_state_cent_freqs,  // size() function pointer
    get_const_function__UBXSecSig__jam_state_cent_freqs,  // get_const(index) function pointer
    get_function__UBXSecSig__jam_state_cent_freqs,  // get(index) function pointer
    fetch_function__UBXSecSig__jam_state_cent_freqs,  // fetch(index, &value) function pointer
    assign_function__UBXSecSig__jam_state_cent_freqs,  // assign(index, value) function pointer
    resize_function__UBXSecSig__jam_state_cent_freqs  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers UBXSecSig_message_members = {
  "ublox_ubx_msgs::msg",  // message namespace
  "UBXSecSig",  // message name
  8,  // number of fields
  sizeof(ublox_ubx_msgs::msg::UBXSecSig),
  UBXSecSig_message_member_array,  // message members
  UBXSecSig_init_function,  // function to initialize message memory (memory has to be allocated)
  UBXSecSig_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t UBXSecSig_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &UBXSecSig_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace ublox_ubx_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<ublox_ubx_msgs::msg::UBXSecSig>()
{
  return &::ublox_ubx_msgs::msg::rosidl_typesupport_introspection_cpp::UBXSecSig_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, ublox_ubx_msgs, msg, UBXSecSig)() {
  return &::ublox_ubx_msgs::msg::rosidl_typesupport_introspection_cpp::UBXSecSig_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
