// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from ublox_ubx_msgs:msg/UBXRxmSpartnKey.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "ublox_ubx_msgs/msg/detail/ubx_rxm_spartn_key__struct.hpp"
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

void UBXRxmSpartnKey_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) ublox_ubx_msgs::msg::UBXRxmSpartnKey(_init);
}

void UBXRxmSpartnKey_fini_function(void * message_memory)
{
  auto typed_message = static_cast<ublox_ubx_msgs::msg::UBXRxmSpartnKey *>(message_memory);
  typed_message->~UBXRxmSpartnKey();
}

size_t size_function__UBXRxmSpartnKey__reserved0(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__UBXRxmSpartnKey__reserved0(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<uint8_t, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__UBXRxmSpartnKey__reserved0(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<uint8_t, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__UBXRxmSpartnKey__reserved0(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint8_t *>(
    get_const_function__UBXRxmSpartnKey__reserved0(untyped_member, index));
  auto & value = *reinterpret_cast<uint8_t *>(untyped_value);
  value = item;
}

void assign_function__UBXRxmSpartnKey__reserved0(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint8_t *>(
    get_function__UBXRxmSpartnKey__reserved0(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint8_t *>(untyped_value);
  item = value;
}

size_t size_function__UBXRxmSpartnKey__key_info(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ublox_ubx_msgs::msg::SpartnKeyInfo> *>(untyped_member);
  return member->size();
}

const void * get_const_function__UBXRxmSpartnKey__key_info(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ublox_ubx_msgs::msg::SpartnKeyInfo> *>(untyped_member);
  return &member[index];
}

void * get_function__UBXRxmSpartnKey__key_info(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ublox_ubx_msgs::msg::SpartnKeyInfo> *>(untyped_member);
  return &member[index];
}

void fetch_function__UBXRxmSpartnKey__key_info(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ublox_ubx_msgs::msg::SpartnKeyInfo *>(
    get_const_function__UBXRxmSpartnKey__key_info(untyped_member, index));
  auto & value = *reinterpret_cast<ublox_ubx_msgs::msg::SpartnKeyInfo *>(untyped_value);
  value = item;
}

void assign_function__UBXRxmSpartnKey__key_info(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ublox_ubx_msgs::msg::SpartnKeyInfo *>(
    get_function__UBXRxmSpartnKey__key_info(untyped_member, index));
  const auto & value = *reinterpret_cast<const ublox_ubx_msgs::msg::SpartnKeyInfo *>(untyped_value);
  item = value;
}

void resize_function__UBXRxmSpartnKey__key_info(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ublox_ubx_msgs::msg::SpartnKeyInfo> *>(untyped_member);
  member->resize(size);
}

size_t size_function__UBXRxmSpartnKey__key_payload(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<uint8_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__UBXRxmSpartnKey__key_payload(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<uint8_t> *>(untyped_member);
  return &member[index];
}

void * get_function__UBXRxmSpartnKey__key_payload(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<uint8_t> *>(untyped_member);
  return &member[index];
}

void fetch_function__UBXRxmSpartnKey__key_payload(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint8_t *>(
    get_const_function__UBXRxmSpartnKey__key_payload(untyped_member, index));
  auto & value = *reinterpret_cast<uint8_t *>(untyped_value);
  value = item;
}

void assign_function__UBXRxmSpartnKey__key_payload(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint8_t *>(
    get_function__UBXRxmSpartnKey__key_payload(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint8_t *>(untyped_value);
  item = value;
}

void resize_function__UBXRxmSpartnKey__key_payload(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<uint8_t> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember UBXRxmSpartnKey_message_member_array[6] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXRxmSpartnKey, header),  // bytes offset in struct
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
    offsetof(ublox_ubx_msgs::msg::UBXRxmSpartnKey, version),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "num_keys",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXRxmSpartnKey, num_keys),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "reserved0",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXRxmSpartnKey, reserved0),  // bytes offset in struct
    nullptr,  // default value
    size_function__UBXRxmSpartnKey__reserved0,  // size() function pointer
    get_const_function__UBXRxmSpartnKey__reserved0,  // get_const(index) function pointer
    get_function__UBXRxmSpartnKey__reserved0,  // get(index) function pointer
    fetch_function__UBXRxmSpartnKey__reserved0,  // fetch(index, &value) function pointer
    assign_function__UBXRxmSpartnKey__reserved0,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "key_info",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ublox_ubx_msgs::msg::SpartnKeyInfo>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXRxmSpartnKey, key_info),  // bytes offset in struct
    nullptr,  // default value
    size_function__UBXRxmSpartnKey__key_info,  // size() function pointer
    get_const_function__UBXRxmSpartnKey__key_info,  // get_const(index) function pointer
    get_function__UBXRxmSpartnKey__key_info,  // get(index) function pointer
    fetch_function__UBXRxmSpartnKey__key_info,  // fetch(index, &value) function pointer
    assign_function__UBXRxmSpartnKey__key_info,  // assign(index, value) function pointer
    resize_function__UBXRxmSpartnKey__key_info  // resize(index) function pointer
  },
  {
    "key_payload",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ublox_ubx_msgs::msg::UBXRxmSpartnKey, key_payload),  // bytes offset in struct
    nullptr,  // default value
    size_function__UBXRxmSpartnKey__key_payload,  // size() function pointer
    get_const_function__UBXRxmSpartnKey__key_payload,  // get_const(index) function pointer
    get_function__UBXRxmSpartnKey__key_payload,  // get(index) function pointer
    fetch_function__UBXRxmSpartnKey__key_payload,  // fetch(index, &value) function pointer
    assign_function__UBXRxmSpartnKey__key_payload,  // assign(index, value) function pointer
    resize_function__UBXRxmSpartnKey__key_payload  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers UBXRxmSpartnKey_message_members = {
  "ublox_ubx_msgs::msg",  // message namespace
  "UBXRxmSpartnKey",  // message name
  6,  // number of fields
  sizeof(ublox_ubx_msgs::msg::UBXRxmSpartnKey),
  UBXRxmSpartnKey_message_member_array,  // message members
  UBXRxmSpartnKey_init_function,  // function to initialize message memory (memory has to be allocated)
  UBXRxmSpartnKey_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t UBXRxmSpartnKey_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &UBXRxmSpartnKey_message_members,
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
get_message_type_support_handle<ublox_ubx_msgs::msg::UBXRxmSpartnKey>()
{
  return &::ublox_ubx_msgs::msg::rosidl_typesupport_introspection_cpp::UBXRxmSpartnKey_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, ublox_ubx_msgs, msg, UBXRxmSpartnKey)() {
  return &::ublox_ubx_msgs::msg::rosidl_typesupport_introspection_cpp::UBXRxmSpartnKey_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
