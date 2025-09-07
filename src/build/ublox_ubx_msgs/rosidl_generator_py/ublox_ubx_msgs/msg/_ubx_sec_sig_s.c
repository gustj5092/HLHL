// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from ublox_ubx_msgs:msg/UBXSecSig.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "ublox_ubx_msgs/msg/detail/ubx_sec_sig__struct.h"
#include "ublox_ubx_msgs/msg/detail/ubx_sec_sig__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"

// Nested array functions includes
#include "ublox_ubx_msgs/msg/detail/jam_state_cent_freq__functions.h"
// end nested array functions include
ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);
bool ublox_ubx_msgs__msg__jam_state_cent_freq__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * ublox_ubx_msgs__msg__jam_state_cent_freq__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool ublox_ubx_msgs__msg__ubx_sec_sig__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[42];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("ublox_ubx_msgs.msg._ubx_sec_sig.UBXSecSig", full_classname_dest, 41) == 0);
  }
  ublox_ubx_msgs__msg__UBXSecSig * ros_message = _ros_message;
  {  // header
    PyObject * field = PyObject_GetAttrString(_pymsg, "header");
    if (!field) {
      return false;
    }
    if (!std_msgs__msg__header__convert_from_py(field, &ros_message->header)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // version
    PyObject * field = PyObject_GetAttrString(_pymsg, "version");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->version = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // jam_det_enabled
    PyObject * field = PyObject_GetAttrString(_pymsg, "jam_det_enabled");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->jam_det_enabled = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // jamming_state
    PyObject * field = PyObject_GetAttrString(_pymsg, "jamming_state");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->jamming_state = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // spf_det_enabled
    PyObject * field = PyObject_GetAttrString(_pymsg, "spf_det_enabled");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->spf_det_enabled = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // spoofing_state
    PyObject * field = PyObject_GetAttrString(_pymsg, "spoofing_state");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->spoofing_state = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // jam_num_cent_freqs
    PyObject * field = PyObject_GetAttrString(_pymsg, "jam_num_cent_freqs");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->jam_num_cent_freqs = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // jam_state_cent_freqs
    PyObject * field = PyObject_GetAttrString(_pymsg, "jam_state_cent_freqs");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'jam_state_cent_freqs'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!ublox_ubx_msgs__msg__JamStateCentFreq__Sequence__init(&(ros_message->jam_state_cent_freqs), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create ublox_ubx_msgs__msg__JamStateCentFreq__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    ublox_ubx_msgs__msg__JamStateCentFreq * dest = ros_message->jam_state_cent_freqs.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!ublox_ubx_msgs__msg__jam_state_cent_freq__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * ublox_ubx_msgs__msg__ubx_sec_sig__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of UBXSecSig */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("ublox_ubx_msgs.msg._ubx_sec_sig");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "UBXSecSig");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  ublox_ubx_msgs__msg__UBXSecSig * ros_message = (ublox_ubx_msgs__msg__UBXSecSig *)raw_ros_message;
  {  // header
    PyObject * field = NULL;
    field = std_msgs__msg__header__convert_to_py(&ros_message->header);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "header", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // version
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->version);
    {
      int rc = PyObject_SetAttrString(_pymessage, "version", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // jam_det_enabled
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->jam_det_enabled);
    {
      int rc = PyObject_SetAttrString(_pymessage, "jam_det_enabled", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // jamming_state
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->jamming_state);
    {
      int rc = PyObject_SetAttrString(_pymessage, "jamming_state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // spf_det_enabled
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->spf_det_enabled);
    {
      int rc = PyObject_SetAttrString(_pymessage, "spf_det_enabled", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // spoofing_state
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->spoofing_state);
    {
      int rc = PyObject_SetAttrString(_pymessage, "spoofing_state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // jam_num_cent_freqs
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->jam_num_cent_freqs);
    {
      int rc = PyObject_SetAttrString(_pymessage, "jam_num_cent_freqs", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // jam_state_cent_freqs
    PyObject * field = NULL;
    size_t size = ros_message->jam_state_cent_freqs.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    ublox_ubx_msgs__msg__JamStateCentFreq * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->jam_state_cent_freqs.data[i]);
      PyObject * pyitem = ublox_ubx_msgs__msg__jam_state_cent_freq__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "jam_state_cent_freqs", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
