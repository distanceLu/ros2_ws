#include "CameraApi.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace std::chrono_literals;

struct CameraDevice {
  int index = -1;
  std::string sn;
  int saturation = 84;
  int gamma = 64;
  int analoggain = 64;
  // int saturation = 0;
  // int gamma = 0;
  // int analoggain = 0;
  bool probe = false;

  int handle = -1;
  tSdkCameraCapbility capability{};
  tSdkFrameHead frame_info{};
  tSdkCameraDevInfo info{};
  BYTE *frame_buffer = nullptr;
  unsigned char *rgb_buffer = nullptr;
};

class CameraNode : public rclcpp::Node {
public:
  CameraNode() : Node("camera_node") {
    declare_parameter<std::string>("cfg_file", "");
    declare_parameter<int>("fre", 20);
    // declare_parameter<double>("exposure_time", 3.0);
    declare_parameter<double>("exposure_time", 4.3);
    declare_parameter<int>("max_devices", 3);
    declare_parameter<int>("default_saturation", 84);
    declare_parameter<int>("default_gamma", 64);
    declare_parameter<int>("default_analoggain", 64);
    // declare_parameter<int>("default_saturation", 10);
    // declare_parameter<int>("default_gamma", 10);
    // declare_parameter<int>("default_analoggain", 10);

    const auto cfg_file = get_parameter("cfg_file").as_string();
    frequency_hz_ = get_parameter("fre").as_int();
    exposure_time_ = get_parameter("exposure_time").as_double();
    max_devices_ = get_parameter("max_devices").as_int();
    default_saturation_ = get_parameter("default_saturation").as_int();
    default_gamma_ = get_parameter("default_gamma").as_int();
    default_analoggain_ = get_parameter("default_analoggain").as_int();

    if (frequency_hz_ <= 0) {
      RCLCPP_WARN(get_logger(), "fre must be > 0, fallback to 20");
      frequency_hz_ = 20;
    }

    CameraSdkInit(1);

    load_config(cfg_file);
    enumerate_and_bind();
    init_bound_cameras();

    if (cameras_.empty()) {
      RCLCPP_ERROR(get_logger(), "No camera initialized, node stays idle");
      return;
    }

    const auto period_ms = std::chrono::milliseconds(1000 / frequency_hz_);
    timer_ = create_wall_timer(period_ms, std::bind(&CameraNode::capture_and_publish, this));

    RCLCPP_INFO(get_logger(), "camera_node started, frequency=%dHz, camera_count=%zu", frequency_hz_, cameras_.size());
  }

  ~CameraNode() override {
    for (auto &cam : cameras_) {
      if (cam.handle != -1) {
        CameraUnInit(cam.handle);
      }
      if (cam.rgb_buffer != nullptr) {
        free(cam.rgb_buffer);
        cam.rgb_buffer = nullptr;
      }
    }
  }

private:
  void load_config(const std::string &cfg_file) {
    configured_.clear();

    YAML::Node config;
    if (!cfg_file.empty()) {
      try {
        config = YAML::LoadFile(cfg_file);
        RCLCPP_INFO(get_logger(), "Using config file: %s", cfg_file.c_str());
      } catch (const std::exception &e) {
        RCLCPP_WARN(get_logger(), "Failed to load config '%s': %s", cfg_file.c_str(), e.what());
      }
    }

    for (int i = 0; i < max_devices_; ++i) {
      const std::string prefix = "Camera" + std::to_string(i);
      const std::string sn_key = prefix + "_sn";

      CameraDevice cam;
      cam.index = i;
      cam.saturation = default_saturation_;
      cam.gamma = default_gamma_;
      cam.analoggain = default_analoggain_;

      if (config[sn_key]) {
        cam.sn = config[sn_key].as<std::string>();
      }

      const std::string sat_key = prefix + "_saturation";
      const std::string gamma_key = prefix + "_gamma";
      const std::string gain_key = prefix + "_analoggain";
      if (config[sat_key]) {
        cam.saturation = config[sat_key].as<int>();
      }
      if (config[gamma_key]) {
        cam.gamma = config[gamma_key].as<int>();
      }
      if (config[gain_key]) {
        cam.analoggain = config[gain_key].as<int>();
      }

      configured_.push_back(cam);
    }
  }

  void enumerate_and_bind() {
    tSdkCameraDevInfo enum_list[8];
    int camera_count = max_devices_ > 0 ? max_devices_ : 3;

    int status = CAMERA_STATUS_FAILED;
    for (int retry = 0; retry < 3; ++retry) {
      camera_count = max_devices_ > 0 ? max_devices_ : 3;
      status = CameraEnumerateDevice(enum_list, &camera_count);
      if (status == CAMERA_STATUS_SUCCESS && camera_count > 0) {
        break;
      }
    }

    if (status != CAMERA_STATUS_SUCCESS || camera_count <= 0) {
      RCLCPP_ERROR(get_logger(), "CameraEnumerateDevice failed, status=%d", status);
      return;
    }

    RCLCPP_INFO(get_logger(), "Enumerated %d camera(s)", camera_count);

    bool has_sn_config = false;
    for (const auto &cfg : configured_) {
      if (!cfg.sn.empty()) {
        has_sn_config = true;
        break;
      }
    }

    if (!has_sn_config) {
      for (int i = 0; i < camera_count; ++i) {
        CameraDevice cam;
        cam.index = i;
        cam.info = enum_list[i];
        cam.sn = enum_list[i].acSn;
        cam.probe = true;
        cam.saturation = default_saturation_;
        cam.gamma = default_gamma_;
        cam.analoggain = default_analoggain_;
        cameras_.push_back(cam);
      }
      return;
    }

    for (auto &cfg : configured_) {
      if (cfg.sn.empty()) {
        continue;
      }
      for (int i = 0; i < camera_count; ++i) {
        const std::string live_sn = enum_list[i].acSn;
        if (cfg.sn == live_sn) {
          cfg.info = enum_list[i];
          cfg.probe = true;
          cameras_.push_back(cfg);
          break;
        }
      }
      if (!cfg.probe) {
        RCLCPP_WARN(get_logger(), "Configured camera not found: index=%d sn=%s", cfg.index, cfg.sn.c_str());
      }
    }
  }

  void init_bound_cameras() {
    std::vector<CameraDevice> initialized;
    std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> pubs;

    for (auto &cam : cameras_) {
      const int init_ret = CameraInit(&cam.info, -1, -1, &cam.handle);
      if (init_ret != CAMERA_STATUS_SUCCESS) {
        RCLCPP_ERROR(get_logger(), "CameraInit failed: index=%d sn=%s ret=%d", cam.index, cam.sn.c_str(), init_ret);
        continue;
      }

      if (CameraGetCapability(cam.handle, &cam.capability) != CAMERA_STATUS_SUCCESS) {
        RCLCPP_ERROR(get_logger(), "CameraGetCapability failed: index=%d sn=%s", cam.index, cam.sn.c_str());
        CameraUnInit(cam.handle);
        cam.handle = -1;
        continue;
      }

      CameraPlay(cam.handle);

      const int max_w = cam.capability.sResolutionRange.iWidthMax;
      const int max_h = cam.capability.sResolutionRange.iHeightMax;
      cam.rgb_buffer = static_cast<unsigned char *>(malloc(max_h * max_w * 3));
      if (cam.rgb_buffer == nullptr) {
        RCLCPP_ERROR(get_logger(), "Malloc failed for camera index=%d sn=%s", cam.index, cam.sn.c_str());
        CameraUnInit(cam.handle);
        cam.handle = -1;
        continue;
      }

      if (cam.capability.sIspCapacity.bMonoSensor) {
        CameraSetIspOutFormat(cam.handle, CAMERA_MEDIA_TYPE_MONO8);
      } else {
        CameraSetIspOutFormat(cam.handle, CAMERA_MEDIA_TYPE_BGR8);
      }

      CameraSetTriggerMode(cam.handle, 2);
      CameraSetStrobePolarity(cam.handle, 0);
      CameraSetSaturation(cam.handle, cam.saturation);
      CameraSetGamma(cam.handle, cam.gamma);
      CameraSetExposureTime(cam.handle, exposure_time_);
      CameraSetAnalogGain(cam.handle, cam.analoggain);

      const std::string topic = "image_topic" + std::to_string(cam.index);
      auto pub = create_publisher<sensor_msgs::msg::Image>(topic, 10);
      pubs.push_back(pub);
      initialized.push_back(cam);

      RCLCPP_INFO(
          get_logger(),
          "Camera ready: index=%d sn=%s topic=%s saturation=%d gamma=%d gain=%d", cam.index,
          cam.sn.c_str(), topic.c_str(), cam.saturation, cam.gamma, cam.analoggain);
    }

    cameras_.swap(initialized);
    publishers_.swap(pubs);
  }

  void capture_and_publish() {
    for (size_t i = 0; i < cameras_.size(); ++i) {
      auto &cam = cameras_[i];

      const int get_ret = CameraGetImageBuffer(cam.handle, &cam.frame_info, &cam.frame_buffer, 200);
      if (get_ret != CAMERA_STATUS_SUCCESS) {
        continue;
      }

      CameraImageProcess(cam.handle, cam.frame_buffer, cam.rgb_buffer, &cam.frame_info);

      sensor_msgs::msg::Image msg;
      msg.header.stamp = now();
      msg.header.frame_id = "camera_frame_" + std::to_string(cam.index);
      msg.height = static_cast<uint32_t>(cam.frame_info.iHeight);
      msg.width = static_cast<uint32_t>(cam.frame_info.iWidth);
      msg.encoding = (cam.frame_info.uiMediaType == CAMERA_MEDIA_TYPE_MONO8) ? "mono8" : "bgr8";
      msg.is_bigendian = false;
      msg.step = msg.width * ((msg.encoding == "mono8") ? 1U : 3U);

      const auto data_size = static_cast<size_t>(msg.height * msg.step);
      msg.data.resize(data_size);
      std::memcpy(msg.data.data(), cam.rgb_buffer, data_size);

      publishers_[i]->publish(msg);
      CameraReleaseImageBuffer(cam.handle, cam.frame_buffer);
    }
  }

  std::vector<CameraDevice> configured_;
  std::vector<CameraDevice> cameras_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> publishers_;

  rclcpp::TimerBase::SharedPtr timer_;

  int frequency_hz_ = 20;
  int max_devices_ = 3;
  int default_saturation_ = 84;
  int default_gamma_ = 64;
  int default_analoggain_ = 64;
  // int default_saturation_ = 0;
  // int default_gamma_ = 0;
  // int default_analoggain_ = 0;
  double exposure_time_ = 4.3;
  // double exposure_time_ = 3.0;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
