#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp/rclcpp.hpp"
#include "common_interface/msg/height.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "common_interface/srv/update_range.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "common_interface/srv/scan3_d.hpp"
#include "common_interface/msg/tcp_pos.hpp"
#include "RVC.h"
#include "yaml-cpp/yaml.h"

namespace
{

enum class ScanMode
{
  SwingLine,
  FixedLine,
};

const char * scan_mode_to_string(const ScanMode mode)
{
  switch (mode) {
    case ScanMode::SwingLine:
      return "swing_line";
    case ScanMode::FixedLine:
      return "fixed_line";
  }
  return "unknown";
}

cv::Mat rvc_image_to_mat(const RVC::Image & image)
{
  if (!image.IsValid()) {
    return {};
  }

  const auto size = image.GetSize();
  const auto type = image.GetType();
  if (type == RVC::ImageType::Mono8) {
    return cv::Mat(size.height, size.width, CV_8UC1, const_cast<unsigned char *>(image.GetDataConstPtr())).clone();
  }
  if (type == RVC::ImageType::BGR8) {
    return cv::Mat(size.height, size.width, CV_8UC3, const_cast<unsigned char *>(image.GetDataConstPtr())).clone();
  }
  if (type == RVC::ImageType::RGB8) {
    cv::Mat rgb(size.height, size.width, CV_8UC3, const_cast<unsigned char *>(image.GetDataConstPtr()));
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }
  return {};
}

Eigen::Quaterniond rpy_to_quaternion(double roll, double pitch, double yaw)
{
  const Eigen::AngleAxisd roll_angle(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch_angle(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw_angle(yaw, Eigen::Vector3d::UnitZ());
  return yaw_angle * pitch_angle * roll_angle;
}

Eigen::Affine3d pose_to_affine(
  double x, double y, double z, double rx, double ry, double rz)
{
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation() = Eigen::Vector3d(x, y, z);
  transform.linear() = rpy_to_quaternion(rx, ry, rz).toRotationMatrix();
  return transform;
}

sensor_msgs::msg::PointCloud2 point_map_to_ros(
  const RVC::PointMap & point_map,
  const rclcpp::Time & stamp,
  const std::string & frame_id)
{
  sensor_msgs::msg::PointCloud2 msg;
  if (!point_map.IsValid()) {
    return msg;
  }

  const auto size = point_map.GetSize();
  const auto * data = point_map.GetPointDataConstPtr();
  if (data == nullptr) {
    return msg;
  }

  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.height = static_cast<uint32_t>(std::max(1, size.height));
  msg.width = static_cast<uint32_t>(std::max(1, size.width));
  msg.is_bigendian = false;
  msg.is_dense = false;
  msg.point_step = 12;
  msg.row_step = msg.point_step * msg.width;
  msg.fields.resize(3);

  msg.fields[0].name = "x";
  msg.fields[0].offset = 0;
  msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  msg.fields[0].count = 1;

  msg.fields[1].name = "y";
  msg.fields[1].offset = 4;
  msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  msg.fields[1].count = 1;

  msg.fields[2].name = "z";
  msg.fields[2].offset = 8;
  msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  msg.fields[2].count = 1;

  const size_t count = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
  msg.data.resize(count * msg.point_step);

  for (size_t i = 0; i < count; ++i) {
    const auto x = static_cast<float>(data[i * 3]);
    const auto y = static_cast<float>(data[i * 3 + 1]);
    const auto z = static_cast<float>(data[i * 3 + 2]);
    std::memcpy(msg.data.data() + i * msg.point_step + 0, &x, sizeof(float));
    std::memcpy(msg.data.data() + i * msg.point_step + 4, &y, sizeof(float));
    std::memcpy(msg.data.data() + i * msg.point_step + 8, &z, sizeof(float));
  }

  return msg;
}

sensor_msgs::msg::Image mat_to_image_msg(
  const cv::Mat & image,
  const rclcpp::Time & stamp,
  const std::string & frame_id,
  const std::string & encoding)
{
  sensor_msgs::msg::Image msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.height = static_cast<uint32_t>(image.rows);
  msg.width = static_cast<uint32_t>(image.cols);
  msg.encoding = encoding;
  msg.is_bigendian = false;
  msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(image.step);
  const auto size = static_cast<size_t>(image.rows) * image.step;
  msg.data.resize(size);
  std::memcpy(msg.data.data(), image.data, size);
  return msg;
}

struct HeightEstimateResult
{
  bool detected{false};
  double height{0.0};
  int number_points{0};
  int filtered_points{0};
  double roi_axis_min{0.0};
  double roi_axis_max{0.0};
  double roi_z_min{0.0};
  double roi_z_max{0.0};
  std::string mode{"no_data"};
};

struct AdaptiveHeightRoi
{
  double y_min{0.0};
  double y_max{0.0};
  double z_min{0.0};
  double z_max{0.0};
  bool adaptive{false};
  std::string mode{"static"};
};

double mean_of_values(const std::vector<double> & values)
{
  if (values.empty()) {
    return 0.0;
  }
  const auto sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

double mean_of_deque(const std::deque<double> & values)
{
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const auto value : values) {
    sum += value;
  }
  return sum / static_cast<double>(values.size());
}

std::vector<double> percentile_band_copy(
  std::vector<double> values,
  double percentile_low,
  double percentile_high)
{
  if (values.empty()) {
    return {};
  }
  percentile_low = std::clamp(percentile_low, 0.0, 1.0);
  percentile_high = std::clamp(percentile_high, 0.0, 1.0);
  if (percentile_low > percentile_high) {
    std::swap(percentile_low, percentile_high);
  }
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  auto index_low = static_cast<std::size_t>(
    std::floor(static_cast<double>(n) * percentile_low));
  auto index_high = static_cast<std::size_t>(
    std::floor(static_cast<double>(n) * percentile_high));
  if (index_low >= n) {
    index_low = n - 1U;
  }
  if (index_high >= n) {
    index_high = n - 1U;
  }
  if (index_high < index_low) {
    index_high = index_low;
  }
  return std::vector<double>(
    values.begin() + static_cast<std::ptrdiff_t>(index_low),
    values.begin() + static_cast<std::ptrdiff_t>(index_high + 1U));
}

double percentile_value_of_sorted(
  const std::vector<double> & sorted_values,
  double percentile)
{
  if (sorted_values.empty()) {
    return 0.0;
  }
  const auto clamped = std::clamp(percentile, 0.0, 1.0);
  auto index = static_cast<std::size_t>(
    std::floor(static_cast<double>(sorted_values.size()) * clamped));
  if (index >= sorted_values.size()) {
    index = sorted_values.size() - 1U;
  }
  return sorted_values[index];
}

void ensure_min_span(double & low, double & high, double min_span)
{
  if (high < low) {
    std::swap(low, high);
  }
  min_span = std::max(0.0, min_span);
  if ((high - low) >= min_span) {
    return;
  }
  const auto center = 0.5 * (low + high);
  const auto half = 0.5 * min_span;
  low = center - half;
  high = center + half;
}

double blend_with_step_limit(
  double previous,
  double target,
  double alpha,
  double max_step)
{
  const auto clamped_alpha = std::clamp(alpha, 0.0, 1.0);
  auto blended = previous + (target - previous) * clamped_alpha;
  if (max_step > 0.0) {
    blended = previous + std::clamp(blended - previous, -max_step, max_step);
  }
  return blended;
}

}  // namespace

class ScanCameraNode : public rclcpp::Node
{
public:
  ScanCameraNode()
  : Node("scan_camera_node")
  {
    declare_parameter<std::string>("config_file", "");
    declare_parameter<std::string>("camera_sn", "");
    declare_parameter<std::string>("point_cloud_topic", "/scan3d/raw_point_cloud");
    declare_parameter<std::string>("fixed_scan_topic", "/fixed_scan");
    declare_parameter<std::string>("image_topic", "/scan3d/raw_image");
    declare_parameter<std::string>("fixed_scan_image_topic", "/image_topic");
    declare_parameter<std::string>("tool_pos_topic", "/tool_pos");
    declare_parameter<std::string>("scan_pose_topic", "/scan_pose");
    declare_parameter<std::string>("frame_id", "camera_3d_optical_frame");
    declare_parameter<bool>("auto_capture", true);
    declare_parameter<double>("capture_period_s", 1.0);
    declare_parameter<int>("scan_time_ms", 800);
    declare_parameter<int>("exposure_time_2d", 25);
    declare_parameter<float>("gain_2d", 10.0);
    declare_parameter<float>("gamma_2d", 0.53);
    declare_parameter<int>("line_exposure_time_us", 300);
    declare_parameter<int>("line_min_distance", 400);
    declare_parameter<int>("line_max_distance", 800);
    declare_parameter<int>("fixed_line_max_distance", 1000);
    declare_parameter<int>("projector_brightness", 100);
    declare_parameter<int>("fixed_line_laser_position", 39960);
    declare_parameter<int>("fixed_line_single_capture_timeout_ms", 500);
    declare_parameter<int>("fixed_line_poll_interval_ms", 20);
    declare_parameter<int>("line_brightness_threshold", 5);
    declare_parameter<bool>("publish_image", true);
    declare_parameter<std::string>("tcp_cloud_topic", "/tcp_cloud_raw");
    declare_parameter<std::string>("estimated_height_topic", "/estimated_height");
    declare_parameter<bool>("enable_estimated_height", true);
    declare_parameter<std::string>("estimated_height_mode", "auto");
    declare_parameter<std::string>("estimated_height_output_frame", "tcp_z");
    declare_parameter<int>("estimated_height_min_points", 25);
    declare_parameter<double>("estimated_height_percentile_low", 0.333);
    declare_parameter<double>("estimated_height_percentile_high", 0.667);
    declare_parameter<int>("estimated_height_temporal_window", 5);
    declare_parameter<std::string>("estimated_height_roi_mode", "adaptive");
    declare_parameter<int>("estimated_height_roi_min_points", 80);
    declare_parameter<double>("estimated_height_roi_percentile_low", 0.02);
    declare_parameter<double>("estimated_height_roi_percentile_high", 0.98);
    declare_parameter<double>("estimated_height_roi_alpha", 0.35);
    declare_parameter<double>("estimated_height_roi_min_y_width", 0.02);
    declare_parameter<double>("estimated_height_roi_min_z_width", 0.02);
    declare_parameter<double>("estimated_height_roi_max_y_step", 0.03);
    declare_parameter<double>("estimated_height_roi_max_z_step", 0.03);
    declare_parameter<int>("estimated_height_roi_fallback_frames", 15);
    declare_parameter<double>("reconnect_period_s", 1.0);

    camera_sn_ = get_parameter("camera_sn").as_string();
    initialize_camera_system();
    load_capture_config();
    load_estimated_height_config();
    initialize_ros_interfaces();
    (void)attempt_camera_connect(false);
  }

  ~ScanCameraNode() override
  {
    close_camera();
  }

private:
  void initialize_camera_system()
  {
    if (system_initialized_) {
      return;
    }
    system_initialized_ = RVC::SystemInit();
    if (!system_initialized_) {
      RCLCPP_ERROR(get_logger(), "RVC::SystemInit failed");
      return;
    }
    RCLCPP_INFO(get_logger(), "RVC system initialized");
  }

  bool attempt_camera_connect(bool log_throttle)
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    return attempt_camera_connect_locked(log_throttle);
  }

  bool attempt_camera_connect_locked(bool log_throttle)
  {
    if (!system_initialized_) {
      log_connect_error("RVC system is not initialized", log_throttle);
      return false;
    }
    if (camera_created_ && camera_.IsOpen()) {
      return true;
    }

    destroy_camera_locked();

    if (try_connect_known_device_locked(log_throttle)) {
      return true;
    }

    std::array<RVC::Device, 8> devices;
    size_t actual_size = 0;
    const auto list_status = RVC::SystemListDevices(
      devices.data(), devices.size(), &actual_size, RVC::SystemListDeviceType::All);
    if (actual_size == 0) {
      log_connect_error(
        "no RVC camera found, list_status=" + std::to_string(list_status) +
        " actual_size=" + std::to_string(actual_size),
        log_throttle);
      return false;
    }
    if (list_status != 0) {
      RCLCPP_WARN(
        get_logger(),
        "RVC::SystemListDevices returned non-zero status=%d with actual_size=%zu, continue with detected devices",
        list_status, actual_size);
    }

    if (try_connect_listed_devices_locked(devices, actual_size, true)) {
      return true;
    }
    if (try_connect_listed_devices_locked(devices, actual_size, false)) {
      return true;
    }

    log_connect_error("RVC camera detected but failed to open", log_throttle);
    return false;
  }

  bool try_connect_known_device_locked(bool log_throttle)
  {
    if (camera_sn_.empty()) {
      return false;
    }

    auto device = RVC::SystemFindDevice(camera_sn_.c_str());
    if (!device.IsValid()) {
      log_connect_error(
        "known RVC camera not detected, sn=" + camera_sn_,
        log_throttle);
      return false;
    }

    RVC::DeviceInfo info;
    if (!device.GetDeviceInfo(&info)) {
      log_connect_error(
        "failed to query known RVC camera info, sn=" + camera_sn_,
        log_throttle);
      return false;
    }
    return connect_device_locked(device, info);
  }

  bool try_connect_listed_devices_locked(
    std::array<RVC::Device, 8> & devices,
    size_t actual_size,
    bool prefer_known_device)
  {
    if (prefer_known_device && camera_sn_.empty()) {
      return false;
    }

    const auto device_count = std::min(actual_size, devices.size());
    for (size_t index = 0; index < device_count; ++index) {
      RVC::DeviceInfo info;
      if (!devices[index].GetDeviceInfo(&info)) {
        RCLCPP_WARN(get_logger(), "failed to query RVC device info for index=%zu", index);
        continue;
      }

      const bool is_known_device = !camera_sn_.empty() && camera_sn_ == info.sn;
      if (prefer_known_device && !is_known_device) {
        continue;
      }
      if (!prefer_known_device && is_known_device) {
        continue;
      }
      if (connect_device_locked(devices[index], info)) {
        return true;
      }
    }
    return false;
  }

  bool connect_device_locked(const RVC::Device & device, const RVC::DeviceInfo & info)
  {
    if (!info.support_x2) {
      RCLCPP_WARN(
        get_logger(), "skip RVC device sn=%s because it does not support X2", info.sn);
      return false;
    }

    const bool supports_swing =
      (info.support_capture_mode & RVC::CaptureMode_SwingLineScan) != 0;
    const bool supports_fixed =
      (info.support_capture_mode & RVC::CaptureMode_FixedLineScan) != 0;
    if (!supports_swing && !supports_fixed) {
      RCLCPP_WARN(
        get_logger(),
        "RVC device sn=%s reports no recognized capture mode (support_capture_mode=%d),"
        " try opening anyway",
        info.sn, static_cast<int>(info.support_capture_mode));
    }

    camera_ = RVC::X2::Create(device);
    camera_created_ = true;
    if (!camera_.Open() || !camera_.IsOpen()) {
      RCLCPP_WARN(
        get_logger(), "failed to open RVC camera sn=%s: %s",
        info.sn, RVC::GetLastErrorMessage());
      destroy_camera_locked();
      return false;
    }

    image_camera_id_ = info.support_extra ? RVC::CameraID_Extra : RVC::CameraID_Left;
    camera_sn_ = info.sn;
    supports_swing_line_ = supports_swing;
    supports_fixed_line_ = supports_fixed;
    supported_scan_mode_ = supports_fixed && !supports_swing ? ScanMode::FixedLine : ScanMode::SwingLine;
    RCLCPP_INFO(
      get_logger(),
      "RVC camera connected sn=%s name=%s port=%s single_capture_mode=%s swing_line=%s fixed_line=%s",
      info.sn, info.name, info.port, scan_mode_to_string(supported_scan_mode_),
      supports_swing ? "true" : "false",
      supports_fixed ? "true" : "false");
    return true;
  }

  void load_capture_config()
  {
    std::string config_file = get_parameter("config_file").as_string();
    if (config_file.empty()) {
      config_file =
        ament_index_cpp::get_package_share_directory("welding_scan3d_camera_driver") +
        std::string("/config/scan_3d.yaml");
    }

    try {
      const auto yaml = YAML::LoadFile(config_file);
      if (yaml["camera"]) {
        const auto camera = yaml["camera"];
        flange_camera_ = pose_to_affine(
          camera["x"].as<double>(), camera["y"].as<double>(), camera["z"].as<double>(),
          camera["rx"].as<double>(), camera["ry"].as<double>(), camera["rz"].as<double>());
      }
      if (yaml["flange_tool"]) {
        const auto flange_tool = yaml["flange_tool"];
        flange_tool_ = pose_to_affine(
          flange_tool["x"].as<double>(), flange_tool["y"].as<double>(), flange_tool["z"].as<double>(),
          flange_tool["rx"].as<double>(), flange_tool["ry"].as<double>(), flange_tool["rz"].as<double>());
      }
      if (yaml["y_min"]) {
        y_min_ = yaml["y_min"].as<double>();
      }
      if (yaml["y_max"]) {
        y_max_ = yaml["y_max"].as<double>();
      }
      if (yaml["y_min_f"]) {
        fixed_y_min_ = yaml["y_min_f"].as<double>();
      } else {
        fixed_y_min_ = y_min_;
      }
      if (yaml["y_max_f"]) {
        fixed_y_max_ = yaml["y_max_f"].as<double>();
      } else {
        fixed_y_max_ = y_max_;
      }
      if (fixed_y_max_ <= fixed_y_min_) {
        fixed_y_min_ = y_min_;
        fixed_y_max_ = y_max_;
      }
      constexpr double kMinFixedBandWidth = 0.4;
      if ((fixed_y_max_ - fixed_y_min_) < kMinFixedBandWidth) {
        const auto center = 0.5 * (fixed_y_min_ + fixed_y_max_);
        fixed_y_min_ = center - kMinFixedBandWidth * 0.5;
        fixed_y_max_ = center + kMinFixedBandWidth * 0.5;
      }
      if (yaml["z_min"]) {
        default_z_min_ = yaml["z_min"].as<double>();
        z_min_ = default_z_min_;
      }
      if (yaml["z_max"]) {
        default_z_max_ = yaml["z_max"].as<double>();
        z_max_ = default_z_max_;
      }
    } catch (const std::exception & ex) {
      RCLCPP_WARN(get_logger(), "failed to load scan config %s: %s", config_file.c_str(), ex.what());
    }

    swing_capture_options_ = RVC::X2::CaptureOptions();
    swing_capture_options_.capture_mode = RVC::CaptureMode_SwingLineScan;
    swing_capture_options_.line_scanner_scan_time_ms = get_parameter("scan_time_ms").as_int();
    swing_capture_options_.line_scanner_exposure_time_us =
      get_parameter("line_exposure_time_us").as_int();
    swing_capture_options_.line_scanner_min_distance = get_parameter("line_min_distance").as_int();
    swing_capture_options_.line_scanner_max_distance = get_parameter("line_max_distance").as_int();
    swing_capture_options_.projector_brightness = get_parameter("projector_brightness").as_int();
    swing_capture_options_.line_scanner_brightness_threshold =
      get_parameter("line_brightness_threshold").as_int();
    swing_capture_options_.correspond2d = true;
    swing_capture_options_.enable_2d_in_capture = true;
    swing_capture_options_.exposure_time_2d = get_parameter("exposure_time_2d").as_int();
    swing_capture_options_.gain_2d = get_parameter("gain_2d").as_double();
    swing_capture_options_.gamma_2d = get_parameter("gamma_2d").as_double();

    fixed_capture_options_ = RVC::X2::CaptureOptions();
    fixed_capture_options_.capture_mode = RVC::CaptureMode_FixedLineScan;
    fixed_capture_options_.line_scanner_exposure_time_us =
      get_parameter("line_exposure_time_us").as_int();
    fixed_capture_options_.line_scanner_min_distance = get_parameter("line_min_distance").as_int();
    fixed_capture_options_.line_scanner_max_distance =
      get_parameter("fixed_line_max_distance").as_int();
    fixed_capture_options_.projector_brightness = get_parameter("projector_brightness").as_int();
    fixed_capture_options_.gain_3d = 0.0F;
    const auto fixed_line_laser_position = get_parameter("fixed_line_laser_position").as_int();
    fixed_capture_options_.line_scanner_laser_position = static_cast<uint16_t>(
      std::clamp<int64_t>(fixed_line_laser_position, 0, 65535));
    fixed_capture_options_.line_scanner_brightness_threshold =
      get_parameter("line_brightness_threshold").as_int();
    fixed_capture_options_.correspond2d = true;
    fixed_capture_options_.enable_2d_in_capture = true;
    fixed_capture_options_.exposure_time_2d = get_parameter("exposure_time_2d").as_int();
    fixed_capture_options_.gain_2d = get_parameter("gain_2d").as_double();
    fixed_capture_options_.gamma_2d = get_parameter("gamma_2d").as_double();
  }

  void load_estimated_height_config()
  {
    enable_estimated_height_ = get_parameter("enable_estimated_height").as_bool();
    estimated_height_mode_ = get_parameter("estimated_height_mode").as_string();
    estimated_height_roi_mode_ = get_parameter("estimated_height_roi_mode").as_string();
    if (estimated_height_roi_mode_ != "static" && estimated_height_roi_mode_ != "adaptive") {
      RCLCPP_WARN(
        get_logger(),
        "invalid estimated_height_roi_mode=%s, fallback to adaptive",
        estimated_height_roi_mode_.c_str());
      estimated_height_roi_mode_ = "adaptive";
    }
    estimated_height_output_frame_ = get_parameter("estimated_height_output_frame").as_string();
    if (estimated_height_output_frame_ != "tcp_z") {
      RCLCPP_WARN(
        get_logger(),
        "estimated_height now uses /scan3d/raw_point_cloud middle-band abs(z) method, "
        "forcing estimated_height_output_frame from %s to tcp_z",
        estimated_height_output_frame_.c_str());
      estimated_height_output_frame_ = "tcp_z";
    }
    estimated_height_min_points_ =
      std::max<int>(8, get_parameter("estimated_height_min_points").as_int());
    estimated_height_percentile_low_ =
      std::clamp(get_parameter("estimated_height_percentile_low").as_double(), 0.0, 1.0);
    estimated_height_percentile_high_ =
      std::clamp(get_parameter("estimated_height_percentile_high").as_double(), 0.0, 1.0);
    if (estimated_height_percentile_low_ >= estimated_height_percentile_high_) {
      RCLCPP_WARN(
        get_logger(),
        "invalid estimated height percentile range [%.3f, %.3f], fallback to [0.333, 0.667]",
        estimated_height_percentile_low_, estimated_height_percentile_high_);
      estimated_height_percentile_low_ = 0.333;
      estimated_height_percentile_high_ = 0.667;
    }
    estimated_height_temporal_window_ =
      std::max<int>(1, get_parameter("estimated_height_temporal_window").as_int());
    estimated_height_roi_min_points_ =
      std::max<int>(estimated_height_min_points_, get_parameter("estimated_height_roi_min_points").as_int());
    estimated_height_roi_percentile_low_ =
      std::clamp(get_parameter("estimated_height_roi_percentile_low").as_double(), 0.0, 1.0);
    estimated_height_roi_percentile_high_ =
      std::clamp(get_parameter("estimated_height_roi_percentile_high").as_double(), 0.0, 1.0);
    if (estimated_height_roi_percentile_low_ >= estimated_height_roi_percentile_high_) {
      RCLCPP_WARN(
        get_logger(),
        "invalid adaptive roi percentile range [%.3f, %.3f], fallback to [0.02, 0.98]",
        estimated_height_roi_percentile_low_, estimated_height_roi_percentile_high_);
      estimated_height_roi_percentile_low_ = 0.02;
      estimated_height_roi_percentile_high_ = 0.98;
    }
    estimated_height_roi_alpha_ =
      std::clamp(get_parameter("estimated_height_roi_alpha").as_double(), 0.0, 1.0);
    estimated_height_roi_min_y_width_ =
      std::max(0.0, get_parameter("estimated_height_roi_min_y_width").as_double());
    estimated_height_roi_min_z_width_ =
      std::max(0.0, get_parameter("estimated_height_roi_min_z_width").as_double());
    estimated_height_roi_max_y_step_ =
      std::max(0.0, get_parameter("estimated_height_roi_max_y_step").as_double());
    estimated_height_roi_max_z_step_ =
      std::max(0.0, get_parameter("estimated_height_roi_max_z_step").as_double());
    estimated_height_roi_fallback_frames_ =
      std::max<int>(0, get_parameter("estimated_height_roi_fallback_frames").as_int());
    estimated_height_roi_valid_ = false;
    estimated_height_roi_consecutive_failures_ = 0;
  }

  void initialize_ros_interfaces()
  {
    point_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      get_parameter("point_cloud_topic").as_string(), rclcpp::SensorDataQoS());
    fixed_scan_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      get_parameter("fixed_scan_topic").as_string(), rclcpp::SensorDataQoS());
    image_pub_ = create_publisher<sensor_msgs::msg::Image>(
      get_parameter("image_topic").as_string(), rclcpp::SensorDataQoS());
    fixed_scan_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
      get_parameter("fixed_scan_image_topic").as_string(), rclcpp::SensorDataQoS());
    tcp_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      get_parameter("tcp_cloud_topic").as_string(), rclcpp::SensorDataQoS());
    estimated_height_pub_ = create_publisher<common_interface::msg::Height>(
      get_parameter("estimated_height_topic").as_string(), 10);
    scan_pose_pub_ = create_publisher<common_interface::msg::TcpPos>(
      get_parameter("scan_pose_topic").as_string(), 1);

    tool_pos_sub_ = create_subscription<common_interface::msg::TcpPos>(
      get_parameter("tool_pos_topic").as_string(), 1,
      [this](const common_interface::msg::TcpPos::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(scan_pose_mutex_);
        scan_pose_ = *msg;
      });

    start_srv_ = create_service<std_srvs::srv::Trigger>(
      "start_capture",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(capture_mutex_);
        auto_capture_enabled_ = true;
        fixed_scan_enabled_ = false;
        stop_fixed_line_scan_locked();
        response->success = true;
        response->message = "3d camera auto capture started";
      });

    start_fix_scan_srv_ = create_service<std_srvs::srv::Trigger>(
      "start_fix_scan",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(capture_mutex_);
        auto_capture_enabled_ = false;
        if (!attempt_camera_connect_locked(false)) {
          response->success = false;
          response->message = "camera unavailable";
          return;
        }
        if (!supports_fixed_line_) {
          response->success = false;
          response->message = "camera does not support fixed line scan";
          return;
        }
        if (!start_fixed_line_scan_locked()) {
          response->success = false;
          response->message = std::string("failed to start fixed scan: ") + RVC::GetLastErrorMessage();
          return;
        }
        fixed_scan_enabled_ = true;
        response->success = true;
        response->message = "fixed scan started";
      });

    stop_srv_ = create_service<std_srvs::srv::Trigger>(
      "stop_capture",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        auto_capture_enabled_ = false;
        fixed_scan_enabled_ = false;
        std::lock_guard<std::mutex> lock(capture_mutex_);
        stop_fixed_line_scan_locked();
        response->success = true;
        response->message = "3d camera auto capture stopped";
      });

    stop_fix_scan_srv_ = create_service<std_srvs::srv::Trigger>(
      "stop_fix_scan",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(capture_mutex_);
        fixed_scan_enabled_ = false;
        stop_fixed_line_scan_locked();
        response->success = true;
        response->message = "fixed scan stopped";
      });

    reload_srv_ = create_service<std_srvs::srv::Trigger>(
      "camera_reload",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        load_capture_config();
        response->success = true;
        response->message = "3d camera config reloaded";
      });

    scan_srv_ = create_service<common_interface::srv::Scan3D>(
      "scan_3d",
      [this](
        const std::shared_ptr<common_interface::srv::Scan3D::Request>,
        std::shared_ptr<common_interface::srv::Scan3D::Response> response) {
        response->success = capture_once(true, &response->points, &response->image, &response->message);
        if (response->success && response->message.empty()) {
          response->message = "3d scan completed";
        }
      });

    update_cropping_range_srv_ = create_service<common_interface::srv::UpdateRange>(
      "/update_camera_3d_node_cropping_z_range",
      [this](
        const std::shared_ptr<common_interface::srv::UpdateRange::Request> request,
        std::shared_ptr<common_interface::srv::UpdateRange::Response>) {
        z_min_ = default_z_min_ + request->delta;
        z_max_ = default_z_max_ + request->delta;
        RCLCPP_INFO(
          get_logger(), "updated cropping z range: (%f, %f)", z_min_, z_max_);
      });

    auto_capture_enabled_ = get_parameter("auto_capture").as_bool();
    const auto capture_period_s = std::max(0.2, get_parameter("capture_period_s").as_double());
    const auto reconnect_period_s = std::max(0.2, get_parameter("reconnect_period_s").as_double());
    capture_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(capture_period_s)),
      [this]() {
        (void)capture_once(false);
      });
    fixed_scan_timer_ = create_wall_timer(
      std::chrono::milliseconds(66),
      [this]() {
        (void)publish_fixed_line_scan_once();
      });
    scan_pose_timer_ = create_wall_timer(
      std::chrono::milliseconds(33),
      [this]() {
        publish_scan_pose();
      });
    reconnect_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(reconnect_period_s)),
      [this]() {
        (void)attempt_camera_connect(true);
      });
  }

  bool capture_once(
    bool force_capture,
    sensor_msgs::msg::PointCloud2 * captured_point_cloud = nullptr,
    sensor_msgs::msg::Image * captured_image = nullptr,
    std::string * detail = nullptr)
  {
    if (!force_capture && !auto_capture_enabled_) {
      return false;
    }

    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (!attempt_camera_connect_locked(force_capture)) {
      if (detail != nullptr) {
        *detail = "camera unavailable";
      }
      return false;
    }
    if (!camera_.IsOpen()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "RVC camera is not open");
      if (detail != nullptr) {
        *detail = "camera is not open";
      }
      return false;
    }

    bool success = false;
    if (supported_scan_mode_ == ScanMode::SwingLine) {
      success = capture_swing_line_once_locked(captured_point_cloud, captured_image);
      if (!success && detail != nullptr) {
        *detail = std::string("RVC swing line capture failed: ") + RVC::GetLastErrorMessage();
      }
    } else {
      success = capture_fixed_line_snapshot_locked(captured_point_cloud, captured_image, detail);
    }

    if (success && detail != nullptr && detail->empty()) {
      *detail = "3d scan completed";
    }
    return success;
  }

  bool publish_fixed_line_scan_once()
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (!fixed_scan_enabled_) {
      return false;
    }
    if (!attempt_camera_connect_locked(true)) {
      return false;
    }
    if (!supports_fixed_line_) {
      return false;
    }
    if (!fixed_line_scan_active_ && !start_fixed_line_scan_locked()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000, "failed to start fixed line scan: %s",
        RVC::GetLastErrorMessage());
      destroy_camera_locked();
      return false;
    }

    RVC::PointMap point_map;
    if (!wait_for_fixed_line_point_map_locked(point_map, std::chrono::milliseconds(0))) {
      return false;
    }
    return publish_capture_outputs_locked(point_map, nullptr, nullptr, true);
  }

  void publish_scan_pose()
  {
    common_interface::msg::TcpPos scan_pose;
    {
      std::lock_guard<std::mutex> lock(scan_pose_mutex_);
      scan_pose = scan_pose_;
    }
    scan_pose_pub_->publish(scan_pose);
  }

  void close_camera()
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    destroy_camera_locked();
    if (system_initialized_) {
      RVC::SystemShutdown();
      system_initialized_ = false;
    }
  }

  void destroy_camera_locked()
  {
    if (!camera_created_) {
      return;
    }
    stop_fixed_line_scan_locked();
    if (camera_.IsOpen()) {
      camera_.Close();
    }
    RVC::X2::Destroy(camera_);
    camera_created_ = false;
    supports_swing_line_ = false;
    supports_fixed_line_ = false;
  }

  bool start_fixed_line_scan_locked()
  {
    if (!supports_fixed_line_) {
      return false;
    }
    if (fixed_line_scan_active_) {
      return true;
    }
    if (!camera_.StartFixedLineScan(fixed_capture_options_)) {
      return false;
    }
    fixed_line_scan_active_ = true;
    return true;
  }

  void stop_fixed_line_scan_locked()
  {
    if (!fixed_line_scan_active_) {
      return;
    }
    camera_.StopFixedLineScan();
    fixed_line_scan_active_ = false;
  }

  bool capture_swing_line_once_locked(
    sensor_msgs::msg::PointCloud2 * captured_point_cloud,
    sensor_msgs::msg::Image * captured_image)
  {
    stop_fixed_line_scan_locked();
    if (!camera_.Capture(swing_capture_options_)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000, "RVC capture failed: %s", RVC::GetLastErrorMessage());
      destroy_camera_locked();
      return false;
    }
    return publish_capture_outputs_locked(
      camera_.GetPointMap(), captured_point_cloud, captured_image, false);
  }

  bool capture_fixed_line_snapshot_locked(
    sensor_msgs::msg::PointCloud2 * captured_point_cloud,
    sensor_msgs::msg::Image * captured_image,
    std::string * detail)
  {
    if (!supports_fixed_line_) {
      if (detail != nullptr) {
        *detail = "camera does not support fixed line scan";
      }
      return false;
    }

    const bool started_here = !fixed_line_scan_active_;
    if (started_here && !start_fixed_line_scan_locked()) {
      if (detail != nullptr) {
        *detail = std::string("failed to start fixed scan: ") + RVC::GetLastErrorMessage();
      }
      return false;
    }

    const auto timeout_ms = started_here ?
      std::max<int64_t>(0, get_parameter("fixed_line_single_capture_timeout_ms").as_int()) : 0;

    RVC::PointMap point_map;
    const bool have_point_map = wait_for_fixed_line_point_map_locked(
      point_map, std::chrono::milliseconds(timeout_ms));
    if (started_here) {
      stop_fixed_line_scan_locked();
    }
    if (!have_point_map) {
      if (detail != nullptr) {
        *detail = std::string("fixed line scan returned no data: ") + RVC::GetLastErrorMessage();
      }
      return false;
    }

    const bool published =
      publish_capture_outputs_locked(point_map, captured_point_cloud, captured_image, true);
    if (!published && detail != nullptr) {
      *detail = "fixed line scan returned empty point cloud";
    }
    return published;
  }

  bool wait_for_fixed_line_point_map_locked(
    RVC::PointMap & point_map,
    std::chrono::milliseconds timeout)
  {
    const auto poll_interval = std::chrono::milliseconds(
      std::max<int64_t>(1, get_parameter("fixed_line_poll_interval_ms").as_int()));
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    do {
      if (camera_.GetFixedLineScanPointMap(point_map) && point_map.IsValid()) {
        return true;
      }
      if (timeout.count() <= 0) {
        break;
      }
      std::this_thread::sleep_for(poll_interval);
    } while (std::chrono::steady_clock::now() < deadline);

    return false;
  }

  bool publish_capture_outputs_locked(
    const RVC::PointMap & point_map,
    sensor_msgs::msg::PointCloud2 * captured_point_cloud = nullptr,
    sensor_msgs::msg::Image * captured_image = nullptr,
    bool publish_estimated_height = false)
  {
    const auto stamp = now();
    const auto frame_id = get_parameter("frame_id").as_string();
    bool published = false;

    if (point_map.IsValid()) {
      auto point_cloud = point_map_to_ros(point_map, stamp, frame_id);
      if (!point_cloud.data.empty()) {
        const auto raw_points = static_cast<int>(point_cloud.width * point_cloud.height);
        auto tcp_cloud = transform_tcp_cloud(point_cloud, stamp, false, 0.0, 0.0);
        auto filtered_cloud = transform_tcp_cloud(
          point_cloud, stamp, true, fixed_y_min_, fixed_y_max_);
        if (captured_point_cloud != nullptr) {
          *captured_point_cloud = point_cloud;
        }
        tcp_cloud_pub_->publish(tcp_cloud);
        if (!filtered_cloud.data.empty()) {
          fixed_scan_pub_->publish(filtered_cloud);
        } else {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "fixed scan cloud filtered to empty, publishing tcp fallback pts=%d y_range=[%.4f, %.4f]",
            raw_points, fixed_y_min_, fixed_y_max_);
          fixed_scan_pub_->publish(tcp_cloud);
        }
        if (publish_estimated_height) {
          publish_estimated_height_message(point_cloud);
        }
        point_cloud_pub_->publish(point_cloud);
        published = true;
      }
    }

    if (get_parameter("publish_image").as_bool()) {
      const auto image = camera_.GetImage(image_camera_id_);
      auto cv_image = rvc_image_to_mat(image);
      if (!cv_image.empty()) {
        auto image_msg = mat_to_image_msg(
          cv_image, stamp, frame_id, cv_image.channels() == 1 ? "mono8" : "bgr8");
        if (captured_image != nullptr) {
          *captured_image = image_msg;
        }
        image_pub_->publish(image_msg);
        fixed_scan_image_pub_->publish(image_msg);
        published = true;
      }
    }

    return published;
  }

  void log_connect_error(const std::string & message, bool throttle)
  {
    if (throttle) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", message.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "%s", message.c_str());
    }
  }

  sensor_msgs::msg::PointCloud2 transform_tcp_cloud(
    const sensor_msgs::msg::PointCloud2 & source,
    const rclcpp::Time & stamp,
    bool apply_y_filter,
    double y_min,
    double y_max) const
  {
    const Eigen::Affine3d tcp_to_camera = flange_tool_.inverse() * flange_camera_;

    sensor_msgs::msg::PointCloud2 transformed;
    transformed.header.stamp = stamp;
    transformed.header.frame_id = "tcp";
    transformed.height = 1;
    transformed.is_bigendian = false;
    transformed.is_dense = false;
    transformed.point_step = 12;
    transformed.fields = source.fields;
    transformed.data.clear();

    sensor_msgs::PointCloud2ConstIterator<float> x_it(source, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y_it(source, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z_it(source, "z");
    const auto count = static_cast<size_t>(source.width) * static_cast<size_t>(source.height);
    for (size_t i = 0; i < count; ++i, ++x_it, ++y_it, ++z_it) {
      if (!std::isfinite(*x_it) || !std::isfinite(*y_it) || !std::isfinite(*z_it)) {
        continue;
      }

      const Eigen::Vector3d camera_point(*x_it, *y_it, *z_it);
      const Eigen::Vector3d tcp_point = tcp_to_camera * camera_point;
      const auto tcp_x = static_cast<float>(tcp_point.x());
      const auto tcp_y = static_cast<float>(tcp_point.y());
      const auto tcp_z = static_cast<float>(tcp_point.z());

      if (!std::isfinite(tcp_x) || !std::isfinite(tcp_y) || !std::isfinite(tcp_z)) {
        continue;
      }

      if (apply_y_filter && (tcp_y < y_min || tcp_y > y_max)) {
        continue;
      }
      const auto before = transformed.data.size();
      transformed.data.resize(before + transformed.point_step);
      std::memcpy(transformed.data.data() + before, &tcp_x, sizeof(float));
      std::memcpy(transformed.data.data() + before + 4, &tcp_y, sizeof(float));
      std::memcpy(transformed.data.data() + before + 8, &tcp_z, sizeof(float));
    }

    transformed.width = static_cast<uint32_t>(transformed.data.size() / transformed.point_step);
    transformed.row_step = transformed.width * transformed.point_step;
    return transformed;
  }

  AdaptiveHeightRoi static_height_roi() const
  {
    AdaptiveHeightRoi roi;
    roi.y_min = std::min(y_min_, y_max_);
    roi.y_max = std::max(y_min_, y_max_);
    roi.z_min = std::min(z_min_, z_max_);
    roi.z_max = std::max(z_min_, z_max_);
    roi.adaptive = false;
    roi.mode = "static";
    return roi;
  }

  AdaptiveHeightRoi resolve_height_roi(
    const std::vector<double> & y_values,
    const std::vector<double> & z_values)
  {
    auto fallback_roi = static_height_roi();
    if (estimated_height_roi_mode_ != "adaptive") {
      return fallback_roi;
    }

    if (
      y_values.size() < static_cast<std::size_t>(estimated_height_roi_min_points_) ||
      z_values.size() < static_cast<std::size_t>(estimated_height_roi_min_points_))
    {
      ++estimated_height_roi_consecutive_failures_;
      if (
        estimated_height_roi_valid_ &&
        (estimated_height_roi_fallback_frames_ <= 0 ||
        estimated_height_roi_consecutive_failures_ <= estimated_height_roi_fallback_frames_))
      {
        auto held_roi = estimated_height_adaptive_roi_;
        held_roi.mode = "adaptive_hold";
        return held_roi;
      }
      fallback_roi.mode = "static_fallback";
      return fallback_roi;
    }

    std::vector<double> sorted_y = y_values;
    std::vector<double> sorted_z = z_values;
    std::sort(sorted_y.begin(), sorted_y.end());
    std::sort(sorted_z.begin(), sorted_z.end());

    auto target_roi = fallback_roi;
    target_roi.y_min = percentile_value_of_sorted(sorted_y, estimated_height_roi_percentile_low_);
    target_roi.y_max = percentile_value_of_sorted(sorted_y, estimated_height_roi_percentile_high_);
    target_roi.z_min = percentile_value_of_sorted(sorted_z, estimated_height_roi_percentile_low_);
    target_roi.z_max = percentile_value_of_sorted(sorted_z, estimated_height_roi_percentile_high_);
    ensure_min_span(target_roi.y_min, target_roi.y_max, estimated_height_roi_min_y_width_);
    ensure_min_span(target_roi.z_min, target_roi.z_max, estimated_height_roi_min_z_width_);

    if (estimated_height_roi_valid_) {
      target_roi.y_min = blend_with_step_limit(
        estimated_height_adaptive_roi_.y_min, target_roi.y_min,
        estimated_height_roi_alpha_, estimated_height_roi_max_y_step_);
      target_roi.y_max = blend_with_step_limit(
        estimated_height_adaptive_roi_.y_max, target_roi.y_max,
        estimated_height_roi_alpha_, estimated_height_roi_max_y_step_);
      target_roi.z_min = blend_with_step_limit(
        estimated_height_adaptive_roi_.z_min, target_roi.z_min,
        estimated_height_roi_alpha_, estimated_height_roi_max_z_step_);
      target_roi.z_max = blend_with_step_limit(
        estimated_height_adaptive_roi_.z_max, target_roi.z_max,
        estimated_height_roi_alpha_, estimated_height_roi_max_z_step_);
      target_roi.mode = "adaptive_blend";
    } else {
      target_roi.mode = "adaptive_init";
    }

    ensure_min_span(target_roi.y_min, target_roi.y_max, estimated_height_roi_min_y_width_);
    ensure_min_span(target_roi.z_min, target_roi.z_max, estimated_height_roi_min_z_width_);
    target_roi.adaptive = true;
    estimated_height_adaptive_roi_ = target_roi;
    estimated_height_roi_valid_ = true;
    estimated_height_roi_consecutive_failures_ = 0;
    return target_roi;
  }

  HeightEstimateResult estimate_height_from_camera_cloud(
    const sensor_msgs::msg::PointCloud2 & cloud)
  {
    HeightEstimateResult result;
    const auto count = static_cast<std::size_t>(cloud.width) * static_cast<std::size_t>(cloud.height);
    std::vector<std::pair<double, double>> finite_points;
    finite_points.reserve(count);
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();

    sensor_msgs::PointCloud2ConstIterator<float> x_it(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y_it(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z_it(cloud, "z");
    for (std::size_t i = 0; i < count; ++i, ++x_it, ++y_it, ++z_it) {
      (void)*y_it;
      if (!std::isfinite(*x_it) || !std::isfinite(*z_it)) {
        continue;
      }
      const auto x = static_cast<double>(*x_it);
      const auto z = static_cast<double>(*z_it);
      finite_points.emplace_back(x, z);
      min_x = std::min(min_x, x);
      max_x = std::max(max_x, x);
    }

    if (finite_points.size() < static_cast<std::size_t>(estimated_height_min_points_)) {
      result.mode = "camera_raw_too_few_points";
      return result;
    }

    const double center_x = 0.5 * (min_x + max_x);
    constexpr double kHalfBandWidth = 0.03;
    result.roi_axis_min = center_x - kHalfBandWidth;
    result.roi_axis_max = center_x + kHalfBandWidth;

    std::vector<double> abs_z_values;
    abs_z_values.reserve(finite_points.size());
    for (const auto & point : finite_points) {
      if (point.first < result.roi_axis_min || point.first > result.roi_axis_max) {
        continue;
      }
      ++result.filtered_points;
      const double abs_z = std::abs(point.second);
      abs_z_values.push_back(abs_z);
      if (result.filtered_points == 1) {
        result.roi_z_min = abs_z;
        result.roi_z_max = abs_z;
      } else {
        result.roi_z_min = std::min(result.roi_z_min, abs_z);
        result.roi_z_max = std::max(result.roi_z_max, abs_z);
      }
    }

    if (result.filtered_points < estimated_height_min_points_) {
      result.mode = "camera_raw_mid_band_too_few_points";
      return result;
    }

    result.detected = true;
    result.height = mean_of_values(abs_z_values);
    result.number_points = static_cast<int>(abs_z_values.size());
    result.mode = "camera_raw_mid_band_abs_z_mean";
    return result;
  }

  void publish_estimated_height_message(
    const sensor_msgs::msg::PointCloud2 & camera_cloud)
  {
    if (!enable_estimated_height_ || estimated_height_pub_ == nullptr) {
      return;
    }
    auto estimate = estimate_height_from_camera_cloud(camera_cloud);
    if (estimated_height_mode_ != "auto") {
      estimate.mode += "_legacy_mode";
    }

    common_interface::msg::Height msg;
    msg.detected = estimate.detected;
    msg.number_points = estimate.number_points;
    if (estimate.detected) {
      estimated_height_history_.push_back(estimate.height);
      while (estimated_height_history_.size() >
        static_cast<std::size_t>(estimated_height_temporal_window_))
      {
        estimated_height_history_.pop_front();
      }
      msg.estimated_height = mean_of_deque(estimated_height_history_);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "estimated_height source=/scan3d/raw_point_cloud frame=%s mode=%s raw=%.6f avg=%.6f "
        "roi_x=[%.4f, %.4f] roi_z=[%.4f, %.4f] filtered_points=%d used_points=%d",
        estimated_height_output_frame_.c_str(), estimate.mode.c_str(),
        estimate.height, msg.estimated_height,
        estimate.roi_axis_min, estimate.roi_axis_max,
        estimate.roi_z_min, estimate.roi_z_max,
        estimate.filtered_points, msg.number_points);
    } else {
      estimated_height_history_.clear();
      msg.estimated_height = 0.0;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "estimated_height source=/scan3d/raw_point_cloud frame=%s failed: %s "
        "roi_x=[%.4f, %.4f] roi_z=[%.4f, %.4f]",
        estimated_height_output_frame_.c_str(), estimate.mode.c_str(),
        estimate.roi_axis_min, estimate.roi_axis_max,
        estimate.roi_z_min, estimate.roi_z_max);
    }
    estimated_height_pub_->publish(msg);
  }

  std::mutex capture_mutex_;
  std::mutex scan_pose_mutex_;
  RVC::X2 camera_;
  bool camera_created_{false};
  bool system_initialized_{false};
  bool auto_capture_enabled_{true};
  bool fixed_scan_enabled_{false};
  bool fixed_line_scan_active_{false};
  bool supports_swing_line_{false};
  bool supports_fixed_line_{false};
  RVC::CameraID image_camera_id_{RVC::CameraID_Left};
  std::string camera_sn_;
  ScanMode supported_scan_mode_{ScanMode::SwingLine};
  RVC::X2::CaptureOptions swing_capture_options_;
  RVC::X2::CaptureOptions fixed_capture_options_;
  double y_min_{-0.01};
  double y_max_{0.01};
  double fixed_y_min_{-0.2};
  double fixed_y_max_{0.2};
  double default_z_min_{-0.035};
  double default_z_max_{0.015};
  double z_min_{-0.035};
  double z_max_{0.015};
  bool enable_estimated_height_{true};
  std::string estimated_height_mode_{"auto"};
  std::string estimated_height_roi_mode_{"adaptive"};
  std::string estimated_height_output_frame_{"tcp_z"};
  int estimated_height_min_points_{25};
  double estimated_height_percentile_low_{0.333};
  double estimated_height_percentile_high_{0.667};
  int estimated_height_temporal_window_{5};
  int estimated_height_roi_min_points_{80};
  double estimated_height_roi_percentile_low_{0.02};
  double estimated_height_roi_percentile_high_{0.98};
  double estimated_height_roi_alpha_{0.35};
  double estimated_height_roi_min_y_width_{0.02};
  double estimated_height_roi_min_z_width_{0.02};
  double estimated_height_roi_max_y_step_{0.03};
  double estimated_height_roi_max_z_step_{0.03};
  int estimated_height_roi_fallback_frames_{15};
  bool estimated_height_roi_valid_{false};
  int estimated_height_roi_consecutive_failures_{0};
  AdaptiveHeightRoi estimated_height_adaptive_roi_;
  Eigen::Affine3d flange_camera_{Eigen::Affine3d::Identity()};
  Eigen::Affine3d flange_tool_{Eigen::Affine3d::Identity()};
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fixed_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tcp_cloud_pub_;
  rclcpp::Publisher<common_interface::msg::Height>::SharedPtr estimated_height_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr fixed_scan_image_pub_;
  rclcpp::Publisher<common_interface::msg::TcpPos>::SharedPtr scan_pose_pub_;
  std::deque<double> estimated_height_history_;
  common_interface::msg::TcpPos scan_pose_;
  rclcpp::Subscription<common_interface::msg::TcpPos>::SharedPtr tool_pos_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_fix_scan_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_fix_scan_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reload_srv_;
  rclcpp::Service<common_interface::srv::Scan3D>::SharedPtr scan_srv_;
  rclcpp::Service<common_interface::srv::UpdateRange>::SharedPtr update_cropping_range_srv_;
  rclcpp::TimerBase::SharedPtr capture_timer_;
  rclcpp::TimerBase::SharedPtr fixed_scan_timer_;
  rclcpp::TimerBase::SharedPtr scan_pose_timer_;
  rclcpp::TimerBase::SharedPtr reconnect_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ScanCameraNode>());
  rclcpp::shutdown();
  return 0;
}
