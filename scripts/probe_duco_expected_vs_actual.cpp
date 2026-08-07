/**
 * Probe: record Duco actual TCP vs command TCP while using the teach pendant.
 *
 * Goal: check whether get_tcp_pose_command / getRobotStatus retain the intended
 * target when physical motion is blocked (e.g. command 5 mm, actual 1 mm).
 *
 * Build+run via: scripts/probe_duco_expected_vs_actual.sh
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "robot_control/DucoCobot.h"

namespace {

std::atomic<bool> g_running{true};

void on_signal(int) { g_running = false; }

int64_t now_us() {
  using clock = std::chrono::system_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(
             clock::now().time_since_epoch())
      .count();
}

double vec_norm3(const std::vector<double> &v) {
  if (v.size() < 3) {
    return 0.0;
  }
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

std::string join6(const std::vector<double> &v) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  for (size_t i = 0; i < 6; ++i) {
    if (i) {
      oss << ',';
    }
    oss << (i < v.size() ? v[i] : 0.0);
  }
  return oss.str();
}

void fill6(std::vector<double> &v) {
  v.resize(6, std::numeric_limits<double>::quiet_NaN());
}

double linear_error_mm(const std::vector<double> &expected,
                       const std::vector<double> &actual) {
  if (expected.size() < 3 || actual.size() < 3) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  for (size_t i = 0; i < 3; ++i) {
    if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  const std::vector<double> error = {
      expected[0] - actual[0],
      expected[1] - actual[1],
      expected[2] - actual[2],
  };
  return 1000.0 * vec_norm3(error);
}

}  // namespace

int main(int argc, char **argv) {
  std::string ip = "192.168.1.10";
  unsigned int port = 7003;
  double hz = 50.0;
  std::string out_csv = "duco_expected_vs_actual.csv";
  bool use_status = true;
  bool dry_connect_only = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--ip") {
      ip = need("--ip");
    } else if (arg == "--port") {
      port = static_cast<unsigned int>(std::stoul(need("--port")));
    } else if (arg == "--hz") {
      hz = std::stod(need("--hz"));
    } else if (arg == "--out") {
      out_csv = need("--out");
    } else if (arg == "--no-status") {
      use_status = false;
    } else if (arg == "--connect-only") {
      dry_connect_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: probe_duco_expected_vs_actual [options]\n"
          << "  --ip IP              robot IP (default 192.168.1.10)\n"
          << "  --port PORT          RPC port (default 7003)\n"
          << "  --hz HZ              sample rate (default 50)\n"
          << "  --out PATH.csv       output csv path\n"
          << "  --no-status          only use get_tcp_pose(_command)\n"
          << "  --connect-only       open+close without recording\n";
      return 0;
    } else {
      std::cerr << "unknown arg: " << arg << "\n";
      return 2;
    }
  }

  if (!std::isfinite(hz) || hz < 1.0 || hz > 200.0) {
    std::cerr << "[probe] --hz must be finite and in [1, 200]\n";
    return 2;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  std::cout << "[probe] connecting " << ip << ":" << port << " ...\n";
  DucoRPC::DucoCobot robot(ip, port);
  const int32_t open_ret = robot.open();
  if (open_ret != 0) {
    std::cerr << "[probe] open() failed, ret=" << open_ret
              << " (is robot_driver_bridge_node still holding the RPC?)\n";
    return 1;
  }
  std::cout << "[probe] open() ok\n";

  if (dry_connect_only) {
    robot.close();
    std::cout << "[probe] connect-only done\n";
    return 0;
  }

  std::ofstream ofs(out_csv);
  if (!ofs) {
    std::cerr << "[probe] cannot write " << out_csv << "\n";
    robot.close();
    return 1;
  }

  ofs << "timestamp_us,"
         "actual_x,actual_y,actual_z,actual_rx,actual_ry,actual_rz,"
         "command_x,command_y,command_z,command_rx,command_ry,command_rz,"
         "cmd_speed_vx,cmd_speed_vy,cmd_speed_vz,cmd_speed_wx,cmd_speed_wy,cmd_speed_wz,"
         "status_actual_x,status_actual_y,status_actual_z,"
         "status_actual_rx,status_actual_ry,status_actual_rz,"
         "status_expect_x,status_expect_y,status_expect_z,"
         "status_expect_rx,status_expect_ry,status_expect_rz,"
         "lin_err_mm,status_lin_err_mm,collision,collision_axis,robot_error,op_mode,"
         "tcp_valid,status_valid,rpc_duration_us,schedule_lag_us,source\n";
  ofs << std::setprecision(17);

  const auto period =
      std::chrono::duration<double>(1.0 / hz);
  std::vector<double> actual;
  std::vector<double> command;
  std::vector<double> cmd_speed;
  std::vector<int8_t> robot_state;

  int64_t rows = 0;
  int64_t status_ok = 0;
  double max_lin_err_mm = 0.0;
  auto next = std::chrono::steady_clock::now();

  std::cout << "[probe] recording -> " << out_csv << " @ " << hz << " Hz\n"
            << "[probe] use teach pendant now: free 5mm step, then blocked 5mm step\n"
            << "[probe] Ctrl+C to stop\n";

  while (g_running) {
    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
    const auto loop_start = std::chrono::steady_clock::now();
    const auto lag_us = std::max<int64_t>(
        0, std::chrono::duration_cast<std::chrono::microseconds>(loop_start - next).count());
    const int64_t t0 = now_us();

    actual.clear();
    command.clear();
    cmd_speed.clear();
    robot_state.clear();

    bool ok_pose = false;
    bool ok_cmd = false;
    bool ok_spd = false;
    try {
      robot.get_tcp_pose(actual);
      ok_pose = actual.size() >= 6;
    } catch (const std::exception &e) {
      std::cerr << "[probe] get_tcp_pose failed: " << e.what() << "\n";
    }
    try {
      robot.get_tcp_pose_command(command);
      ok_cmd = command.size() >= 6;
    } catch (const std::exception &e) {
      std::cerr << "[probe] get_tcp_pose_command failed: " << e.what() << "\n";
    }
    try {
      robot.get_tcp_speed_command(cmd_speed);
      ok_spd = cmd_speed.size() >= 6;
    } catch (const std::exception &e) {
      std::cerr << "[probe] get_tcp_speed_command failed: " << e.what() << "\n";
    }
    try {
      robot.get_robot_state(robot_state);
    } catch (const std::exception &e) {
      std::cerr << "[probe] get_robot_state failed: " << e.what() << "\n";
    }
    fill6(actual);
    fill6(command);
    fill6(cmd_speed);

    std::vector<double> status_actual(6, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> status_expect(6, std::numeric_limits<double>::quiet_NaN());
    bool status_valid = false;
    int collision = 0;
    int collision_axis = -1;
    int robot_error = 0;
    int op_mode = robot_state.size() > 3 ? static_cast<int>(robot_state[3]) : -1;
    std::string source = "get_tcp_*";

    if (use_status) {
      try {
        DucoRPC::RobotStatusList st{};
        robot.getRobotStatus(st);
        if (st.cartActualPosition.size() >= 6 &&
            st.cartExpectPosition.size() >= 6) {
          std::copy_n(st.cartActualPosition.begin(), 6, status_actual.begin());
          std::copy_n(st.cartExpectPosition.begin(), 6, status_expect.begin());
          status_valid = true;
          ++status_ok;
        }
        collision = st.collision ? 1 : 0;
        collision_axis = static_cast<int>(st.collisionAxis);
        robot_error = static_cast<int>(st.robotError);
        source = "get_tcp_*+getRobotStatus";
      } catch (const std::exception &e) {
        std::cerr << "[probe] getRobotStatus failed: " << e.what() << "\n";
      }
    }

    const int64_t t1 = now_us();
    const int64_t ts = (t0 + t1) / 2;
    const int64_t rpc_duration_us = t1 - t0;

    const bool tcp_valid = ok_pose && ok_cmd;
    const double lin_err_mm = tcp_valid
                                  ? linear_error_mm(command, actual)
                                  : std::numeric_limits<double>::quiet_NaN();
    const double status_lin_err_mm =
        status_valid
            ? linear_error_mm(status_expect, status_actual)
            : std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(lin_err_mm) && lin_err_mm > max_lin_err_mm) {
      max_lin_err_mm = lin_err_mm;
    }

    ofs << ts << ',' << join6(actual) << ',' << join6(command) << ','
        << join6(cmd_speed) << ',' << join6(status_actual) << ','
        << join6(status_expect)
        << ',' << lin_err_mm << ',' << status_lin_err_mm << ',' << collision
        << ',' << collision_axis << ',' << robot_error << ',' << op_mode << ','
        << (tcp_valid ? 1 : 0) << ',' << (status_valid ? 1 : 0) << ','
        << rpc_duration_us << ',' << lag_us << ',' << source << '\n';
    ++rows;

    if (rows % static_cast<int64_t>(std::max(1.0, hz)) == 0) {
      std::cout << std::fixed << std::setprecision(3)
                << "[probe] rows=" << rows << " lin_err_mm=" << lin_err_mm
                << " status_lin_err_mm=" << status_lin_err_mm
                << " max_lin_err_mm=" << max_lin_err_mm
                << " collision=" << collision << " op_mode=" << op_mode
                << " ok_pose=" << ok_pose << " ok_cmd=" << ok_cmd
                << " ok_spd=" << ok_spd << " status_valid=" << status_valid
                << " rpc_ms=" << (rpc_duration_us / 1000.0) << "\n";
    }

    const auto loop_end = std::chrono::steady_clock::now();
    if (loop_end > next + std::chrono::duration_cast<std::chrono::steady_clock::duration>(period)) {
      // Do not burst RPC calls trying to catch up after a slow controller response.
      next = loop_end;
    } else {
      std::this_thread::sleep_until(next);
    }
  }

  ofs.flush();
  robot.close();

  std::cout << "[probe] stopped. rows=" << rows << " status_ok=" << status_ok
            << " max_lin_err_mm=" << max_lin_err_mm << "\n"
            << "[probe] csv: " << out_csv << "\n";
  return 0;
}
