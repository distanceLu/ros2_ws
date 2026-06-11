#include "robot_control/DucoCobot.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <unistd.h>
#include <vector>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include "robot_control/msg/joint_pos.hpp"
#include "robot_control/msg/jog_pos.hpp"
#include <std_srvs/srv/empty.hpp>
#include "robot_control/srv/move.hpp"
#include "robot_control/srv/speed.hpp"
#include "robot_control/srv/speedl.hpp"
#include "robot_control/srv/special_speedl.hpp"
#include "robot_control/srv/track.hpp"

using namespace std;
using namespace DucoRPC;

class RobotDriver : public rclcpp::Node
{
public:
    RobotDriver() : Node("robot_driver")
    {
        // 声明参数
        this->declare_parameter<std::string>("robot_ip", "192.168.1.10");
        this->declare_parameter<int>("publish_rate", 90);
        
        ip_ = this->get_parameter("robot_ip").as_string();
        rate_ = this->get_parameter("publish_rate").as_int();
        global_speed_ = 1.0;
        speedl_flag_ = false;
        
        RCLCPP_INFO(this->get_logger(), "robot connect ip: %s", ip_.c_str());
        RCLCPP_INFO(this->get_logger(), "robot publish rate: %d", rate_);
        
        // 初始化机械臂连接
        robot_point_ = std::make_unique<DucoCobot>(ip_, 7003);
        
        try
        {
            int result = robot_point_->open();
            RCLCPP_INFO(this->get_logger(), "open: %d", result);
            
            result = robot_point_->power_on(true);
            RCLCPP_INFO(this->get_logger(), "power_on: %d", result);
            
            result = robot_point_->enable(true);
            RCLCPP_INFO(this->get_logger(), "enable: %d", result);
            
            vector<int8_t> state;
            robot_point_->get_robot_state(state);
            RCLCPP_INFO(this->get_logger(), "state size: %zu", state.size());
            
            robot_point_->stop(true);
            robot_point_->trackClearQueue();
            
            // 创建发布者
            pos_pub_ = this->create_publisher<robot_control::msg::JogPos>("tool_pos", 10);
            flange_pub_ = this->create_publisher<robot_control::msg::JogPos>("flange_pos", 10);
            joint_pos_pub_ = this->create_publisher<robot_control::msg::JointPos>("joint_pos", 10);
            
            // 创建服务
            reset_service_ = this->create_service<std_srvs::srv::Empty>(
                "reset", std::bind(&RobotDriver::resetCallback, this, 
                std::placeholders::_1, std::placeholders::_2));
            
            movej_service_ = this->create_service<robot_control::srv::Move>(
                "movj", std::bind(&RobotDriver::movjCallback, this,
                std::placeholders::_1, std::placeholders::_2));
            
            movejog_service_ = this->create_service<robot_control::srv::Move>(
                "mov_jog", std::bind(&RobotDriver::movjogCallback, this,
                std::placeholders::_1, std::placeholders::_2));
            
            set_speed_service_ = this->create_service<robot_control::srv::Speed>(
                "set_speed", std::bind(&RobotDriver::setspeedCallback, this,
                std::placeholders::_1, std::placeholders::_2));
            
            speedl_service_ = this->create_service<robot_control::srv::Speedl>(
                "speedl", std::bind(&RobotDriver::speedlCallback, this,
                std::placeholders::_1, std::placeholders::_2));
            
            speedstop_service_ = this->create_service<std_srvs::srv::Empty>(
                "speed_stop", std::bind(&RobotDriver::speedstopCallback, this,
                std::placeholders::_1, std::placeholders::_2));
            
            mov_tcp_service_ = this->create_service<robot_control::srv::Move>(
                "mov_tcp", std::bind(&RobotDriver::movtcpCallback, this,
                std::placeholders::_1, std::placeholders::_2));
            
            mov_tcp_service_with_name_ = this->create_service<robot_control::srv::Move>(
                "mov_tcp_s", std::bind(&RobotDriver::movtcpCallback_with_name, this,
                std::placeholders::_1, std::placeholders::_2));
            
            special_speedl_service_ = this->create_service<robot_control::srv::SpecialSpeedl>(
                "speedl_s", std::bind(&RobotDriver::speedl_control_Callback, this,
                std::placeholders::_1, std::placeholders::_2));
            
            track_service_ = this->create_service<robot_control::srv::Track>(
                "track", std::bind(&RobotDriver::trackCallback, this,
                std::placeholders::_1, std::placeholders::_2));
            
            // 创建定时器
            timer_ = this->create_wall_timer(
                std::chrono::milliseconds(1000 / rate_),
                std::bind(&RobotDriver::timerCallback, this));
            
            // 初始化 TF 广播器
            tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Initialization error: %s", e.what());
            if (robot_point_)
                robot_point_->close();
        }
    }
    
    ~RobotDriver()
    {
        if (robot_point_)
            robot_point_->close();
    }

private:
    // 服务回调函数
    void setspeedCallback(const robot_control::srv::Speed::Request::SharedPtr req,
                          robot_control::srv::Speed::Response::SharedPtr res)
    {
        global_speed_ = req->speed;
        RCLCPP_INFO(this->get_logger(), "Global speed set to: %.2f", global_speed_);
    }
    
    void resetCallback(const std_srvs::srv::Empty::Request::SharedPtr req,
                       std_srvs::srv::Empty::Response::SharedPtr res)
    {
        if (robot_point_)
        {
            robot_point_->disable(true);
            sleep(1);
            robot_point_->enable(true);
            RCLCPP_INFO(this->get_logger(), "Reset done");
        }
    }
    
    void movjCallback(const robot_control::srv::Move::Request::SharedPtr req,
                      robot_control::srv::Move::Response::SharedPtr res)
    {
        std::vector<double> joints;
        joints.push_back(req->a);
        joints.push_back(req->b);
        joints.push_back(req->c);
        joints.push_back(req->d);
        joints.push_back(req->e);
        joints.push_back(req->f);
        
        int result = 0;
        if (robot_point_)
            result = robot_point_->movej2(joints, 0.5 * global_speed_, 0.5 * global_speed_, 0, req->block);
        
        if (result == 4)
            RCLCPP_INFO(this->get_logger(), "movj success");
        else
            RCLCPP_ERROR(this->get_logger(), "movj result: %d", result);
    }
    
    void movjogCallback(const robot_control::srv::Move::Request::SharedPtr req,
                        robot_control::srv::Move::Response::SharedPtr res)
    {
        if (!robot_point_) return;
        
        std::vector<double> pos;
        pos.push_back(req->a);
        pos.push_back(req->b);
        pos.push_back(req->c);
        pos.push_back(req->d);
        pos.push_back(req->e);
        pos.push_back(req->f);
        
        std::vector<double> q_near;
        std::vector<double> tool;
        std::vector<double> wobj;
        std::vector<double> joint_pos;
        
        robot_point_->cal_ikine(joint_pos, pos, q_near, tool, wobj);
        
        if (joint_pos.size() != 6) return;
        
        int result = robot_point_->movej2(joint_pos, 0.5 * global_speed_, 0.5 * global_speed_, 0, req->block);
        
        if (result == 4)
            RCLCPP_INFO(this->get_logger(), "movjog success");
        else
            RCLCPP_ERROR(this->get_logger(), "movjog result: %d", result);
    }
    
    void speedlCallback(const robot_control::srv::Speedl::Request::SharedPtr req,
                        robot_control::srv::Speedl::Response::SharedPtr res)
    {
        if (!robot_point_) return;
        
        if (speedl_flag_)
            robot_point_->speed_stop(true);
        
        std::vector<double> speed_array;
        speed_array.push_back(req->x);
        speed_array.push_back(req->y);
        speed_array.push_back(req->z);
        speed_array.push_back(req->rx);
        speed_array.push_back(req->ry);
        speed_array.push_back(req->rz);
        
        int task_id = robot_point_->speedl(speed_array, 1.0, -1, false);
        int state = robot_point_->get_noneblock_taskstate(task_id);
        
        speedl_flag_ = true;
        RCLCPP_INFO(this->get_logger(), "speedl called, task_id: %d, state: %d", task_id, state);
    }
    
    void speedl_control_Callback(const robot_control::srv::SpecialSpeedl::Request::SharedPtr req,
                                  robot_control::srv::SpecialSpeedl::Response::SharedPtr res)
    {
        if (!robot_point_) return;
        
        std::vector<double> speed_array;
        speed_array.push_back(req->x);
        speed_array.push_back(req->y);
        speed_array.push_back(req->z);
        speed_array.push_back(req->rx);
        speed_array.push_back(req->ry);
        speed_array.push_back(req->rz);
        
        int task_id = robot_point_->speedl(speed_array, 1.0, req->time, false);
        int state = robot_point_->get_noneblock_taskstate(task_id);
        
        RCLCPP_INFO(this->get_logger(), "special_speedl called, task_id: %d", task_id);
    }
    
    void speedstopCallback(const std_srvs::srv::Empty::Request::SharedPtr req,
                           std_srvs::srv::Empty::Response::SharedPtr res)
    {
        if (robot_point_)
        {
            robot_point_->speed_stop(true);
            speedl_flag_ = false;
            RCLCPP_INFO(this->get_logger(), "speed stop done");
        }
    }
    
    void movtcpCallback(const robot_control::srv::Move::Request::SharedPtr req,
                        robot_control::srv::Move::Response::SharedPtr res)
    {
        std::vector<double> joints;
        joints.push_back(req->a);
        joints.push_back(req->b);
        joints.push_back(req->c);
        joints.push_back(req->d);
        joints.push_back(req->e);
        joints.push_back(req->f);
        
        int result = 0;
        if (robot_point_)
            result = robot_point_->tcp_move(joints, 0.1 * global_speed_, 0.05 * global_speed_, 0, "", req->block);
        
        if (result == 4)
            RCLCPP_INFO(this->get_logger(), "mov_tcp success");
        else
            RCLCPP_ERROR(this->get_logger(), "mov_tcp result: %d", result);
    }
    
    void movtcpCallback_with_name(const robot_control::srv::Move::Request::SharedPtr req,
                                   robot_control::srv::Move::Response::SharedPtr res)
    {
        std::vector<double> joints;
        joints.push_back(req->a);
        joints.push_back(req->b);
        joints.push_back(req->c);
        joints.push_back(req->d);
        joints.push_back(req->e);
        joints.push_back(req->f);
        
        int result = 0;
        if (robot_point_)
            result = robot_point_->tcp_move(joints, 0.1 * global_speed_, 0.05 * global_speed_, 0, req->name, req->block);
        
        if (result == 4)
            RCLCPP_INFO(this->get_logger(), "mov_tcp_s success");
        else
            RCLCPP_ERROR(this->get_logger(), "mov_tcp_s result: %d", result);
    }
    
    void trackCallback(const robot_control::srv::Track::Request::SharedPtr req,
                       robot_control::srv::Track::Response::SharedPtr res)
    {
        if (!robot_point_) return;
        
        robot_point_->trackClearQueue();
        robot_point_->disable_vibration_control();
        std::vector<PointOP> pos_list;
        
        RCLCPP_INFO(this->get_logger(), "Track callback called with %zu points", req->points.size());
        
        if (req->points.size() > 0)
        {
            auto p = req->points[0];
            RCLCPP_INFO(this->get_logger(), "First point: %.2f %.2f %.2f %.2f %.2f %.2f",
                        p.x, p.y, p.z, p.rx, p.ry, p.rz);
        }
        
        for (size_t i = 0; i < req->points.size(); i++)
        {
            PointOP posdata;
            std::vector<double> point;
            point.push_back(req->points[i].x);
            point.push_back(req->points[i].y);
            point.push_back(req->points[i].z);
            point.push_back(req->points[i].rx);
            point.push_back(req->points[i].ry);
            point.push_back(req->points[i].rz);
            
            posdata.pos = point;
            posdata.vel = req->speed;
            
            if (i == 0)
                posdata.blend_time = 100;
            else
            {
                auto sign = req->points[i].x - req->points[i-1].x;
                if (sign < 0.0)
                    posdata.blend_time = 90;
                else
                    posdata.blend_time = 100;
            }
            
            pos_list.push_back(posdata);
        }
        
        robot_point_->track_enqueue_op_vel(pos_list, true);
        RCLCPP_INFO(this->get_logger(), "Track running");
        robot_point_->track_cart_vel_motion(req->acc, "", "", true);
        RCLCPP_INFO(this->get_logger(), "Track done");
    }
    
    void timerCallback()
    {
        if (!robot_point_) return;
        
        std::vector<double> pos;
        std::vector<double> joint_pos;
        std::vector<double> flange_pos;
        
        // 获取并发布 TCP 位姿
        robot_point_->get_tcp_pose(pos);
        if (pos.size() >= 6)
        {
            robot_control::msg::JogPos pose_msg;
            pose_msg.x = pos[0];
            pose_msg.y = pos[1];
            pose_msg.z = pos[2];
            pose_msg.rx = pos[3];
            pose_msg.ry = pos[4];
            pose_msg.rz = pos[5];
            pos_pub_->publish(pose_msg);
        }
        
        // 获取并发布关节角度
        robot_point_->get_actual_joints_position(joint_pos);
        if (joint_pos.size() >= 6)
        {
            robot_control::msg::JointPos j;
            j.j1 = joint_pos[0];
            j.j2 = joint_pos[1];
            j.j3 = joint_pos[2];
            j.j4 = joint_pos[3];
            j.j5 = joint_pos[4];
            j.j6 = joint_pos[5];
            joint_pos_pub_->publish(j);
        }
        
        // 获取并发布法兰位姿
        robot_point_->get_flange_pose(flange_pos);
        if (flange_pos.size() >= 6)
        {
            robot_control::msg::JogPos flange;
            flange.x = flange_pos[0];
            flange.y = flange_pos[1];
            flange.z = flange_pos[2];
            flange.rx = flange_pos[3];
            flange.ry = flange_pos[4];
            flange.rz = flange_pos[5];
            flange_pub_->publish(flange);
        }
        
        // 发布 TF 变换
        if (pos.size() >= 6)
        {
            geometry_msgs::msg::TransformStamped t;
            t.header.stamp = this->now();
            t.header.frame_id = "base";
            t.child_frame_id = "tcp";
            t.transform.translation.x = pos[0];
            t.transform.translation.y = pos[1];
            t.transform.translation.z = pos[2];
            
            tf2::Quaternion quat;
            quat.setRPY(pos[3], pos[4], pos[5]);
            t.transform.rotation.x = quat.x();
            t.transform.rotation.y = quat.y();
            t.transform.rotation.z = quat.z();
            t.transform.rotation.w = quat.w();
            
            tf_broadcaster_->sendTransform(t);
        }
    }
    
    // 成员变量
    std::unique_ptr<DucoCobot> robot_point_;
    std::string ip_;
    int rate_;
    double global_speed_;
    bool speedl_flag_;
    
    rclcpp::Publisher<robot_control::msg::JogPos>::SharedPtr pos_pub_;
    rclcpp::Publisher<robot_control::msg::JogPos>::SharedPtr flange_pub_;
    rclcpp::Publisher<robot_control::msg::JointPos>::SharedPtr joint_pos_pub_;
    
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_service_;
    rclcpp::Service<robot_control::srv::Move>::SharedPtr movej_service_;
    rclcpp::Service<robot_control::srv::Move>::SharedPtr movejog_service_;
    rclcpp::Service<robot_control::srv::Speed>::SharedPtr set_speed_service_;
    rclcpp::Service<robot_control::srv::Speedl>::SharedPtr speedl_service_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr speedstop_service_;
    rclcpp::Service<robot_control::srv::Move>::SharedPtr mov_tcp_service_;
    rclcpp::Service<robot_control::srv::Move>::SharedPtr mov_tcp_service_with_name_;
    rclcpp::Service<robot_control::srv::SpecialSpeedl>::SharedPtr special_speedl_service_;
    rclcpp::Service<robot_control::srv::Track>::SharedPtr track_service_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotDriver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
