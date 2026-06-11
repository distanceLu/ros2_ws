#!/usr/bin/env python3
from ctypes import *
import rospy
from std_msgs.msg import Int32MultiArray 

import roslib.packages

path = roslib.packages.get_pkg_dir('robot_control')


# 全局变量lib
lib = cdll.LoadLibrary( str(path) + '/../../lib/libMegmeetComm.so')


def Initialize(Channel,  Btr0Btr1 = 0x031C, HwType = 0, IOPort = 0, Interrupt = 0):
    return lib.Initialize(Channel,Btr0Btr1,HwType, IOPort, Interrupt)

def Handshake(Channel):
    return lib.MegmeetHandshakeCmd(Channel)
    

def Uninitialize(Channel):
    return lib.Uninitialize(Channel)


def MegmeetCmd(
        Channel,nId,
        #开始焊接
        bStart,
        # 机器人准备就绪
        bReady,
        # 电源工作模式，0. 直流一元化 1. 脉冲一元化 2. JOB 模式 3. 断续焊
        bMode,
        # 气体检测
        bGass,
        # 点动送丝
        bWire,
        # 反抽送丝
        bReverseWrie,
        # 电源故障复位
        bAlarmReset,
        # 寻位使能
        bEnableLocation,
        # JOB模式，JOB号
        bJobNum,
        # 焊接给定电流/送丝速度
        sCurrent,
        # 焊接给定电压/一元化修正值
        sVoltage
        ):
    return lib.MegmeetCmd(Channel,nId,bStart,bReady,bMode,bGass,bWire,bReverseWrie,bAlarmReset,bEnableLocation,bJobNum,sCurrent,sVoltage)


def MegmeetRecv(Channel,  nId):
    # 起弧成功
    bStart = c_bool()
    # 焊接状态
    bStatus = c_bool()
    # 焊接电源故障
    bAlarm = c_bool()
    # 通信就绪
    bCommReady = c_bool()
    # 故障代码
    bAlarmCode = c_byte()
    # 寻位成功
    bLocateSuceed = c_bool()
    # 送丝机正常
    bWireOK = c_bool()
    # 给定超范围
    bOutRange = c_bool()
    # 焊接实时电压
    sCurrent = c_ushort()
    # 焊接实时电流
    sVoltage = c_ushort()
    # 送丝机实时速度
    sWireSpeed = c_ushort()
    bRet = lib.MegmeetRecv(Channel,
                           nId,
                           byref(bStart),
                           byref(bStatus),
                           byref(bAlarm),
                           byref(bCommReady),
                           byref(bAlarmCode),
                           byref(bLocateSuceed),
                           byref(bWireOK),
                           byref(bOutRange),
                           byref(sCurrent),
                           byref(sVoltage),
                           byref(sWireSpeed))
    return bRet,bStart,bStatus,bAlarm,bCommReady,bAlarmCode,bLocateSuceed,bWireOK,bOutRange,sCurrent,sVoltage,sWireSpeed

'''
import time

def test():
    wStatus = Initialize(81);
    print(wStatus)
    cnt = 200 
    while True:
        #bSend = MegmeetCmd(81,0x0415,True,True,0,True,False,False,False,False,0,230,386)
        #print(bSend)
        #bRecv = MegmeetRecv(81,0x3C2)
        #print(bRecv)
        cnt = cnt -1 
        time.sleep(0.01)
        if cnt == 0 :
            break
        else:
            print(cnt)
    wStatus = Uninitialize(81)


test()
'''



import rospy
from std_msgs.msg import String  # 可替换为你的消息类型
from std_srvs.srv import Empty,EmptyResponse,EmptyRequest
from std_msgs.msg import Bool
from robot_control.srv import set_value,set_valueResponse
cnt = 0 
start = False
control = False

_welding_state_pub = None
def _get_welding_state_pub():
    global _welding_state_pub
    if _welding_state_pub is None:
        _welding_state_pub = rospy.Publisher(
            "/welding_start_signal", Bool, queue_size=1, latch=True
        )
        # 给 publisher 一点时间注册
        rospy.sleep(0.05)
    return _welding_state_pub

def handle_megmeet(req):
    global cnt,control
    control = True
    test_func()
    control = False
    cnt = 25 # 1s  
    return EmptyResponse()

def handle_wire(req):
    global control
    cnt = 60 
    rate = rospy.Rate(30)
    print('1')
    while not rospy.is_shutdown():
        cnt = cnt -1 
        #送气 两秒
        bSend = MegmeetCmd(81,0x0415,False,True,0,True,True,False,False,False,0,200,410)
        rate.sleep()
        bRecv = MegmeetRecv(81,0x3C2)
        print('handle wire recv: ', bRecv)
        if cnt == 0:
            print('done')
            break
    return EmptyResponse()


def set_vol_cb(req):
    global param_V
    param_V = int(req.data)
    print('setting vol ',param_V)
    return set_valueResponse()

def set_current_cb(req):
    global param_A
    param_A = int(req.data)
    print('setting current', param_A)
    return set_valueResponse()


def test_func():
    print('test_fun() 送气')
    cnt = 60 
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        cnt = cnt -1 
        #送气 两秒
        MegmeetCmd(81,0x0415,False,True,4,True,False,False,False,False,0,0,0)
        rate.sleep()
        bRet,bStart,bStatus,bAlarm,bCommReady,bAlarmCode,bLocateSuceed,bWireOK,bOutRange,sCurrent,sVoltage,sWireSpeed = MegmeetRecv(81,0x3C2)
        if cnt == 0:
            break

def test_func3():
    cnt = 60 
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        cnt = cnt -1 
        #送气 两秒
        if cnt > 30 :
            MegmeetCmd(81,0x0415,False,True,4,True,False,True,False,False,0,0,0)
        else:
            MegmeetCmd(81,0x0415,False,True,4,True,True,False,False,False,0,0,0)
        
        bRet,bStart,bStatus,bAlarm,bCommReady,bAlarmCode,bLocateSuceed,bWireOK,bOutRange,sCurrent,sVoltage,sWireSpeed = MegmeetRecv(81,0x3C2)
        rate.sleep()
        if cnt == 0:
            break



is_enable_megmeet = True # wuxy add 是否发送起弧命令
param_A = 155 #   6+4 160
param_V = 180 #   6+4 180
# 130 160 

def handle_start() -> Bool:
    print('handle_start')
    rate = rospy.Rate(30)
    local_cnt = 0
    timeout_cnt = 300
    while not rospy.is_shutdown():
        #起弧
        # bSend = MegmeetCmd(81,0x0415,True,True,4,True,False,False,False,False,0, 210,230)
        bSend = MegmeetCmd(81,0x0415,True,True,4,True,False,False,False,False,0, param_A, param_V)

        rate.sleep()
        bRet,bStart,bStatus,bAlarm,bCommReady,bAlarmCode,bLocateSuceed,bWireOK,bOutRange,sCurrent,sVoltage,sWireSpeed = MegmeetRecv(81,0x3C2)
        # print('1start:',bStart)
        # print('1status:',bStatus)
        #起弧成功 返回  否则一直阻塞等待
        local_cnt += 1
        if local_cnt == timeout_cnt:
            rospy.logwarn('[MEGMEET] megmeet start time out')
            return False

        if bStart.value :
            rospy.loginfo('[MEGMEET] megmeet start ok')
            break
    return True



def handle_stop():
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        #收弧
        bSend = MegmeetCmd(81,0x0415,False,True,4,False,False,False,False,False,0,0,0)
        rate.sleep()
        bRet,bStart,bStatus,bAlarm,bCommReady,bAlarmCode,bLocateSuceed,bWireOK,bOutRange,sCurrent,sVoltage,sWireSpeed = MegmeetRecv(81,0x3C2)
        #焊接状态 false 返回  否则一直阻塞等待
        # print('status',bStatus)
        # print('start',bStart)
        if bStatus.value == False:
            rospy.loginfo('[MEGMEET] megmeet stop ok')
            break



def send_signal_welding_start():
    pub = _get_welding_state_pub()
    pub.publish(Bool(data = True))
    rospy.loginfo('[MEGMEET] publish /welding_start_signal: True')

def send_signale_welding_end():
    pub = _get_welding_state_pub()
    pub.publish(Bool(data = False))
    rospy.loginfo('[MEGMEET] publish /welding_start_signal: False')


def megmeet_start(req):
    rospy.loginfo('[MEGMEET] megmeet start signal recv!')
    global cnt,control, is_enable_megmeet

    if is_enable_megmeet == False:
        # 仅发送起弧成功信号，不控制焊机
        send_signal_welding_start()
        return EmptyResponse()

    control = True

    #预先送气两秒
    test_func()
    #一边起弧一边送气
    start_ret = handle_start()

    if start_ret == False:
        # 超时
        return EmptyResponse()

    # 起弧成功 -> 发布 True
    send_signal_welding_start()

    control = False
    cnt = 8888 # 1s  
    return EmptyResponse()




def megmeet_stop(req):
    rospy.loginfo('[MEGMEET] megmeet stop signal recv!')
    global cnt,control,is_enable_megmeet

    if is_enable_megmeet == False:
        # 仅发送收弧信号，不控制焊机
        send_signale_welding_end()
        return EmptyResponse()

    cnt = 0 # 1s 
    control = True

    #送气两秒
    #test_func3()
    handle_stop()

    # 收弧成功 -> 发布False
    send_signale_welding_end()
     
    control = False
    

    # reset welding setting 

    try:
        # 创建 service proxy
        reset_setting = rospy.ServiceProxy('/reset_welding_setting', Empty)
        req = EmptyRequest()
        # 方式1：直接调用（推荐）
        resp = reset_setting(req)     
        return EmptyResponse()
        
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)
        return EmptyResponse()


def test_func2():
    cnt = 30 
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        cnt = cnt -1 
        #送气 1秒
        bSend = MegmeetCmd(81,0x0415,True,True,0,True,False,False,False,False,0,230,386)
        #print(bSend)
        bRecv = MegmeetRecv(81,0x3C2)
        #print(bRecv)
        cnt = cnt -1 
        if cnt == 0:
            break 






import time

def talker():
    global cnt,control

    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)

    rospy.loginfo(f"[MEGMEET] params loaded: A={param_A}, V={param_V}")


    wStatus = Initialize(81)
    Handshake(81)
    print('[MEGMEET] !!!!!! test wStatus: ', wStatus)

    rate = rospy.Rate(30)  # 30Hz频率

    s = rospy.Service('test_megmeet', Empty, handle_megmeet)
    s1 = rospy.Service('megmeet_start', Empty, megmeet_start)
    s2 = rospy.Service('megmeet_stop', Empty, megmeet_stop)
    s2 = rospy.Service('send_wire', Empty, handle_wire)
    s3 = rospy.Service('set_vol', set_value, set_vol_cb)
    s4 = rospy.Service('set_current', set_value, set_current_cb)
    
    welding_pub = rospy.Publisher('/welding_state', Int32MultiArray, queue_size=1)
    rospy.set_param('/welding_current', param_A)
    rospy.set_param('/welding_vol', param_V)
    ccc = 0
    while not rospy.is_shutdown():
        # 发送电流电压到topic /welding_state
        welding_pub.publish(Int32MultiArray(data=[int(param_A), int(param_V)]))


        #bRecv = MegmeetRecv(81,0x3C2)
        '''
        if start == True:
            print('gas....')
            test_func()
        '''
        if cnt > 0 :
                                                                                    #电流 490 30%
            #bSend = MegmeetCmd(81,0x0415,True,True,0,True,False,False,False,False,0, 180,210)
            
            # bSend = MegmeetCmd(81,0x0415,True,True,4,True,False,False,False,False,0, 210,230)
            bSend = MegmeetCmd(81,0x0415,True,True,4,True,False,False,False,False,0, param_A,param_V)
             
            #bSend = MegmeetCmd(81,0x0415,False,True,0,True,True,False,False,False,0,200,410)

            if cnt == 8888:
                cnt = cnt + 1 
                
            cnt = cnt -1 
            bRecv = MegmeetRecv(81,0x3C2)
        else:
            if control == False:
                #300 18.2，400 21 ，700 20.8
                bSend = MegmeetCmd(81,0x0415,False,True,4,False,False,False,False,False,0,param_A,param_V)
                recv = MegmeetRecv(81,0x3C2)
                if recv[0] == False:
                    # print('recv failed! hand shake')
                    Handshake(81)
                # print("----")
                #print(recv)

        ccc= ccc+1
        # if ccc%30 == 0: 
        #     print(ccc,cnt)
        rate.sleep()
    wStatus = Uninitialize(81)
    

if __name__ == '__main__':


    try:
        talker()
    except rospy.ROSInterruptException:
        pass

