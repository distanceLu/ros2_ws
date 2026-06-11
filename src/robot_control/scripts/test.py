from ctypes import *
import rospy




# 全局变量lib
lib = cdll.LoadLibrary('/home/k/catkin_ws/src/robot_control/lib/libMegmeetComm.so')


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


import time

def test():
    wStatus = Initialize(81);
    print(wStatus)
    cnt = 2000 
    while True:
    	bSend = MegmeetCmd(81,0x0415,False,True,4,False,False,False,False,False,0,180,210)
    	time.sleep(0.005)
    	recv = MegmeetRecv(81,0x3C2)
    	print(recv)
    	cnt = cnt -1 
    	if cnt == 0:
           break
    	
    	
    	
    cnt = 400 
    while True:
        #起弧
        bSend = MegmeetCmd(81,0x0415,True,True,4,True,False,False,False,False,0,230,386)
        print(bSend)
        bRecv = MegmeetRecv(81,0x3C2)
        print(bRecv)
        cnt = cnt -1 
        time.sleep(0.005)
        if cnt == 0 :
            break
        else:
            print(cnt)
    wStatus = Uninitialize(81)


test()




