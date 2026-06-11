#!/usr/bin/env python3
import rospy
import HaplyHardwareAPI
import threading
from std_msgs.msg import UInt8

class HaplyHandleNode:
    def __init__(self):
        rospy.init_node('haply_handle_node', anonymous=False)
        self.handle_rate_hz = 1000

        self.versegrip = None
        self.handle_stream = None
        self.running = False

        # 发布按键消息 uint8
        self.btn_pub = rospy.Publisher('/haply_btn', UInt8, queue_size=10)

        self._prev_buttons = None

        connected_handles = HaplyHardwareAPI.detect_handles()
        if not connected_handles:
            rospy.loginfo('detect_handles disabled.')


            # connected_handles = ['/dev/ttyACM0']
        rospy.loginfo('\n[HAPLY HANDLE] connected_handles: %s', connected_handles)
        if not connected_handles:
            rospy.logerr('[HAPLY HANDLE] No Haply handles detected. '
                         'Check USB connection, permissions (dialout), and power.')
            return
        self.handle_stream = HaplyHardwareAPI.SerialStream(connected_handles[0])
        self.versegrip = HaplyHardwareAPI.Handle(self.handle_stream)

        try:
            init_resp = self.versegrip.GetVersegripStatus()
            self._prev_buttons = int(init_resp['buttons'])
            rospy.loginfo('[HAPLY HANDLE] initial buttons: %s', self._prev_buttons)
        except Exception as e:
            rospy.logwarn('[HAPLY HANDLE] initial read failed: %s', str(e))
            self._prev_buttons = 0

        self.running = True
        self._reader_thread = threading.Thread(
            target = self._loop_read_handle, daemon = True
        )
        self._reader_thread.start()

        # while True:
        #     response = self.versegrip.GetVersegripStatus()
        #     print('[HAPLY HANDLE] buttons: ', response['buttons'])        

        rospy.on_shutdown(self._on_shutdown)

    def _loop_read_handle(self):
        rate = rospy.Rate(self.handle_rate_hz)
        while not rospy.is_shutdown() and self.running:
            try:
                response = self.versegrip.GetVersegripStatus()
                # print('[HAPLY HANDLE] buttons: ', response['buttons'])
                curr = int(response['buttons'])

                if self._prev_buttons is not None:
                    if self._prev_buttons == 0 and curr != 0:
                        self.btn_pub.publish(UInt8(curr))
                        rospy.loginfo('[HAPLY HANDLE] publish btn %d', curr)

                self._prev_buttons = curr

            except Exception as e:
                # 设备临时超时/串口被占用等，做节流告警
                rospy.logwarn_throttle(1.0, "[HAPLY HANDLE] read failed: %s", str(e))
            rate.sleep()

    def _on_shutdown(self):
        self.running = False
        try:
            if hasattr(self, '_reader_thread') and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1.0)
        except Exception:
            pass    

if __name__ == '__main__':
    node = HaplyHandleNode()
    rospy.spin()


