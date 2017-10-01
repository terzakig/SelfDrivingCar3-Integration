import rospy

from pid import PID
from yaw_controller import YawController


LAT_JERK_LIMIT = 3.0

class LatController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel

        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle

        # init feed forward yaw-rate control
        self.yawControl = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # init PID
        self.steer_PID = PID(kp=0.1, ki=0.001, kd=0.5)
        
    def control(self,target_spd, target_yawRate, current_spd, CTE, delta_t):
        
        # feed forward control to drive curvature of road
        steer_feedForward = self.yawControl.get_steering(target_spd, target_yawRate, current_spd)
        # limit steering angle
        steer_feedForward = max(min(steer_feedForward, self.max_angle), self.min_angle)
        
        # PID control
        steer_PID = self.steer_PID.step(CTE, delta_t, mn=self.min_angle-steer_feedForward, mx=self.max_angle-steer_feedForward)
        
        # steering command
        steer = steer_feedForward + steer_PID
        
        return steer
        
    def reset(self):
        self.steer_PID.reset()
