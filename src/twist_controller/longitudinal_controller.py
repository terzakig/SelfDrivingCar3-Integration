import rospy

from pid import PID
from lowpass import LowPassFilter

FILT_TAU_ACCEL = 0.1 # filter time constant for low pass filter long. acceleration
TS = 0.05 # cycle time for 20 Hz 

LONG_JERK_LIMIT = 3.0

class LongController(object):
    def __init__(self,vehicle_mass,brake_deadband,decel_limit,accel_limit,wheel_radius):
        self.vehicle_mass = vehicle_mass
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.max_trq = vehicle_mass * accel_limit * wheel_radius # max throttle torque
        self.min_trq = vehicle_mass * decel_limit * wheel_radius # max brake torque (negative!)
        self.trq_brake_deadband = -1.0 * vehicle_mass * brake_deadband * wheel_radius 
        
        # init low pass filter for acceleration
        self.accel_LowPassFilter = LowPassFilter(FILT_TAU_ACCEL,TS)
        self.last_spd = 0.0
        
        self.last_target_accel = 0.0
        
        self.last_brake_actv = False # brake active
        
        # init PID
        self.accel_PID = PID(kp=0.0, ki=0.005, kd=0.2)
        
    def control(self,target_spd,current_spd,delta_t):
        
        # calulate filter acceleration of vehicle
        accel_raw = (current_spd - self.last_spd) / delta_t
        accel_filt = self.accel_LowPassFilter.filt(accel_raw)
        
        # calculate speed error
        spd_err = target_spd - current_spd
        # calulate target acceleration from speed error
        # could be tuned
        k_accel = 0.2
        target_accel = k_accel * spd_err * spd_err * spd_err
        
        # limit jerk
        accel_change_limit = LONG_JERK_LIMIT * delta_t
        target_accel = max(min(target_accel, self.last_target_accel+accel_change_limit), self.last_target_accel-accel_change_limit)
        
        # check for min and max allowed acceleration
        target_accel = max( min(target_accel, self.accel_limit), self.decel_limit)
        
        # calculate torque from target_accel using mass
        trq_feedForward = target_accel * self.vehicle_mass * self.wheel_radius
        
        # use PID to get better control performance
        accel_err = target_accel - accel_filt
        trq_PID = self.accel_PID.step(accel_err, delta_t, mn=self.min_trq-trq_feedForward, mx=self.max_trq-trq_feedForward)
        
        # calulate overall torque
        trq = trq_feedForward + trq_PID
        
        # calulate throttle
        # published throttle is a percentage
        # guess 100% is accel_limit
        if trq > 0:
            throttle = trq / self.max_trq * 100.0
            brake = 0.0
            self.last_brake_actv = False
        elif self.last_brake_actv:
            throttle = 0.0
            brake = -trq
            self.last_brake_actv = True
        elif trq < self.trq_brake_deadband:
            throttle = 0.0
            brake = -trq
            self.last_brake_actv = True
        else:
            throttle = 0.0
            brake = 0.0
            self.last_brake_actv = False
            
        # output for debug
        #rospy.logwarn("accel_raw: %f" % accel_raw + "; accel_filt: %f" % accel_filt + "; target_accel: %f" % target_accel + "; current_spd: %f" % current_spd)
            
        # write values for next iteration
        self.last_spd = current_spd
        self.last_target_accel = target_accel
        
        return throttle, brake
        
    def reset(self):
        self.accel_PID.reset()
