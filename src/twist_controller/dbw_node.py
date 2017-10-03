#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import Lane

import math
import numpy as np

from longitudinal_controller import LongController
from lateral_controller import LatController

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # below this speed no control any more
        # and we hold the vehicle if target speed is zero 
        min_speed = 0.1


        # init controller
        self.longControl = LongController(vehicle_mass,brake_deadband,decel_limit,accel_limit,wheel_radius)
        self.latControl = LatController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # init variables
        self.dbw_enabled = False
        self.velocity = None
        self.pose = None
        self.twist = None
        self.waypoints = None
        self.last_timestamp = 0.0

        # Subscribe to needed topics
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_cb, queue_size=1)
        rospy.Subscriber('/final_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb, queue_size=1)
	
        
        self.loop()

    def loop(self):
        rate = rospy.Rate(20) # 20Hz
        while not rospy.is_shutdown():
            self.control_step()
            rate.sleep()

    def control_step(self):
        now = rospy.get_rostime()
        timestamp = now.to_sec()
        delta_t = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        flag_dataRX = self.velocity is not None and \
                      self.pose is not None and \
                      self.twist is not None and \
                      self.waypoints is not None
                      
        throttle = 0.0
        brake = 0.0
        steer = 0.0

        #rospy.logwarn("delta_t: %f" % delta_t)

        if flag_dataRX and delta_t >0:
            current_spd = self.velocity.linear.x
            if self.dbw_enabled:
                # longitudinal control
                target_spd = self.twist.linear.x
                throttle, brake =  self.longControl.control(target_spd,current_spd,delta_t)
                
                #rospy.logwarn("target_spd: %f" % target_spd + "; current_spd: %f" % current_spd)
                #rospy.logwarn("throttle: %f" % throttle + "; brake: %f" % brake)
                
                # lateral control
                target_yawRate = self.twist.angular.z
                CTE = self.calc_CTE(self.waypoints, self.pose)
                #rospy.logwarn("CTE: %f" % CTE)
                steer = self.latControl.control(target_spd, target_yawRate, current_spd, CTE, delta_t)
                
            else:
                self.longControl.reset(current_spd)
                self.latControl.reset()
                
        else:
            self.longControl.reset(0.0)
            self.latControl.reset()
        
            

        self.publish(throttle, brake, steer)

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

    def dbw_cb(self, message):
        # extract dbw_enabled variable
        self.dbw_enabled = bool(message.data)

    def velocity_cb(self, message):
        # extract velocity
        self.velocity = message.twist

    def pose_cb(self, message):
        # extract position
        self.pose = message.pose

    def twist_cb(self, message):
        # extract the twist message """
        self.twist = message.twist

    def waypoints_cb(self, message):
        # extract waypoints
        self.waypoints = message.waypoints
        
    def transfromWPcarCoord(self, waypoints, pose):
        n = len(waypoints)
         # get car's x and y position and heading angle
        car_x = pose.position.x
        car_y = pose.position.y
        # get orientation
        s = self.pose.orientation.w # quaternion scalar
        v1 = self.pose.orientation.x # vector part 1
        v2 = self.pose.orientation.y # vector part 2
        v3 = self.pose.orientation.z # vector part 3        
        
        # now obtaining orientation of the car (assuming rotation about z: [0;0;1])
        car_theta = 2 * np.arccos(s)
        # Constraining the angle in [-pi, pi)
        if car_theta > np.pi:
            car_theta = -(2 * np.pi - car_theta)
        #car_theta = pose.orientation.z
       
        # transform waypoints in vehicle coordiantes
        wp_carCoord_x = np.zeros(n)
        wp_carCoord_y = np.zeros(n)

        for i in range(n):
            wp_x = waypoints[i].pose.pose.position.x
            wp_y = waypoints[i].pose.pose.position.y            
            wp_carCoord_x[i] = (wp_y-car_y)*math.sin(car_theta)-(car_x-wp_x)*math.cos(car_theta)
            wp_carCoord_y[i] = (wp_y-car_y)*math.cos(car_theta)-(wp_x-car_x)*math.sin(car_theta)
            
        return wp_carCoord_x, wp_carCoord_y
                
    def calc_CTE(self, waypoints, pose):
        # transfrom waypoints into vehicle coordinates
        wp_carCoord_x, wp_carCoord_y = self.transfromWPcarCoord(waypoints, pose)
        
        # get waypoint which should be used for controller input
        idxNearestWPfront = self.findNearestWPfront(wp_carCoord_x)
                
        # use only 20 first points
        n_points = 20
        n = min(n_points, len(wp_carCoord_x-idxNearestWPfront))

        if (n<n_points):
            rospy.logerr('dbw_node: Not enough waypots received for lateral control!')
            return 0.0
        elif (idxNearestWPfront<0):
            rospy.logerr('dbw_node: No waypoint in front of car for lateral control received!')
            return 0.0
        else:
            # Interpolate waypoints (already transformed to vehicle coordinates) to a polynomial of 3rd degree
            coeffs = np.polyfit(wp_carCoord_x[idxNearestWPfront:idxNearestWPfront+n], wp_carCoord_y[idxNearestWPfront:idxNearestWPfront+n], 3)
            p = np.poly1d(coeffs)
            # distance to track is polynomial at car's position x = 0
            CTE = p(0.0)
            return CTE
        
    def findNearestWPfront(self, wp_carCoord_x):
        # this function return first waypoint in front of car
        # it is asumed that wayponts are already ordered (by waypoint_updater)
        # if return value is negative then no point in front of car is found!
        
        # transfrom waypoint in vehcile coordnates
        idxNearestWPfront = -1
        for i in range(len(wp_carCoord_x)):
            if wp_carCoord_x[i] >= 0.0:
                idxNearestWPfront = i
                break
        
        return idxNearestWPfront
                
if __name__ == '__main__':
    DBWNode()
