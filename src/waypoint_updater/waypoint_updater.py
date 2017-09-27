#!/usr/bin/env python

from KDTree import *
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import numpy as np

import math

import sys
sys.setrecursionlimit(10000) # 10000 is an example, try with different values

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 3 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
         

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.flag_waypoints_retrieved = False # flag for retrieving waypoints only once
        self.base_waypoints = None
        self.num_waypoints = -1 # just a shortcut to avoid using len() all the time

        self.pose_stamp = None
        self.car_x = None
        self.car_y = None
        #self.car_z = None # keep always 0
        self.car_theta = None

        # The track waypoints stored as:
        # [x, y, s, d, v_s, v_d, index]
        self.track = None
        self.track_root = None # the root of the kd-tree

        rospy.spin()

    # Find the next waypoint given a car pose [x, y, theta]   
    # NOTE: For now I am assuming that the car is going in the direction
    #       in which the way-point index increases. This probably tru, 
    #       but we really need to verify..... So, for now theta is not used
    def findNextWaypoint(self):
        # first find the closest node in the KD-tree:
        (dist, nearest_node) = self.track_root.NNSearch([self.car_x, self.car_y], INF_, self.track_root)
        # now this node maybe 'behind ' or 'ahead' of the car.
        # we need to clear this out
        nn_index = nearest_node.index
        next_wp_index = ( nn_index + 1 ) % self.num_waypoints
        # compute the direction (unit) vector from the nearest waypoint 
        # to the next:
        ux = self.base_waypoints[next_wp_index].pose.pose.position.x - self.base_waypoints[nn_index].pose.pose.position.x
        uy = self.base_waypoints[next_wp_index].pose.pose.position.y - self.base_waypoints[nn_index].pose.pose.position.y
        # now the norm of u:
        norm_u = np.sqrt( ux **2 + uy ** 2 )
        # normalizing
        ux /= norm_u
        uy /= norm_u
        
        # now get the difference vector from the nearest wp to ther car's position
        vx = self.car_x - self.base_waypoints[nn_index].pose.pose.position.x
        vy = self.car_y - self.base_waypoints[nn_index].pose.pose.position.y
        # Get the dot product with v
        dot = vx * ux + vy * uy
        
        if dot < 0:
            # Car is behind the nearest waypoint
            return nn_index
        else:
            # car is ahead of the nearest waypoint
            return next_wp_index
        
        


    def pose_cb(self, msg):
        # unwrapping the vehicle pose
        self.car_x = msg.pose.position.x
        self.car_y = msg.pose.position.y
        #car_z = msg.pose.position.z # not used but hey...
        # get the time stamp. might be useful to calculate latency
        self.pose_stamp = msg.header.stamp
       
        # get orientation
        s = msg.pose.orientation.w # quaternion scalar
        v1 = msg.pose.orientation.x # vector part 1
        v2 = msg.pose.orientation.y # vector part 2
        v3 = msg.pose.orientation.z # vector part 3        
        
        # now obtaining orientation of the car (assuming rotation about z: [0;0;1])
        self.car_theta = 2 * np.arccos(s)
        if self.car_theta > np.pi:
            self.car_theta = -(2 * np.pi - self.car_theta)
        # Now get the next waypoint....
        if self.flag_waypoints_retrieved:
            next_wp_index = self.findNextWaypoint()
            #rospy.logwarn("Next waypoint : %d", next_wp_index)
            #rospy.logwarn("car : (%f , %f) ", self.car_x, self.car_y)
            #rospy.logwarn("nwp : (%f , %f) ", self.base_waypoints[next_wp_index].pose.pose.position.x, self.base_waypoints[next_wp_index].pose.pose.position.y)
            
           # publish the nodes
            self.publishWaypoints(next_wp_index)
    
    # returns a string that represents the waypoints as a matlab matrix            
    def getWaypointMatrixAtr(self):
        str_ = " [ "
        for wp in self.base_waypoints:
            str_ += str(wp.pose.pose.position.x) + " , " + str(wp.pose.pose.position.y) + " ; "                 
        str_ = str_ + " ] "
        
        return str_
    
    # Fill a KD-tree that holds the track list
    def fillTrack(self):
        # cumulative distance in the s-axis of the Frenet frame        
        s = 0
        wp_index = 0
        for wp in self.base_waypoints:
            x = wp.pose.pose.position.x
            y = wp.pose.pose.position.y
            if wp_index > 0:
                s += np.sqrt( (x - x_prev)**2 + (y - y_prev) ** 2 )
                
            if self.track_root == None:
                self.track_root = KD2TreeNode(x,
                                              y,
                                              s,
                                              0,
                                              wp_index, 
                                              X_SPLIT) 
            else:
                self.track_root.insertNode(x, y, s, 0, wp_index)
            # increase the index
            wp_index += 1
            x_prev = x
            y_prev = y
        self.num_waypoints = wp_index
        rospy.logwarn("********Track 2D KD-tree created************")
        rospy.logwarn("%d waypoints inserted in the KD tree. ", self.num_waypoints)
    
    # This function publishes the next waypoints
    # For now it sets velocity to the same value...
    def publishWaypoints(self, next_wp_index):
        
        msg = Lane()
        #msg.header.stamp = rospy.Time
        msg.waypoints = []
        for i in range(LOOKAHEAD_WPS):
            # index of the trailing waypoints 
            index = i + next_wp_index            
            wp = Waypoint()
            wp.pose.pose.position.x = self.base_waypoints[index].pose.pose.position.x
            wp.pose.pose.position.y = self.base_waypoints[index].pose.pose.position.y
            # Velocity
            # TODO - TODO : Fill it with sensible velocities using
            #               feedback from the traffic light node etc...
            wp.twist.twist.linear.x = 20 # just a value...            
            # add the waypoint to the list
            msg.waypoints.append(wp)
        
        # publish the message
        self.final_waypoints_pub.publish(msg)
            
        
    
    def waypoints_cb(self, lanemsg):
        # unwrap the message
        if not self.flag_waypoints_retrieved:
            #header = lanemsg.header
            self.base_waypoints = lanemsg.waypoints
            # Now create the track            
            self.fillTrack()
            # raise the flag so that we don't have to do this again...
            self.flag_waypoints_retrieved = True            
            
            return
            
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
       # rospy.logwarn("Traffic waypoint int32 : %d", msg.data)
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
