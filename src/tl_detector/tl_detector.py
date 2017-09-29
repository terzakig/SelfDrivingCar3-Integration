#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
import random

STATE_COUNT_THRESHOLD = 3

class Point:
    def __init__(self, t):
        self.x = t[0]
        self.y = t[1]

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_index = 0
        closest_dis = -1

        if self.waypoints is not None:
            wps = self.waypoints.waypoints
            for i in range(len(wps)):
                dis = (wps[i].pose.pose.position.x - pose.x) ** 2 + \
                    (wps[i].pose.pose.position.y - pose.y) ** 2

                if (closest_dis == -1) or (closest_dis > dis):
                    closest_dis = dis
                    closest_index = i
        return closest_index


    # George: Here's my version of project_to_image_plane
    #     I am avoiding the TransformListener object as I am not sure about how
    #     to configure it without having doubts. The transform is easy to 
    #     work-out directly from the pose vector and gives me control over the 
    #     coordinate frame.
    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """
        # Retreving camera intronsics
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        # image size        
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # caching the world coordinates of the point
        Px = point_in_world.x
        Py = point_in_world.y
        Pz = point_in_world.z

        # Using the pose to obatin the camera frame as rotation matrix R and 
        # a world position p (NOTEL ASSUMING THAT THE CAMERA COINCIDES WITH 
        # THE CAR'S BARYCENTER (not really correct as it is somewhat of the ground!):
        Cx = self.pose.pose.position.x
        Cy = self.pose.pose.position.y
        Cz = self.pose.pose.position.z # not used but hey...
        
        # get orientation (just the scalar part of the the quaternion)
        s = self.pose.pose.orientation.w # quaternion scalar
        
        # now obtaining orientation of the car (assuming rotation about z: [0;0;1])
        theta = 2 * np.arccos(s)
        # Constraining the angle in [-pi, pi)
        if theta > np.pi:
            theta = -(2 * np.pi - self.car_theta)

        # transforming the world point to the camera frame as:
        #
        #               Mc = R' * (Mw - p)

        #        where R' = [ cos(theta)  sin(theta)   0; 
        #                   -sin(theta)  cos(theta)   0;
        #                      0              0      1]
        #
        # Thus,
        p_camera = [ np.cos(theta) * (Px - Cx) + np.sin(theta) * (Py - Cy) , \
                     -np.sin(theta) * (Px - Cx) + np.cos(theta) * (Py - Cy) , \
                     Pz - Cz]
                                                        


        # NOTE: From the simulator, it appears from the change in the angle 
        # that the positive direction of rotation is counter-clockwise. This 
        # means that there are two possible frame arrangements:
        #
        # a) A RIGHT-HAND frame: In this frame, z - points upwards (oposite to the 
        # image y axis)and y points to the left (oposite to the image x-axis)
        #
        # b) A LEFT_HAND frame: In this frame, z-points downwards (same as the
        #    image y-axis) and y points left (oposite to the image x-axis).
        #
        # thus, there are two ways of obtaining the image projection:
        
        
        x1 =  fx * ( -p_camera[1] ) / p_camera[0] + 0.5*image_width
        y1 =  fy * (  p_camera[2] ) / p_camera[0] + 0.5*image_height

        # or,

        x2 =  fx * ( -p_camera[1] ) / p_camera[0] + 0.5*image_width
        y2 =  fy * ( -p_camera[2] ) / p_camera[0] + 0.5*image_height

        # obviously, only one is correct, but needs to be veryfied with a 
        # known point and oits projection, which we dont have. But I would guess 
        # that x2, y2 are the correct ones. However, it never hurts to try both
        # jus to be sure...

        # choosing the second... 
        # TODO-TODO-TODO: TRY also (x1, y1) !!!! 

        return (x2, y2)


    def project_to_image_plane_old(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        rot = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        M = self.listener.fromTranslationRotation(trans, rot)
        p_world = np.array([[point_in_world.x], [point_in_world.y], [point_in_world.z], [1.0]])
        p_camera = np.dot(M, p_world)
        # print('=====')
        # print(p_camera)

        x = -fx * p_camera[1] / p_camera[0] + 0.5*image_width
        y = -fy * p_camera[2] / p_camera[0] + 0.5*image_height

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)
        # print(light)
        # print(x,y)
        #TODO use light location to zoom in on traffic light in image
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # ret = cv2.rectangle(cv_image, (x-20,y), (x+20,y+100), (0, 0, 255))
        # cv2.imwrite('/home/student/Desktop/New/' + str(random.random()) + '.jpg', ret)
        # print('saveimg')

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_wp = -1
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_light_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position)

            #TODO find the closest visible traffic light (if one exists)
            for i, stop_line in enumerate(stop_line_positions):
                stop_line_wp = self.get_closest_waypoint(Point(stop_line))
                if stop_line_wp >= car_position:
                    if (light_wp == -1) or (light_wp > stop_line_wp):
                        light_wp = stop_line_wp
                        light = self.lights[i]

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
