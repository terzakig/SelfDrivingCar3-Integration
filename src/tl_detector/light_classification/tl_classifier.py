from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.lower_red_1 = np.array([0, 43, 46])
        self.upper_red_1 = np.array([10, 255, 255])
        self.lower_red_2 = np.array([156, 43, 46])
        self.upper_red_2 = np.array([180, 255, 255])
        self.lower_green = np.array([35, 43, 46])
        self.upper_green = np.array([77, 255, 255])

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask_red = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1) + \
            cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)

        red = cv2.countNonZero(mask_red)
        green = cv2.countNonZero(mask_green)
        
        if red > green:
            return TrafficLight.RED
        return TrafficLight.UNKNOWN
