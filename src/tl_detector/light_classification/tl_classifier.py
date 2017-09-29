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
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

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

        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        #TODO: Yellow/Orange detection to be done

        red   = cv2.countNonZero(mask_red)
        green = cv2.countNonZero(mask_green)
        yellow = cv2.countNonZero(mask_yellow)
        
        if red > green and red > yellow:
            return TrafficLight.RED
        elif green > red and green > yellow:
            return TrafficLight.GREEN
        elif yellow > red and yellow > green:
            return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
