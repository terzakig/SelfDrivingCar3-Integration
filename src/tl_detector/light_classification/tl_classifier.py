from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
#        self.lower_red_1 = np.array([0, 43, 46])
#        self.upper_red_1 = np.array([10, 255, 255])
#        self.lower_red_2 = np.array([156, 43, 46])
#        self.upper_red_2 = np.array([180, 255, 255])
        self.lower_red_1 = np.array([0, 100, 100])
        self.upper_red_1 = np.array([10, 255, 255])
        
        self.lower_red_2 = np.array([160, 100, 100])
        self.upper_red_2 = np.array([179, 255, 255])
        
        self.lower_green = np.array([35, 43, 46])
        self.upper_green = np.array([77, 255, 255])
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        
        h = 30
        w = 30
        self.putative_red = np.zeros((h, w,  3), dtype = np.uint8)
        # now fill in the 1s in a dist of radius template_dim / 2 and
        # centrer (template_dim / 2, template_dim / 2)
        radius_x =  w / 2 - 3 # leave some margin for black pixels
        radius_y = h /2 - 3         
        cx = w / 2
        cy = h / 2
        N = 30
         
        for deg in range(360):
            for i in range(N):
                r_x = (1.0 * radius_x) / N * (i + 1)
                r_y = (1.0 * radius_y) / N * (i + 1)
                                
                x = int(r_x * np.cos(deg * np.pi / 180) + cx )
                y = int(r_y * np.sin(deg * np.pi / 180) + cy )
                # NOTE: Create a BGR to acommodate OpenCV conventions....                
                self.putative_red[y][x][0] = 0
                self.putative_red[y][x][1] = 0
                self.putative_red[y][x][2] = 255
        
        #self.tl_putative = cv2.imread("/home/george/SelfDrivingCar-Final/CarND-Capstone/ros/src/tl_detector/light_classification/red_tl_light.png")
        #print("Shape of putative image : " , self.tl_putative.shape)
    

    def matchRedTemplate(self, image, template_x, template_y):
        
        w = template_x
        h = template_y
        # first create a template of size template_dim
#        template = np.zeros((template_y, template_x,  3), dtype = np.uint8)
#        # now fill in the 1s in a dist of radius template_dim / 2 and
#        # centrer (template_dim / 2, template_dim / 2)
#        radius_x =  template_x / 2 - 2 # leave some margin for black pixels
#        radius_y = template_y /2 - 2         
#        cx = template_x / 2
#        cy = template_y / 2
#        N = 100
#         
#        for deg in range(360):
#            for i in range(N):
#                r_x = (1.0 * radius_x) / N * (i + 1)
#                r_y = (1.0 * radius_y) / N * (i + 1)
#                                
#                x = int(r_x * np.cos(deg * np.pi / 180) + cx )
#                y = int(r_y * np.sin(deg * np.pi / 180) + cy )
#                # NOTE: Create a BGR to acommodate OpenCV conventions....                
#                template[y][x][0] = 0
#                template[y][x][1] = 0
#                template[y][x][2] = 255
        
        template = cv2.resize(self.putative_red, (template_x, template_y))
        # Done making the template!
        
        # Convert images to hsv
        #template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Now doing plain old template matching in hsv space!               
        # All the 6 distances
        #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        #           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        
        method = 'cv2.TM_CCORR_NORMED'
        display_img = image.copy()
        method = eval(method)

        # Apply template Matching
        res = cv2.matchTemplate(image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if (max_val > 0.7):
            state = TrafficLight.RED
            color = (0, 0, 255)
        else:
            state = TrafficLight.UNKNOWN
            color = (255, 0, 0)

        #print("Maximum template score : ", 1.0 * max_val  )

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
    
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(display_img, top_left, bottom_right, color, 2)

        

        return (display_img, state)


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        # debug
        #rospy.logwarn("Size of image : (%i, %i, %i)", image.shape[0], image.shape[1], image.shape[2])
                
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # checking for Non image. We shouldn't be here though.....
        if (hsv is None):
            return TrafficLight.UNKNOWN
        # debug
        #rospy.logwarn("Size of image : (%i, %i, %i)", hsv.shape[0], hsv.shape[1], hsv.shape[2])
        
        mask_red = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1) + \
            cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)

        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        #TODO: Yellow/Orange detection to be done

        red   = cv2.countNonZero(mask_red)        
        #print("Non-zero red : ", red)
        green = cv2.countNonZero(mask_green)
        yellow = cv2.countNonZero(mask_yellow)
        
        
#        if red > green and red > yellow:
#            return TrafficLight.RED
#        elif green > red and green > yellow:
#            return TrafficLight.GREEN
#        elif yellow > red and yellow > green:
#            return TrafficLight.YELLOW
        #print("red pixels ", red)
        if (red == 0):
            return TrafficLight.UNKNOWN
        if (red >= 200):
            return TrafficLight.RED
        if red < 200:        
            if ( yellow / (1.0 * red) < 1 ):
                #print("RED")                
                return TrafficLight.RED
            elif ( yellow / (1.0 * red) > 1 ):
                #print("YELLOW")                
                return TrafficLight.YELLOW
                    
