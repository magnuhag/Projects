#! /usr/bin/env python
from rospy
from geometry_msgs.msg import Twist        #publisher
from sensor_msgs.msg import LaserScan    #subscriber

class MoveRobot():
    def __init__(self):

        rospy.init_node('topics_quiz_node')

        #Publisher----------------------------------------------------------------------------
        self.publisher    = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.publish_rate = rospy.Rate(2)
        self.move         = Twist()
        #Subscriber---------------------------------------------------------------------------
        self.laser_sub    = rospy.Subscriber('/kobuki/laser/scan', LaserScan, self.callback)
        self.laser_scan   = LaserScan()

    def callback(self, msg):
        print(msg)


    def get_laserscan_to_cmd(self):
    #print (self.laser_scan.ranges[360])
        if  self.laser_scan.ranges[360] > 1:
                self.move.linear.x = 0.5
                self.move.angular.z = 0.0
        else:
            pass




if __name__ == "__main__":
    move_robot = MoveRobot()
    move_robot.get_laserscan_to_cmd()


    #while not rospy.is_shutdown():
     #       self.publisher.publish(self.move)
      #      self.publish_rate.sleep()
