#!/usr/bin/env python  
import roslib
import rospy

import tf
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker


def broadcastCameraTransform(msg):
    br = tf.TransformBroadcaster()
    print "Sending transform"
    br.sendTransform((msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z),
                     (msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w),
                     msg.header.stamp,
                     "cam0",
                     "world")

def main():
    rospy.init_node('vicon_tf_broadcaster')
    rospy.Subscriber('/vicon/tf', TransformStamped, broadcastViconTransform)
    rospy.spin()


if __name__ == '__main__':
    main()