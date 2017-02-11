#!/usr/bin/env python  
import roslib
import rospy

import tf
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker


def broadcastImuInViconTransform(msg):
    br = tf.TransformBroadcaster()
    print "Sending transform"
    br.sendTransform((msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z),
                     (msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w),
                     msg.header.stamp,
                     "vicon",
                     "world")

def rotateImuToVicon():
    """
    Rotation to go from imu_world to vicon_world frame.
    """
    rot = np.array([ [-1, 0, 0], 
                     [0, -1, 0],
                     [0, 0, 1]])
    return np.dot(rot, xyz_vector)


def main():
    rovio_odometry_topic =  '/rovio/odometry'
    rospy.init_node('imu_in_vicon_broadcaster')
    rospy.Subscriber(vicon_tf_topic, TransformStamped, broadcastImuInViconTransform)
    rospy.spin()


if __name__ == '__main__':
    main()