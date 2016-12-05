#!/usr/bin/env python
# -*- coding: utf-8 -*-

#ros modules
import rosbag
import roslib
import rospy
import tf

#plotting modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as p
from mpl_toolkits.mplot3d import Axes3D

#math modules
import math
from transforms3d.quaternions import rotate_vector, quat2mat, qinverse, qmult
from transforms3d.euler import quat2euler
import numpy as np

#message type for Vicon truth pose data, and the rovio estimated pose
from geometry_msgs.msg import TransformStamped, PointStamped, QuaternionStamped, PoseStamped


def areSynchronized(msg1, msg2, epsilon):
	"""
	epsilon: a value (nanosecs) i.e 10^7 -> 10 ms
	Return True if two timestamps are within an epsilon of each other
	If two timestamps are within epsilon, they are close enough to be consider simultaneous.
	"""
	t1 = float(msg1.header.stamp.secs) + float(msg1.header.stamp.nsecs) / 1e9
	t2 = float(msg2.header.stamp.secs) + float(msg2.header.stamp.nsecs) / 1e9
	delta = abs(t2-t1)
	if delta < epsilon:
		return True
	else:
		return False


class syncedPose(object):
	def __init__(self, truthTFMsg, estPoseMsg):
		self.truthTFMsg = truthTFMsg
		self.estTFMsg = estPoseMsg #note: estTFMsg is of type nav_msgs/Odometry
		self.truthTime = float(self.truthTFMsg.header.stamp.secs) + float(self.truthTFMsg.header.stamp.nsecs) / 1e9
		self.estTime = float(self.estTFMsg.header.stamp.secs) + float(self.estTFMsg.header.stamp.nsecs) / 1e9
		self.syncedTime = (self.truthTime + self.estTime) / 2

	def pp(self):
		print("TruthXYZ:", self.getTruthXYZ())
		print("EstXYZ:", self.getEstXYZ())
		print("TruthQuat:", self.getTruthQuat())
		print("EstQuat:", self.getEstQuat())
		print("Truth Euler:", self.getTruthEulerAngles())
		print("Est Euler:", self.getEstEulerAngles())

	def getEstTransform(self):
		tfm = TransformStamped()
		tfm.header = self.estTFMsg.header
		tfm.child_frame_id = self.estTFMsg.child_frame_id
		tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z = self.getEstXYZ()
		tfm.transform.rotation.w, tfm.transform.rotation.x, tfm.transform.rotation.y, tfm.transform.rotation.z = self.getEstQuat()
		return tfm

	def getEstTF(self):
		return self.estTFMsg

	def getTruthTF(self):
		return self.truthTFMsg

	def transformPose(self, tf_ros, truth_frame_name, world_frame_name, tf_time):
		"""
		tf_ros: a TransformerROS object with the the necessary frames set up 
		This function transforms the quad's pose to the world frame, and then to the vicon frame
		"""
		pose = PoseStamped()
		pose.header = self.estTFMsg.header
		pose.header.stamp = tf_time
		pose.pose = self.estTFMsg.pose.pose
		pose = tf_ros.transformPose(world_frame_name, pose)
		pose = tf_ros.transformPose(truth_frame_name, pose)
		return pose # returns the pose part of of the msg

	def getTruthQuat(self):
		return (self.truthTFMsg.transform.rotation.w, self.truthTFMsg.transform.rotation.x, self.truthTFMsg.transform.rotation.y, self.truthTFMsg.transform.rotation.z)
	def getEstQuat(self):
		return (self.estTFMsg.pose.pose.orientation.w, self.estTFMsg.pose.pose.orientation.x, self.estTFMsg.pose.pose.orientation.y, self.estTFMsg.pose.pose.orientation.z)
	def getTruthXYZ(self):
		return (self.truthTFMsg.transform.translation.x, self.truthTFMsg.transform.translation.y, self.truthTFMsg.transform.translation.z)
	def getEstXYZ(self):
		return (self.estTFMsg.pose.pose.position.x, self.estTFMsg.pose.pose.position.y, self.estTFMsg.pose.pose.position.z)
	def getTruthEulerAngles(self):
		return [math.degrees(i) for i in quat2euler((self.truthTFMsg.transform.rotation.w, self.truthTFMsg.transform.rotation.x, self.truthTFMsg.transform.rotation.y, self.truthTFMsg.transform.rotation.z))]
	def getEstEulerAngles(self):
		return [math.degrees(i) for i in quat2euler((self.estTFMsg.pose.pose.orientation.w, self.estTFMsg.pose.pose.orientation.x, self.estTFMsg.pose.pose.orientation.y, self.estTFMsg.pose.pose.orientation.z))]

	# def getEstRotationMatrix(self):
	# 	return quat2mat((self.estTFMsg.transform.rotation.w, self.estTFMsg.transform.rotation.x, self.estTFMsg.transform.rotation.y, self.estTFMsg.transform.rotation.z))


def buildSyncedPoseList(bagfile, epsilon, truth_topic, est_topic, pose_topic):
	"""
	Bagfile: the full path to the .bag file that contains truth/est transforms
	epsilon: the value (nanosecs) that two msg stamps must be within to be considered simultaneous
	truth_tf: the topic name of the truth transform (i.e '/vicon/firefly_sbx/firefly_sbx')
	est_tf: the topic name of the estimated transform (i.e '/rovio/transform')
	"""
	bag = rosbag.Bag(bagfile)
	holdMsg = None
	holdTopic = None
	#this list will store all of the syncedPose objects that we can make from the bagfile
	syncedPoses = []

	#topic, message data object, time
	counter = 0
	for topic, msg, t in bag.read_messages(topics=[truth_topic, pose_topic]):

		#see if we're already holding a msg to find a match
		if holdMsg == None:
			holdMsg = msg
			holdTopic = topic
			continue #go to the next msg

		if topic != holdTopic: #this means we found a matching msg

			#check if the matching msg we just found is within epsilon nanosecs of our hold topic
			if areSynchronized(holdMsg, msg, epsilon):
				#check whether the holdTopic is truth or estimated
				# add the syncedPose object to a chronological list
				if holdTopic == truth_topic:
					syncedPoses.append(syncedPose(holdMsg, msg))
				else:
					syncedPoses.append(syncedPose(msg, holdMsg))

			# if the msgs were synchronized, then we want to stop holding our message and move on
			# or, if msgs were not synchronized, we want to throw away the held msg and find a new one
			holdMsg = None
			holdTopic = None

		else: #this means we just found another msg from the same topic we are holding
				#we should update our hold msg to be the latest msg
			holdMsg = msg

	return syncedPoses

def rotatePoint(originPoint, point, theta):
	"""
	Calculates the radius of the current point (distance between originPoint and point)
	theta: the angle between the point's coordinate system and the desired coordinate system
	originPoint: the (x0,y0,z0) point to rotate our (x,y,z) point around
	point: the (x,y,z) point that we want to put into our new coordinate system
	"""
	squareSum = 0

	#get radial distance in the xy plane
	for i in range(2):
		squareSum += (point[i] - originPoint[i]) ** 2
	xy_rad = math.sqrt(squareSum)
	new_x = xy_rad * math.sin(theta)
	new_y = xy_rad * math.cos(theta)

	return (new_x, new_y, point[2])


def offsetRotation(rot_mat, point):
	"""
	Applies the given rot_mat to an xyz point.
	"""
	pt = np.array(point)
	mat1 = np.mat(pt)
	mat2 = np.mat(rot_mat)
	new_pt = np.array(np.dot(rot_mat, np.transpose(mat1)))
	#print("Pt:", pt, "New pt:", new_pt)
	return (new_pt[0][0], new_pt[1][0], new_pt[2][0])


# BAGFILES TO ANALYZE #
# 1. euroc dataset (provided by ethz-asl)
EASY_RECORD_FILE = '/home/mknowles/bagfiles/euroc/easy_record.bag'
MEDIUM_RECORD_FILE = '/home/mknowles/bagfiles/euroc/medium_record.bag'
DIFFICULT_RECORD_FILE = '/home/mknowles/bagfiles/euroc/difficult_record.bag'
# 2. Kyel's STAR dataset
STAR0 = '/home/mknowles/bagfiles/star/star0_rovio.bag'
STAR1 = '/home/mknowles/bagfiles/star/star1_rovio.bag'
STAR2 = '/home/mknowles/bagfiles/star/star2_rovio.bag'
# END BAGFILE DEFS #

WORLD_FRAME_NAME = 'world'
VICON_FRAME_NAME = 'vicon/firefly_sbx/firefly_sbx'
QUAD_FRAME_NAME = 'imu'

def main():

	#make a TransformerROS object (handles transforms operations)
	t = tf.TransformerROS(True, rospy.Duration(10))

	#SETUP
	BAGFILE = STAR1 #the full path to the bagfile
	TRUTH_TF = '/vicon/tf' #the name of the truth transform topic
	EST_TF = '/rovio/transform' # the name of the estimated transform (odometry for rovio) topic
	POSE_TOPIC = '/rovio/odometry'

	#get a chronological list of synced poses from the bag
	syncedPoses = buildSyncedPoseList(BAGFILE, 1e7, TRUTH_TF, EST_TF, POSE_TOPIC)
	print("Plotting", len(syncedPoses), "synchronized poses.")

	#create an MPL figure
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# extract lists of x,y,z coords for plotting
	truth_x = []
	truth_y = []
	truth_z = []
	est_x = []
	est_y = []
	est_z = []

	#store the offsets between IMU frame zero and Vicon frame zero
	x_offset = 0
	y_offset = 0
	z_offset = 0
	roll_offset = 0
	pitch_offset = 0
	yaw_offset = 0

	counter = 0
	for i in syncedPoses:
		xt, yt, zt = i.getTruthXYZ()
		xe, ye, ze = i.getEstXYZ()

		# roll_t, pitch_t, yaw_t = i.getTruthEulerAngles()
		# roll_e, pitch_e, yaw_e = i.getEstEulerAngles()

		if counter<1:
			#calculate relative quaternion
			quat_rel = qmult(qinverse(np.array(i.getEstQuat())), np.array(i.getTruthQuat()))
			rel_rot_mat = quat2mat(quat_rel)
			x_est_offset = 0-xe
			y_est_offset = 0-ye
			z_est_offset = 0-ze
			x_truth_offset = 0-xt
			y_truth_offset = 0-yt
			z_truth_offset = 0-zt
			# roll_offset = roll_t - roll_e
			# pitch_offset = pitch_t - roll_t
			# yaw_offset = yaw_t - roll_t
			counter += 1

			i.pp()

			#determine rotation matrix to apply
			est_rot_mat = quat2mat(i.getEstQuat())
			truth_rot_mat = quat2mat(i.getTruthQuat())

		#first translate by some offset, then rotate into the truth coordinate frame
		translated_xyz_est = np.array([xe+x_est_offset, ye+y_est_offset, ze+z_est_offset])
		translated_xyz_truth = np.array([xt+x_truth_offset, yt+y_truth_offset, zt+z_truth_offset])
		rotated_xyz_truth = np.dot(truth_rot_mat, translated_xyz_truth)
		rel_rotated_xyz_est = rotate_vector(translated_xyz_est, quat_rel)

		est_x.append(-rel_rotated_xyz_est[0])
		est_y.append(rel_rotated_xyz_est[1])
		est_z.append(-rel_rotated_xyz_est[2])
		truth_x.append(translated_xyz_truth[0])
		truth_y.append(translated_xyz_truth[1])
		truth_z.append(translated_xyz_truth[2])

	#format the plot: truth is RED, estimated is GREEN
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('Estimated Pose vs. Truth (meters)')
	fig.add_axes(ax)
	ax.plot(truth_x, truth_y, zs=truth_z, zdir='z', color="r")
	ax.plot(est_x, est_y, zs=est_z, zdir='z', color="g")
	p.show()


if __name__ == '__main__':
	main()