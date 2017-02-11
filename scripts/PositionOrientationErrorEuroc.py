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
from transforms3d.quaternions import rotate_vector, quat2mat, qinverse, qmult, mat2quat
from transforms3d.euler import quat2euler, euler2mat
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
VICON_FRAME_NAME = '/vicon/firefly_sbx/firefly_sbx'
QUAD_FRAME_NAME = 'imu'


def transformBLDToFLU(xyz_vector):
	"""
	Vicon and World frame are in Forward-Left-Up
	STAR Imu is in Back-Left-Down
	To transorm Rovio's Imu into the Vicon world frame, use rotation matrix
	rot = [ [-1, 0, 0]
			[0, 1, 0]
			[0, 0, -1] ]
	Note that this is equivalent to reversing the x-axis and z-axis
	"""
	rot = np.array([ [-1, 0, 0],
					 [0, 1, 0],
					 [0, 0, -1] ])
	return np.dot(rot, xyz_vector)

def transformURFToFLU(xyz_vector):
	rot = np.array([ [0, 0, 1],
					 [0, -1, 0],
					 [1, 0, 0] ])
	return np.dot(rot,xyz_vector)

def transformRovioImuToVicon(xyz_vector):
	"""
	Transforms the Rovio Imu world frame into the Vicon World frame
	This is a 180 deg rotation around z-axis
	"""
	rot = np.array([ [-1, 0, 0], 
					 [0, -1, 0],
					 [0, 0, 1]])
	return np.dot(rot, xyz_vector)

def transformRovioQuaternionToViconCF(rovio_quat):
	"""
	Rotates a rovio imu quaternion (in imu world frame)
	into the vicon world coordinate frame
	"""
	rot = np.array([ [0, 0, 1],
                     [0, 1, 0],
                     [-1, 0,0]])
	rel_quat = mat2quat(rot)
	new_quat = qmult(rovio_quat,rel_quat)
	#print "Eulers of imu in vicon world:", [math.degrees(i) for i in quat2euler(new_quat)]
	return new_quat

def getRelativeQuaternionAtoB(quat_a, quat_b):
	"""
	Given two quaternions A and B, determines the quaternion that rotates A to B
	(Multiplying A by q_rel will give B)
	"""
	q_rel = qmult(qinverse(quat_a), quat_b)
	return q_rel

def main():
	#SETUP
	BAGFILE = DIFFICULT_RECORD_FILE #the full path to the bagfile
	TRUTH_TF = VICON_FRAME_NAME #the name of the truth transform topic
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
		xyz_est = np.array([xe,ye,ze])


		if counter<1:
			# NOTE: Rovio starts at (x0,y0,z0) = (0,0,0)
			# determine how to translate the Vicon to (0,0,0) in the Rovio frame
			x_truth_offset = xe-xt
			y_truth_offset = ye-yt
			z_truth_offset = ze-zt
			counter += 1
			rovio_quat_in_vicon = transformRovioQuaternionToViconCF(i.getEstQuat())
			#print "Rovio eulers in vicon:", [math.degrees(j) for j in quat2euler(rovio_quat_in_vicon, axes='sxyz')]
			rovio_quat_offset_to_vicon = getRelativeQuaternionAtoB(rovio_quat_in_vicon, i.getTruthQuat())
			rovio_euler_offset_to_vicon = [math.degrees(j) for j in quat2euler(rovio_quat_offset_to_vicon)]
			yaw_offset = rovio_euler_offset_to_vicon[2]+2
			print "Imu euler offset to vicon:", rovio_euler_offset_to_vicon
			rovio_mat_offset_to_vicon = quat2mat(rovio_quat_offset_to_vicon)

		#translate the Vicon truth so that it begins at (0,0,0) in the Rovio frame
		translated_xyz_truth = np.array([xt+x_truth_offset, yt+y_truth_offset, zt+z_truth_offset])

		# apply the transformation from the Imu Frame to the Vicon World Frame
		imu_in_vicon = transformRovioImuToVicon(xyz_est)

		# apply a small extra rotation to correct for callibration differences between imu and vicon 
		z_off = euler2mat(0,0,math.radians(yaw_offset))
		#easyrecord z_off: 13.11163 yaw
		imu_in_vicon_offset = np.dot(z_off, imu_in_vicon)

		est_x.append(imu_in_vicon_offset[0])
		est_y.append(imu_in_vicon_offset[1])
		est_z.append(imu_in_vicon_offset[2])
		truth_x.append(translated_xyz_truth[0])
		truth_y.append(translated_xyz_truth[1])
		truth_z.append(translated_xyz_truth[2])


	#format the plot: truth is RED, estimated is GREEN
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_xlim([-3,1])
	ax.set_ylim([-3,1])
	ax.set_zlim([-2,2])
	ax.set_title('EUROC3: Estimated Pose vs. Truth (meters)')
	fig.add_axes(ax)
	# print "TX:", np.shape(truth_x)
	# print "TY:", np.shape(truth_y)
	# print "EX:", np.shape(est_x)
	# print "EY:",
	ax.plot(truth_x, truth_y, zs=truth_z, zdir='z', color="r")
	ax.plot(est_x, est_y, zs=est_z, zdir='z', color="g")
	p.show()


if __name__ == '__main__':
	main()