#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rosbag
import roslib
import rospy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as p
from mpl_toolkits.mplot3d import Axes3D
import math
from transforms3d.quaternions import rotate_vector, quat2mat
from transforms3d.euler import quat2euler
import numpy as np

#message type for Vicon truth pose data, and the rovio estimated pose
from geometry_msgs.msg import TransformStamped


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
	def __init__(self, truthTFMsg, estTFMsg):
		self.truthTFMsg = truthTFMsg
		self.estTFMsg = estTFMsg
		self.truthTime = float(self.truthTFMsg.header.stamp.secs) + float(self.truthTFMsg.header.stamp.nsecs) / 1e9
		self.estTime = float(self.estTFMsg.header.stamp.secs) + float(self.estTFMsg.header.stamp.nsecs) / 1e9
		self.syncedTime = (self.truthTime + self.estTime) / 2

	def getTruthXYZ(self):
		return (self.truthTFMsg.transform.translation.x, self.truthTFMsg.transform.translation.y, self.truthTFMsg.transform.translation.z)

	def getEstXYZ(self):
		return (self.estTFMsg.transform.translation.x, self.estTFMsg.transform.translation.y, self.estTFMsg.transform.translation.z)

	def getTruthEulerAngles(self):
		return quat2euler((self.truthTFMsg.transform.rotation.w, self.truthTFMsg.transform.rotation.x, self.truthTFMsg.transform.rotation.y, self.truthTFMsg.transform.rotation.z))

	def getEstEulerAngles(self):
		return quat2euler((self.estTFMsg.transform.rotation.w, self.estTFMsg.transform.rotation.x, self.estTFMsg.transform.rotation.y, self.estTFMsg.transform.rotation.z))

	def getEstRotationMatrix(self):
		return quat2mat((self.estTFMsg.transform.rotation.w, self.estTFMsg.transform.rotation.x, self.estTFMsg.transform.rotation.y, self.estTFMsg.transform.rotation.z))


def buildSyncedPoseList(bagfile, epsilon, truth_topic, est_topic):
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
	for topic, msg, t in bag.read_messages(topics=[truth_topic, est_topic]):

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

def main():

	#SETUP
	BAGFILE = STAR1 #the full path to the bagfile
	TRUTH_TF = '/vicon/tf' #the name of the truth transform topic
	EST_TF = '/rovio/transform' # the name of the estimated transform (odometry for rovio) topic

	#get a chronological list of synced poses from the bag
	syncedPoses = buildSyncedPoseList(BAGFILE, 1e8, TRUTH_TF, EST_TF)
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

	counter = 0
	for i in syncedPoses:
		xt, yt, zt = i.getTruthXYZ()
		xe, ye, ze = i.getEstXYZ()

		roll_t, pitch_t, yaw_t = i.getTruthEulerAngles()
		roll_est, pitch_est, yaw_est = i.getEstEulerAngles()

		truth_x.append(xt)
		truth_y.append(yt)
		truth_z.append(zt)
		#signs between IMU and truth x,y are flipped!
		# xe*=-1
		# ye*=-1

		if counter==0:
			x_offset = xt-xe
			y_offset = yt-ye
			z_offset = zt-ze
			est_eulers = i.getEstEulerAngles()
			truth_eulers = i.getTruthEulerAngles()
			counter += 1
			callibrationRotMat = i.getEstRotationMatrix()
			print(callibrationRotMat)
			print("Truth Eulers:", truth_eulers)
			print("Est. Eulers:", est_eulers)

		translated_xyz = (xe+x_offset, ye+y_offset, ze+z_offset)
		# rotated_xyz = offsetRotation(callibrationRotMat, translated_xyz)
		# print "Rotated", rotated_xyz

		est_x.append(translated_xyz[0])
		est_y.append(translated_xyz[1])
		est_z.append(translated_xyz[2])

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