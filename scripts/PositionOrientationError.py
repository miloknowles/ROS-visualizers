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
import numpy as np
from transforms3d.quaternions import rotate_vector, quat2mat
from transforms3d.euler import quat2euler

#message type for Vicon truth pose data, and the rovio estimated pose
from geometry_msgs.msg import TransformStamped

#paths to the bagfiles to analyze
EASY_RECORD_FILE = '/home/mknowles/bagfiles/euroc/easy_record.bag'
MEDIUM_RECORD_FILE = '/home/mknowles/bagfiles/euroc/medium_record.bag'
DIFFICULT_RECORD_FILE = '/home/mknowles/bagfiles/euroc/difficult_record.bag'

#these are the topics that contain estimated pose and truth pose
truth_tf = '/vicon/firefly_sbx/firefly_sbx'
est_tf = '/rovio/transform'
TOPICS = [truth_tf, est_tf]


def areSynchronized(msg1, msg2, epsilon):
	"""
	epsilon: a value (nanosecs) i.e 10^7 -> 10 ms
	Return True if two timestamps are within an epsilon of each other
	If two timestamps are within epsilon, they are close enough to be consider simultaneous.
	"""
	t1 = float(msg1.header.stamp.secs) + float(msg1.header.stamp.nsecs) / 1000000000
	t2 = float(msg2.header.stamp.secs) + float(msg2.header.stamp.nsecs) / 1000000000
	delta = abs(t2-t1)
	if delta < epsilon:
		return True
	else:
		return False


class syncedPose(object):
	def __init__(self, truthTFMsg, estTFMsg):
		self.truthTFMsg = truthTFMsg
		self.estTFMsg = estTFMsg
		self.truthTime = float(self.truthTFMsg.header.stamp.secs) + float(self.truthTFMsg.header.stamp.nsecs) / 1000000000
		self.estTime = float(self.estTFMsg.header.stamp.secs) + float(self.estTFMsg.header.stamp.nsecs) / 1000000000
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


def buildSyncedPoseList(bagfile, epsilon, truth_tf, est_tf):
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
	for topic, msg, t in bag.read_messages(topics=[truth_tf, est_tf]):
		
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
				if holdTopic == truth_tf:
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

def getPositionAndRotationArrays(syncedPoses):
	#times stores the timestamps in order for the synced poses
	times = []

	truth_x = []
	truth_y = []
	truth_z = []
	est_x = []
	est_y = []
	est_z = []

	# lists of roll, pitch, yaw for error plotting
	truth_roll = []
	truth_pitch = []
	truth_yaw = []
	est_roll = []
	est_pitch = []
	est_yaw = []

	#store the offsets between IMU frame zero and Vicon frame zero
	x_offset = 0
	y_offset = 0
	z_offset = 0
	#now go through every synced pose and extract the right information
	counter = 0
	for i in syncedPoses:
		times.append(i.syncedTime)

		xt, yt, zt = i.getTruthXYZ()
		xe, ye, ze = i.getEstXYZ()

		roll_t, pitch_t, yaw_t = i.getTruthEulerAngles()
		roll_est, pitch_est, yaw_est = i.getEstEulerAngles()

		#add the truth coordinates to their respective arrays
		truth_roll.append(roll_t)
		truth_pitch.append(pitch_t)
		truth_yaw.append(yaw_t)
		truth_x.append(xt)
		truth_y.append(yt)
		truth_z.append(zt)

		#signs between IMU and truth x,y are flipped!
		xe*=-1
		ye*=-1

		#if this is the first synced pose, use it to calculate position offsets
		if counter==0:
			x_offset = xt-xe
			y_offset = yt-ye
			z_offset = zt-ze
			counter += 1
			print("Initial Truth Eulers:", roll_t, pitch_t, yaw_t)
			print("Initial Est. Eulers:", roll_est, pitch_est, yaw_est)

		#add the estimated coordinates to their respective arrays
		est_x.append(xe+x_offset)
		est_y.append(ye+y_offset)
		est_z.append(ze+z_offset)
		est_roll.append(roll_est)
		est_pitch.append(pitch_est)
		est_yaw.append(yaw_est)

	return (est_x, est_y, est_z, est_roll, est_pitch, est_yaw, truth_x, truth_y, truth_z, truth_roll, truth_pitch, truth_yaw, times)


def plotEstimationVSTruth(syncedPoses, x=True,y=True,z=True,roll=True,pitch=True,yaw=True):
	"""
	Plots each coordinate vs. truth over time.
	Changing the parameter flags will determine which plots are shown.
	syncedPoses: an array of synced pose objects
	"""
	xe, ye, ze, re, pe, ye, xt, yt, zt, rt, pt, yt, times = getPositionAndRotationArrays(syncedPoses)
	print len(xe)

	if x:
		#Plot 1: X vs. truth
		plt.figure(1) #create the first plot
		plt.subplot(211)
		plt.plot(times, xe, 'ro')

	if y:
		#Plot 2: Y vs. truth
		plt.figure(2)
		plt.subplot(211)
		plt.plot(times, ye, 'bo')

	if z:
		#Plot 2: Y vs. truth
		plt.figure(3)
		plt.subplot(211)
		plt.plot(times, ze, 'go')

	if roll:
		pass

	if pitch:
		pass

	if yaw:
		pass

		plt.show()


def plotEstimationError(x_error=True,y_error=True,z_error=True,roll_error=True,pitch_error=True,yaw_error=True):
	pass

def main():
	#get a chronological list of synced poses from the bag
	syncedPoses = buildSyncedPoseList(DIFFICULT_RECORD_FILE, 100000000, truth_tf, est_tf)
	print("Created", len(syncedPoses), "synced poses")

	#display the right plots
	plotEstimationVSTruth(syncedPoses)

if __name__ == '__main__':
	main()