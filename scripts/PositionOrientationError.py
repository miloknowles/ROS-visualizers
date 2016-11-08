#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plots the X, Y, Z, Roll, Pitch, and Yaw estimations vs. Truth along with their estimation error over time.
"""



import rosbag
import roslib
import rospy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as p
import math
import numpy as np
from transforms3d.quaternions import rotate_vector, quat2mat
from transforms3d.euler import quat2euler

#message type for Vicon truth pose data, and the rovio estimated pose
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

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
		roll,pitch,yaw = quat2euler((self.truthTFMsg.transform.rotation.w, self.truthTFMsg.transform.rotation.x, self.truthTFMsg.transform.rotation.y, self.truthTFMsg.transform.rotation.z))
		return (math.degrees(roll),math.degrees(pitch),math.degrees(yaw))
	def getEstEulerAngles(self):
		roll, pitch, yaw = quat2euler((self.estTFMsg.transform.rotation.w, self.estTFMsg.transform.rotation.x, self.estTFMsg.transform.rotation.y, self.estTFMsg.transform.rotation.z))
		return (math.degrees(roll),math.degrees(pitch),math.degrees(yaw))

	def getSyncedTime(self):
		return self.syncedTime

	def getEstRotationMatrix(self):
		return quat2mat((self.estTFMsg.transform.rotation.w, self.estTFMsg.transform.rotation.x, self.estTFMsg.transform.rotation.y, self.estTFMsg.transform.rotation.z))

def removeDiscontinuitiesFromRotation(rotationArray):
	"""
	Takes in an array of angles, and fixes discontinuity jumps caused by angles wrapping around 360 degrees.
	"""
	offset = 0

	#note: ignore the 1st element in the array because it will consider index -1 (last item in the array)
	for i in range(1, len(rotationArray)):

		#if the angle suddently jumps by more than 300 degrees, assume that we're wrapping around
		#shift up or down by 360 degrees as needed
		if rotationArray[i] - rotationArray[i-1] > 300:
			offset = -360
		elif rotationArray[i] - rotationArray[i-1] < -300:
			offset = 360
		else:
			offset = 0
		rotationArray[i] += offset
	return rotationArray

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

def getPositionAndRotationArrays(syncedPoses, xReverse=False, yReverse=False, zReverse=False, rollReverse=0, pitchReverse=0, yawReverse=0, rmvRotationDisc=True):
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
	good_ye = []
	for i in syncedPoses:
		times.append(i.getSyncedTime())

		xt, yt, zt = i.getTruthXYZ()
		xe, ye, ze = i.getEstXYZ()
		good_ye.append(ye)

		roll_t, pitch_t, yaw_t = i.getTruthEulerAngles()
		roll_e, pitch_e, yaw_e = i.getEstEulerAngles()

		#add the truth coordinates to their respective arrays
		truth_roll.append(roll_t)
		truth_pitch.append(pitch_t)
		truth_yaw.append(yaw_t)
		truth_x.append(xt)
		truth_y.append(yt)
		truth_z.append(zt)


		#if this is the first synced pose, use it to calculate position offsets
		if counter==0:
			x_offset = xt-xe
			y_offset = yt-ye
			z_offset = zt-ze
			roll_offset = roll_t - roll_e
			pitch_offset = pitch_t - pitch_e
			print("Pitch offset:", pitch_offset)
			yaw_offset = yaw_t - yaw_e
			counter += 1
			print("Offsets:", x_offset, y_offset, z_offset, roll_offset, pitch_offset, yaw_offset)
			print("Initial Truth Eulers:", roll_t, pitch_t, yaw_t)
			print("Initial Est. Eulers:", roll_e, pitch_e, yaw_e)

		#add the estimated coordinates to their respective arrays
		if xReverse:
			xe*=-1
		if yReverse:
			ye*=-1
		if zReverse:
			ze*=-1
		if rollReverse:
			roll_e*=-1
		if pitchReverse:
			pitch_e*=-1
		if yawReverse:
			yaw_e*=-1

		est_x.append(xe+x_offset)
		est_y.append(ye+y_offset) #need to multiply x and y by -1 because TF conventions are different for IMU and Vicon
		est_z.append(ze+z_offset)
		est_roll.append(roll_e+roll_offset)
		est_pitch.append(pitch_e+pitch_offset)
		est_yaw.append(yaw_e+yaw_offset)

	#est_y data is good leaving this function!!!!

	if rmvRotationDisc:
		est_roll = removeDiscontinuitiesFromRotation(est_roll)
		est_pitch = removeDiscontinuitiesFromRotation(est_pitch)
		est_yaw = removeDiscontinuitiesFromRotation(est_yaw)

	return (est_x, est_y, est_z, est_roll, est_pitch, est_yaw, truth_x, truth_y, truth_z, truth_roll, truth_pitch, truth_yaw, times)


def plotEstimationVSTruth(syncedPoses, x=0,y=0,z=0,roll=True,pitch=True,yaw=True):
	"""
	Plots each coordinate vs. truth over time.
	Changing the parameter flags will determine which plots are shown.
	syncedPoses: an array of synced pose objects
	"""
	xe, ye, ze, roll_e, pitch_e, yaw_e, xt, yt, zt, roll_t, pitch_t, yaw_t, times = getPositionAndRotationArrays(syncedPoses,xReverse=True,yReverse=True)
	
	#make sure the data is 1-to-1 by running a few tests
	# assert(len(ye)==len(times),"Mismatch in number of time values and position estimate values")
	# assert(len(yt)==len(times),"Mismatch in number of time values and position truth values")
	# assert(len(yaw_e)==len(times), "Mismatch in number of time values and rotation estimate values")
	# assert(len(pitch_t)==len(times), "Mismatch in number of time values and rotation truth values")


	if x:
		#Plot 1: X vs. truth
		plt.figure(1) #create the first plot
		plt.title("X vs. Truth")
		plt.subplot(211)
		plt.plot(times, xe,'r', times, xt, 'y')

		plt.subplot(212)
		x_err = [xe[i]-xt[i] for i in range(len(xe))]
		plt.plot(times,x_err,'m')

	if y:
		#Plot 2: Y vs. truth
		#print(ye)
		plt.figure(2)
		plt.title("Y vs. Truth")
		plt.subplot(211)
		plt.plot(times, ye,'g', times, yt,'y')

		plt.subplot(212)
		y_err = [ye[i]-yt[i] for i in range(len(ye))]
		plt.plot(times,y_err,'m')

	if z:
		#Plot 2: Y vs. truth
		plt.figure(3)
		plt.title("Z vs. Truth")
		plt.subplot(211)
		plt.plot(times, ze,'b', times, zt,'y')

		plt.subplot(212)
		z_err = [ze[i]-zt[i] for i in range(len(ze))]
		plt.plot(times,z_err,'m')

	if roll:
		plt.figure(4)
		plt.title("Roll vs. Truth")
		plt.subplot(211)
		plt.plot(times, roll_e, 'r', times, roll_t, 'y')

		plt.subplot(212)
		roll_err = [roll_e[i]-roll_t[i] for i in range(len(roll_e))]
		plt.plot(times,roll_err,'m')

	if pitch:
		plt.figure(5)
		plt.title("Pitch vs. Truth")
		plt.subplot(211)
		plt.plot(times, pitch_e, 'g', times, pitch_t, 'y')

		plt.subplot(212)
		pitch_err = [pitch_e[i]-pitch_t[i] for i in range(len(pitch_e))]
		plt.plot(times,pitch_err,'m')

	if yaw:
		plt.figure(6)
		plt.title("Yaw vs. Truth")
		plt.subplot(211)
		plt.plot(times, yaw_e, 'b', times, yaw_t, 'y')

		plt.subplot(212)
		yaw_err = [yaw_e[i]-yaw_t[i] for i in range(len(yaw_e))]
		plt.plot(times,yaw_err,'m')

	plt.show()






def main():
	#get a chronological list of synced poses from the bag
	syncedPoses = buildSyncedPoseList(EASY_RECORD_FILE, 10000000, truth_tf, est_tf)
	print("Created", len(syncedPoses), "synced poses")

	#display the right plots
	plotEstimationVSTruth(syncedPoses)

if __name__ == '__main__':
	main()