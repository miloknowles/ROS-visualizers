#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plots the X, Y, Z, Roll, Pitch, and Yaw estimations vs. Truth along with their estimation error over time.
"""

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
	def getSyncedTime(self):
		return self.syncedTime
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

def removeDiscontinuitiesFromRotation(rotationArray):
	"""
	Takes in an array of angles, and fixes discontinuity jumps caused by angles wrapping around 360 degrees.
	"""
	offset = 0

	#note: ignore the 1st element in the array because it will consider index -1 (last item in the array)
	for i in range(1, len(rotationArray)):

		#if the angle suddently jumps by more than 300 degrees, assume that we're wrapping around
		#shift up or down by 360 degrees as needed
		if rotationArray[i] - rotationArray[i-1] > 250:
			offset = -360
		elif rotationArray[i] - rotationArray[i-1] < -250:
			offset = 360
		else:
			offset = 0
		rotationArray[i] += offset
	return rotationArray

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

	#now go through every synced pose and extract the right information
	counter = 0
	#good_ye = []
	for i in syncedPoses:
		times.append(i.getSyncedTime())

		xt, yt, zt = i.getTruthXYZ()
		xe, ye, ze = i.getEstXYZ()
		xyz_est = np.array([xe,ye,ze])
		#good_ye.append(ye)

		roll_t, pitch_t, yaw_t = i.getTruthEulerAngles()
		roll_e, pitch_e, yaw_e = i.getEstEulerAngles()

		# TRUTH EULER ANGLE #
		truth_roll.append(roll_t)
		truth_pitch.append(pitch_t)
		truth_yaw.append(yaw_t)

		#if this is the first synced pose, use it to calculate position offsets
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

			#get initial roll, pitch offsets
			roll_offset = roll_t - roll_e
			pitch_offset = pitch_t - pitch_e 


		#translate the Vicon truth so that it begins at (0,0,0) in the Rovio frame
		translated_xyz_truth = np.array([xt+x_truth_offset, yt+y_truth_offset, zt+z_truth_offset])

		# apply the transformation from the Imu Frame to the Vicon World Frame
		imu_in_vicon = transformRovioImuToVicon(xyz_est)

		# apply a small extra rotation to correct for callibration differences between imu and vicon 
		z_off = euler2mat(0,0,math.radians(yaw_offset))
		#easyrecord z_off: 13.11163 yaw
		imu_in_vicon_offset = np.dot(z_off, imu_in_vicon)

		# BUILD ARRAYS OF EST/TRUTH XYZ #
		est_x.append(imu_in_vicon_offset[0])
		est_y.append(imu_in_vicon_offset[1])
		est_z.append(imu_in_vicon_offset[2])
		truth_x.append(translated_xyz_truth[0])
		truth_y.append(translated_xyz_truth[1])
		truth_z.append(translated_xyz_truth[2])

		# EST EULER ANGLES #
		est_roll.append(roll_e+roll_offset)
		est_pitch.append(pitch_e+pitch_offset)
		est_yaw.append(yaw_e+yaw_offset+180)

	# remove wraparound discontinuities from roll/pitch/yaw
	if rmvRotationDisc:
		est_roll = removeDiscontinuitiesFromRotation(est_roll)
		est_pitch = removeDiscontinuitiesFromRotation(est_pitch)
		est_yaw = removeDiscontinuitiesFromRotation(est_yaw)
		truth_yaw = removeDiscontinuitiesFromRotation(truth_yaw)
		#truth_roll = removeDiscontinuitiesFromRotation(truth_roll)

	return (est_x, est_y, est_z, est_roll, est_pitch, est_yaw, truth_x, truth_y, truth_z, truth_roll, truth_pitch, truth_yaw, times)


def plotEstimationVSTruth(syncedPoses, x=0,y=0,z=0,roll=1,pitch=0,yaw=1):
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
		fig1 = plt.figure(1) #create the first plot
		fig1.suptitle("X Position vs. Truth",fontsize=14, fontweight='bold')
		
		ax1 = fig1.add_subplot(211)
		ax1.set_xlabel('time (sec)')
		ax1.set_ylabel('position (m)')
		ax1.plot(times, xe,'r',label='estimated')
		ax1.plot(times, xt, 'y', label='truth')
		ax1.legend(loc='best')

		ax2 = fig1.add_subplot(212)
		ax2.set_ylabel('error (m)')
		x_err = [xe[i]-xt[i] for i in range(len(xe))]
		ax2.plot(times,x_err,'m')

	if y:
		fig2 = plt.figure(2) #create the first plot
		fig2.suptitle("Y Position vs. Truth",fontsize=14, fontweight='bold')
		
		ax1 = fig2.add_subplot(211)
		ax1.set_xlabel('time (sec)')
		ax1.set_ylabel('position (m)')
		ax1.plot(times, ye,'r',label='estimated')
		ax1.plot(times, yt, 'y', label='truth')
		ax1.legend(loc='best')

		ax2 = fig2.add_subplot(212)
		ax2.set_ylabel('error (m)')
		y_err = [ye[i]-yt[i] for i in range(len(ye))]
		ax2.plot(times,y_err,'m')

	if z:
		fig3 = plt.figure(3) #create the first plot
		fig3.suptitle("Z Position vs. Truth",fontsize=14, fontweight='bold')
		
		ax1 = fig3.add_subplot(211)
		ax1.set_xlabel('time (sec)')
		ax1.set_ylabel('position (m)')
		ax1.plot(times, ze,'r',label='estimated')
		ax1.plot(times, zt, 'y', label='truth')
		ax1.legend(loc='best')

		ax2 = fig3.add_subplot(212)
		ax2.set_ylabel('error (m)')
		z_err = [ze[i]-zt[i] for i in range(len(ze))]
		ax2.plot(times,z_err,'m')

	if roll:
		fig4 = plt.figure(4) #create the first plot
		fig4.suptitle("Roll Angle vs. Truth",fontsize=14, fontweight='bold')
		
		ax1 = fig4.add_subplot(211)
		ax1.set_xlabel('time (sec)')
		ax1.set_ylabel('roll angle (deg)')
		ax1.plot(times, roll_e,'r',label='estimated')
		ax1.plot(times, roll_t, 'y', label='truth')
		ax1.legend(loc='best')

		ax2 = fig4.add_subplot(212)
		ax2.set_ylabel('error (deg)')
		roll_err = [roll_e[i]-roll_t[i] for i in range(len(roll_e))]
		ax2.plot(times,roll_err,'m')

	if pitch:
		fig5 = plt.figure(5) #create the first plot
		fig5.suptitle("Pitch Angle vs. Truth",fontsize=15, fontweight='bold')
		
		ax1 = fig5.add_subplot(211)
		ax1.set_xlabel('time (sec)')
		ax1.set_ylabel('pitch angle (deg)')
		ax1.plot(times, pitch_e,'r',label='estimated')
		ax1.plot(times, pitch_t, 'y', label='truth')
		ax1.legend(loc='best')

		ax2 = fig5.add_subplot(212)
		ax2.set_ylabel('error (deg)')
		pitch_err = [pitch_e[i]-pitch_t[i] for i in range(len(pitch_e))]
		ax2.plot(times,pitch_err,'m')

	if yaw:
		fig6 = plt.figure(6) #create the first plot
		fig6.suptitle("Yaw Angle vs. Truth",fontsize=16, fontweight='bold')
		
		ax1 = fig6.add_subplot(211)
		ax1.set_xlabel('time (sec)')
		ax1.set_ylabel('yaw angle (deg)')
		ax1.plot(times, yaw_e,'r',label='estimated')
		ax1.plot(times, yaw_t, 'y', label='truth')
		ax1.legend(loc='best')

		ax2 = fig6.add_subplot(212)
		ax2.set_ylabel('error (deg)')
		yaw_err = [yaw_e[i]-yaw_t[i] for i in range(len(yaw_e))]
		ax2.plot(times,yaw_err,'m')

	plt.show()



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

def main():
	#SETUP
	BAGFILE = MEDIUM_RECORD_FILE #the full path to the bagfile
	TRUTH_TF = VICON_FRAME_NAME #the name of the truth transform topic
	EST_TF = '/rovio/transform' # the name of the estimated transform (odometry for rovio) topic
	POSE_TOPIC = '/rovio/odometry'

	#get a chronological list of synced poses from the bag
	syncedPoses = buildSyncedPoseList(BAGFILE, 1e7, TRUTH_TF, EST_TF, POSE_TOPIC)
	print("Plotting", len(syncedPoses), "synchronized poses.")

	#display the right plots
	plotEstimationVSTruth(syncedPoses)



if __name__ == '__main__':
	main()