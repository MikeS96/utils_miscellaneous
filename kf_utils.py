# Miscelaneous of useful tracking functions

# coding: utf-8

import cv2
import numpy as np
from random import randint
import math
import vision_utils #Python file with util functions
from pykalman import KalmanFilter


#this functions allows the tracker initialization
def tracker_initializer(count_vehicles, u, v, classes, count_car, count_motbike, count_bike):
	#increase the vehicle counter
	count_vehicles += 1
	i_mean=np.asarray([u, 0, v, 0]) #State vector
	i_cov= 1 * np.eye(4) #covariance matrix
	#Method to select the class and color and increment the respective counter
	color, count_car, count_motbike, count_bike = vision_utils.selector(classes, count_car, count_motbike, count_bike)
	#Initiates a KalmanFilter for the object detected
	kf = KalmanFilter(transition_covariance=0.01 * np.eye(4), observation_covariance=0.005*np.eye(2), initial_state_mean=[u, 0, v, 0], transition_matrices = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], observation_matrices = [[1, 0, 0, 0], [0,0,1,0]])
	return count_vehicles, i_mean, i_cov, color, count_car, count_motbike, count_bike, kf

#This method fills the Asociation matrix with the minimum distance between tracks and detections
def distance_calculator(d, fdetects, detections_f, box, tracks, u, v, Am):
    fdetects.append(d) #Append to fdetects the d index
    detections_f[d]=box #append to the directionary the detection information in the d pos
	#For every track in the dictionary...
    for t in list(tracks):
		#Extract the centroid of the Tracker in the dictionary (ut = Cx and vt = Cy)
        ut = tracks[t][4][0] ##Extract Cx
        vt = tracks[t][4][2] ##Extract Cy
		#Compute the euclidian distance between the centroid of all the tracks with a detection (Fills a columns)
        dist = math.sqrt((ut-u)**2+(vt-v)**2)
        #Save the distance value in the correspond position of the matrix of tracks vs detections
        Am[t,d]=dist
    return Am, fdetects, detections_f

#Method to update a tracking with a specific detection
def tracker_updater(tracks, Am_b, fdetects, detections_f, Am_v):
	# Track update (This save the number of index, if there is 4 tracks it saves [0,2,3]
	track_ids=list(tracks)
	#Sweap the number of tracks in the tracks dictionary (one index per every data array in the dictionary)
	for ti in range(0,len(tracks)):
	    box={} #Create box dictionary to store the effective detections
	    auxv=0	#Aux counter to check if in three iterations there is not update, do not plot the track.
		#This for loop save in idx the index and val the information related to the Am_b matrix, in the row sweap every track and in fdetects the columns with detections...
	    for idx, val in enumerate(Am_b[ti,fdetects]):  #Moves through the row and select the first True that it founds 
			#If there is a true in the list index...                  
	        if val==True:
	        	dt=fdetects[idx] #Assign in dt the index value of fdetects in the idx pos
	        	# copy in box the information related to the detection that was made in the dt value and corresponds to a true
	        	box = detections_f[dt]
	        	#If the min distance is bigger than 3 pixels, means that there is no detection for the current tracker.
	        	if Am_v[ti] <= 5:
	        		auxv = 1
	        		x, y, w, h = box[0]
	        		u, v = box[1]
	        		# Update the measurements of the tracker
	        		tracks[track_ids[ti]][0]=(x, y, w, h)
	        		tracks[track_ids[ti]][1]=(u, v)
	        		#Update filter
	        		c_mean = tracks[track_ids[ti]][4]
	        		c_cov = tracks[track_ids[ti]][5]
	        		#update the mean and cov of the respective kalman filter... This receive the last states, the covariance matrix and the observations
	        		n_mean, n_cov = tracks[track_ids[ti]][3].filter_update(c_mean, c_cov, np.asarray([u,v]))
	        		#Update the tracks dictionary with the updated information
	        		tracks[track_ids[ti]][4]=n_mean
	        		tracks[track_ids[ti]][5]=n_cov
				
		#Flag to update the kf and say that its already updated
	    if auxv==0:
	        tracks[track_ids[ti]][6]+=1
	return tracks


#Method to do the prediction step in the KF
def predictor(tracks):
    for ti in list(tracks):
    	c_mean = tracks[ti][4]
    	c_cov = tracks[ti][5]
    	pred_mean, pred_cov = tracks[ti][3].filter_update(c_mean, c_cov, observation = None)
    	tracks[ti][4] = pred_mean
    	tracks[ti][5] = pred_cov
    return tracks

#Find the minimun of the Asociation matrix for each row
def minimun_index_finder(Am, detections):
	#Extract indexes of the min values for each row
	min_values = np.argmin(Am, axis = 1)
	#Create a zero vector with the size of Am
	Am_b = np.zeros((len(Am),len(detections)))*False
	#Assign a True in the positions with the minimun value
	for i in range(len(Am_b)):
		Am_b[i][min_values[i]] = True
	return Am_b
