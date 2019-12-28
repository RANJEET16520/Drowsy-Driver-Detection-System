import os
import cv2
import dlib
import time
import pygame 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from imutils import face_utils

pygame.mixer.init()
pygame.mixer.music.load('audio/alarm.wav')
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

video_capture = cv2.VideoCapture(0)
time.sleep(2)

Image_data = []
Image_status = []

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])

	ear = (A+B) / (2*C)
	return ear

def mouth_aspect_ratio(mouth):
	A = distance.euclidean(mouth[13], mouth[19])
	B = distance.euclidean(mouth[14], mouth[18])
	C = distance.euclidean(mouth[15], mouth[17])
	D = distance.euclidean(mouth[12], mouth[16])
	
	mer = 2*(A+B+C)/D
	return mer

def Write_data(img, output, path_to_store, frame_count):
	path = '{}/{}_{}.jpg'.format(path_to_store, frame_count, str(output))
	cv2.imwrite(path, img)
	return


def Drowsy(video, thval, fcnt, path_to_store):
	EYE_ASPECT_RATIO_THRESHOLD = thval
	EYE_ASPECT_RATIO_CONSEC_FRAMES = fcnt
	frame_count = 0

	COUNTER = 0
	vidObj = cv2.VideoCapture(video) 
	while(True):
		ret, frame = vidObj.read()

		if ret == True:
			frame = cv2.flip(frame,1)
			Org_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame = cv2.resize(frame, (320, 240), interpolation = cv2.INTER_AREA) 
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			faces = detector(gray, 0)

			face_rectangle = face_cascade.detectMultiScale(frame, 1.3, 5)

			for (x,y,w,h) in face_rectangle:
				cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)

			for face in faces:
				# print(gray)
				Image_data.append(gray)
				shape = predictor(gray, face)
				shape = face_utils.shape_to_np(shape)

				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				mouth = shape[mStart:mEnd]

				# print("Left Eye :{}".format(leftEye))
				# print("Right Eye :{}".format(rightEye))
				# print("Mouth :{}".format(mouth))

				leftEyeAspectRatio = eye_aspect_ratio(leftEye)
				rightEyeAspectRatio = eye_aspect_ratio(rightEye)
				mouthAspectRatio = mouth_aspect_ratio(mouth)

				eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				mouthHull = cv2.convexHull(mouth)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

				if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD and (mouthAspectRatio < 0.8) or mouthAspectRatio > 2.0):
					Image_status.append([0])
					Write_data(gray, 0, path_to_store, frame_count)
					COUNTER += 1
					if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
						pygame.mixer.music.play(-1)
						cv2.putText(frame, "You are Drowsy", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
				else:
					Image_status.append([1])
					Write_data(gray, 1, path_to_store, frame_count)
					pygame.mixer.music.stop()
					COUNTER = 0
				# print(eyeAspectRatio, Image_status[-1], mouthAspectRatio)

				frame_count += 1
			cv2.imshow('Video', frame)
			cv2.imshow('Orginal', Org_frame)
			if(cv2.waitKey(30) & 0xFF == ord('q')):
				break
	
		else:
			pygame.mixer.music.stop()
			return


Data_class = {'glasses':[0.25,12], 'nightglasses':[0.24,12], 'night_noglasses':[0.22,12], 'noglasses':[0.22,12], 'sunglasses':[0.31,12]}
video_class = ['sleepyCombination.avi', 'slowBlinkWithNodding.avi', 'yawning.avi']

Image_path = 'Images'
main_path = 'Training_Evaluation_Dataset/Training Dataset'

def Folder_Movement(Image_path, main_path):
	data_path = "{}/{}".format(Image_path, main_path)

	for content in os.listdir(main_path):
		content_path = '{}/{}'.format(main_path, content)
		content_data = "{}/{}".format(Image_path, content)
	
		if content != '001':
			continue
		
		if not os.path.exists(content_data):
			os.mkdir(content_data)
			print("{} Created!".format(content_data))

		for folder in os.listdir(content_path):
			folder_path = '{}/{}'.format(content_path, folder)
			folder_data = "{}/{}".format(content_data, folder)
		
			if not os.path.exists(folder_data):
				os.mkdir(folder_data)
				print("{} Created!".format(folder_data))

			for file in os.listdir(folder_path):
				if ((file.endswith('.avi') or file.endswith('.mp4')) and file in video_class):
					fname = file.split('.')
					data_file = "{}/{}".format(folder_data, fname[0])		
					if not os.path.exists(data_file):
						os.mkdir(data_file)
						print("{} Created!".format(data_file))
					
					file_path = "{}/{}".format(folder_path, file)
					print(file_path)
					thval, fcnt = Data_class[folder]
					print(Data_class[folder])
					Drowsy(file_path, thval, fcnt, data_file)

	video_capture.release()
	cv2.destroyAllWindows()	
	return

Folder_Movement(Image_path, main_path)
