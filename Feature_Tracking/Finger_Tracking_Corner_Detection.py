"""
Author: John Suchanek
Code Description: Index Finger Recognition and Tracking coordinates
of index finger tip within the boundaries by pink tape.
Date: September 2023
"""

#may need to pip install these packages
import cv2
import mediapipe as mp #for finger tracking
#import threading
import numpy as np

#initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

#initialize a VideoCapture object to access the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

#check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

#define the HSV range for pink color (you may need to adjust these values)
lower_pink = np.array([140, 100, 100])
upper_pink = np.array([170, 255, 255])

#minimum area to consider for a contour as a piece of pink tape (adjust as needed)
min_tape_area = 10

#initialize a flag to detect the corners
corners_detected = False

tt = 0

while True:
    tt += 1
    #capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    #convert the frame to RGB format for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    fingertip_x = None
    fingertip_y = None

    #check if any hand landmarks are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            #get the coordinates of the index finger tip (Landmark 8)
            index_finger_tip = landmarks.landmark[8]
            height, width, _ = frame.shape
            x = int(index_finger_tip.x * width)
            y = int(index_finger_tip.y * height)

            #update fingertip coordinates
            fingertip_x = x
            fingertip_y = y

            #draw a green point at the fingertip
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

    #convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if tt == 10:
        cv2.imwrite("hsv.jpg", hsv_frame)

    #create a mask to filter the pink color
    pink_mask = cv2.inRange(hsv_frame, lower_pink, upper_pink)
    if tt == 10:
        cv2.imwrite("pink_mask.jpg", pink_mask)

    #bitwise AND to get the pink color in the original frame
    pink_filtered_frame = cv2.bitwise_and(frame, frame, mask=pink_mask)
    if tt == 10:
        cv2.imwrite("pink_filtered_frame.jpg", pink_filtered_frame)

    #convert the frame to grayscale for contour detection
    gray_frame = cv2.cvtColor(pink_filtered_frame, cv2.COLOR_BGR2GRAY)
    if tt == 10:
        cv2.imwrite("gray_frame.jpg", gray_frame)

    #find contours in the grayscale frame
    contours, _ = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #initialize a list to store detected tape positions
    detected_tape_positions = []

    for contour in contours:
        #filter by contour area
        if cv2.contourArea(contour) > min_tape_area:
            #get the centroid of the contour as a tape position
            M = cv2.moments(contour)
            if M["m00"] != 0:  #check if the zeroth order moment (m00) is not zero (ensuring the contour has an area)
                x = int(M["m10"] / M["m00"])  #calculate the x-coordinate of the centroid using the first order moments
                y = int(M["m01"] / M["m00"])  #calculate the y-coordinate of the centroid using the first order moments
                detected_tape_positions.append((x, y))  # Append the centroid coordinates as a tuple to the list

    #sort the detected tape positions by the sum of x and y coordinates
    detected_tape_positions.sort(key=lambda point: (point[0] + point[1]))


    #if four distinct tape positions are found, update the filtered matrix
    if len(detected_tape_positions) == 4:
        corners_detected = True
        #print(detected_tape_positions)

    #draw circles at the positions of the detected pink tape
    for x, y in detected_tape_positions:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    #draw a rectangle connecting the corners
    if corners_detected:
        cv2.polylines(frame, [np.array(detected_tape_positions)], isClosed=True, color=(0, 255, 0), thickness=2)

        #draw a green point at the tracked fingertip coordinates when it's within the specified rectangle
        #print(fingertip_x, fingertip_y)
        x_offset = detected_tape_positions[0][0]
        y_offset = detected_tape_positions[0][1]
        adjusted_tape_positions = [(x - x_offset, y - y_offset) for x, y in detected_tape_positions]
        print(adjusted_tape_positions)

        if (
            fingertip_x is not None
            and fingertip_y is not None
            ):
            fingertip_x -= detected_tape_positions[0][0] 
            fingertip_y -= detected_tape_positions[0][1]

            if (
                adjusted_tape_positions[0][0] <= fingertip_x <= adjusted_tape_positions[3][0]
                and adjusted_tape_positions[0][1] <= fingertip_y <= adjusted_tape_positions[3][1]
            ):
                print(adjusted_tape_positions)
                cv2.circle(frame, (fingertip_x, fingertip_y), 8, (0, 255, 0), -1)
                fingertip_coordinates = f"Fingertip coordinates: ({fingertip_x}, {fingertip_y})"
                cv2.putText(frame, fingertip_coordinates, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(fingertip_coordinates)
            else:
                print("finger out of bounds")

    #display the video frame
    cv2.imshow('Video with Tracking', frame)

    corners_detected = False

    #break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
