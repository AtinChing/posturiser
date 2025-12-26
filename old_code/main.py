import cv2
import numpy as np
import math
from scipy import stats
import tkinter.messagebox
import time
from plyer import notification
def view_image(i):
    cv2.imshow('view', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_warning():
    tkinter.messagebox.showwarning("Warning", "Correct your posture right now!")

#i_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) # Removing colour information and going to grayscale as its easier for computer to make out an image through no colour rather than colour (RGB to be specific)
#print(i_gray.shape)
#print(i_gray[0, 0])
#view_image(i_gray)


#sobelx = cv2.Sobel(i_gray, cv2.CV_64F, 1, 0) # Same picture, just each pixel represents gradient of change of colour in the x direction
#abs_sobelx = np.absolute(sobelx)
##view_image(abs_sobelx/np.max(abs_sobelx))
#
#sobely = cv2.Sobel(i_gray, cv2.CV_64F, 0, 1) # Same picture, just each pixel represents gradient of change of colour in the y direction
#abs_sobely = np.absolute(sobely)
##view_image(abs_sobely/np.max(abs_sobely))
#
#magnitude = np.sqrt(sobelx**2 + sobely**2) # Same picture, just each pixel represents gradient of change of colour diagonally
##view_image(magnitude/np.max(magnitude))
#
#edges = cv2.Canny(i_gray, 200, 250) # Just a picture with some edges. 200 and 250 are the higher and lower bounds of accepted brightness for the lines.
##view_image(edges)

# Hough transform for lines
#lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi/180.0, threshold=20, minLineLength=10, maxLineGap=5)
#new_i = i.copy()
#for l in lines:
#    x1, y1, x2, y2 = l[0]
#    cv2.line(new_i, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
#view_image(new_i)

## Hough transform for circles
#circles = cv2.HoughCircles(i_gray, method=cv2.HOUGH_GRADIENT, dp=2, minDist=30, param1=150, param2=40, minRadius=15, maxRadius=25)
#newer_i = i_gray.copy()
#for x, y, r in circles[0]:
#    cv2.circle(newer_i, (int(x), int(y)), int(r), (0, 0, 255), thickness=1)
##view_image(newer_i)
#
## Blur image to reduce information
#i_blurred = cv2.GaussianBlur(i_gray, ksize = (21,21), sigmaX=0)
##view_image(i_blurred)
#circles2 = cv2.HoughCircles(i_blurred, method=cv2.HOUGH_GRADIENT, dp=2, minDist=30, param1=150, param2=40, minRadius=15, maxRadius=25)
#newer_i = i.copy()
#for x, y, r in circles2[0]:
#    cv2.circle(newer_i, (int(x), int(y)), int(r), (0, 0, 255), thickness=1)
#view_image(newer_i)

#print(i[1, 2, 0])
cap = cv2.VideoCapture(0)

 #copy pasted code
root = tkinter.Tk()
root.mainloop()
root.withdraw() # Tkinter displays a blank window whenever you create a root, so for user convenience, withdraw this window.
def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=2, search_width = 20):
    y = int(y)
    x = int(x)
    columns = img_gray[y:y+h, int(x-search_width/2):int(x+search_width/2)]
    column_average = columns.mean(axis=1)
    try:
        gradient = np.gradient(column_average, 3)
    except(Exception): return None
    gradient = np.absolute(gradient) # abs gradient value
    max_indicies = np.argpartition(gradient, -top_values_to_consider)[-top_values_to_consider:] # indicies of the top 5 values
    max_values = gradient[max_indicies]
    if(max_values.sum() < top_values_to_consider): return None # return none if no large gradient exists - probably no shoulder in the range
    weighted_indicies = (max_indicies * max_values)
    weighted_average_index = weighted_indicies.sum() / max_values.sum()
    try:
        index = int(weighted_average_index)
    except(Exception): return None # Using recursion to make a non-null value is returned
    index = y + index
    return index

def detect_shoulder(img_gray, face, direction, x_scale=0.75, y_scale=0.75):
    x_face, y_face, w_face, h_face = face # define face components
    # define shoulder box componenets
    w = int(x_scale * w_face)
    h = int(y_scale * h_face)
    y = y_face + h_face * 5/4 # half way down head position
    if(direction == "right"): x = x_face + w_face - w / 20 # right end of the face box
    if(direction == "left"): x = x_face - w + w/20 # w to the left of the start of face box
    rectangle = (x, y, w, h)
    # calculate position of shoulder in each x strip
    x_positions = []
    y_positions = []
    for delta_x in range(w):
        this_x = x + delta_x
        this_y = calculate_max_contrast_pixel(img_gray, this_x, y, h)
        if(this_y is None): continue # dont add if no clear best value
        x_positions.append(this_x)
        y_positions.append(this_y)
    # extract line from positions
    #line = [(x_positions[5], y_positions[5]), (x_positions[-10], y_positions[-10])]
    lines = []
    for index in range(len(x_positions)):
        lines.append((x_positions[index], y_positions[index]))
    # extract line of best fit from lines
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions,y_positions)
        line_y0 = int(x_positions[0] * slope + intercept)
        line_y1 = int(x_positions[-1] * slope + intercept)
        line = [(x_positions[0], line_y0), (x_positions[-1], line_y1)]
    except(ValueError):
        return None
    # decide on value
    #value = intercept
    value = np.array([line[0][1], line[1][1]]).mean()
    # return rectangle and positions
    return line, lines, rectangle, value
# end of copy pasted code


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

tkinter.messagebox.showwarning(title = "Get ready", message = "We're going to measure your correct posture, sit straight and face your camera.")
#time.sleep(2.5)
shoulder_y_values = []
for i in range(5):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Our inbuilt xml document that contains data about how to scan faces (which is generated )
    #upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_upperbody.xml") # Inbuilt one for upper body
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.05, 3)
    left_shoulder = detect_shoulder(grayscale, faces[0], "left")
    right_shoulder = detect_shoulder(grayscale, faces[0], "right")
    if left_shoulder == None or right_shoulder == None: continue
    left_shoulder_rectangle = left_shoulder[2]
    right_shoulder_rectangle = right_shoulder[2]
    left_shoulder_x, left_shoulder_y = left_shoulder[2][0], left_shoulder[2][1] # Left shoulder coordinates
    right_shoulder_x, right_shoulder_y = right_shoulder[2][0], right_shoulder[2][1] # Right shoulder coordinates
    shoulder_y_values.append((left_shoulder_y + right_shoulder_y)/2)
    

average_correct_y_height = 0 
for value in shoulder_y_values: average_correct_y_height += value
average_correct_y_height /= 5 # Calculating mean/average of heights.

while True:
    #time.sleep(5)
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_AREA)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Our inbuilt xml document that contains data about how to scan faces (which is generated )
    #upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_upperbody.xml") # Inbuilt one for upper body
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 10, 350)
    i_blurred = cv2.GaussianBlur(grayscale, ksize = (29,1), sigmaX=0)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi/180.0, threshold=1, minLineLength=1, maxLineGap=20)
    faces = face_cascade.detectMultiScale(frame, 1.03, 4, minSize=(100, 100))
    #upperbody = upperbody_cascade.detectMultiScale(frame, 1.05, 6)
    #for l in lines:
    #    x1, y1, x2, y2 = l[0]
    #    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
    #circles2 = cv2.HoughCircles(i_blurred, method=cv2.HOUGH_GRADIENT, dp=2, minDist=30, param1=150, param2=40, minRadius=15, maxRadius=25)
    #for x, y, r in circles2[0]:
    #    cv2.circle(frame, (int(x), int(y)), int(r), (0, 0, 255), thickness=1)

    try:
        left_shoulder = detect_shoulder(grayscale, faces[0], "left")
        right_shoulder = detect_shoulder(grayscale, faces[0], "right")
        if left_shoulder == None or right_shoulder == None: continue
        left_shoulder_rectangle = left_shoulder[2]
        right_shoulder_rectangle = right_shoulder[2]
        left_shoulder_x, left_shoulder_y = left_shoulder[2][0], left_shoulder[2][1] # Left shoulder coordinates
        right_shoulder_x, right_shoulder_y = right_shoulder[2][0], right_shoulder[2][1] # Right shoulder coordinates
        average_shoulder_height = (left_shoulder_y + right_shoulder_y)/2
        shoulder_slouch_diff = average_correct_y_height - average_shoulder_height
    except(Exception):
        continue

    # Calculate the angle between the shoulders' horizontal line and a reference line (e.g., vertical line)
    #reference_line_angle = math.atan2(right_shoulder_y - left_shoulder_y, right_shoulder_x - left_shoulder_x)

    # Convert the angle from radians to degrees
    #reference_line_angle_degrees = math.degrees(reference_line_angle)

    # Determine slouching by checking the reference line angle against a threshold
    slouch_threshold = 15  # Adjust the threshold as needed

    if abs(shoulder_slouch_diff) > slouch_threshold: # shoulder slouch diff is usually a negative value, so have to get absolute value for it to make sure program detects incorrect posture properly.
        print("Bad posture")
        root.after(1000, show_warning)
        #print(f'correct level is {average_correct_y_height} \n current level is {average_shoulder_height} \n difference is {shoulder_slouch_diff}')
    else:
        print("Good posture")
        #print(f'correct level is {average_correct_y_height} \n current level is {average_shoulder_height} \n difference is {shoulder_slouch_diff}')

    #print(reference_line_angle_degrees)
    #if abs(reference_line_angle_degrees) > slouch_threshold:
    #    print("Slouching shoulders detected!")
    #else:
    #    print("Good posture")
    #End of chat gpt code

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
    #for (x, y, w, h) in upperbody:
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
    #cv2.rectangle(frame, (int(left_shoulder_rectangle[0]), int(left_shoulder_rectangle[1])), (int(left_shoulder_rectangle[0]) + left_shoulder_rectangle[2], int(left_shoulder_rectangle[1]) + left_shoulder_rectangle[3]), (0, 255, 0), 5)
    #cv2.rectangle(frame, (int(right_shoulder_rectangle[0]), int(right_shoulder_rectangle[1])), (int(right_shoulder_rectangle[0]) + right_shoulder_rectangle[2], int(right_shoulder_rectangle[1]) + right_shoulder_rectangle[3]), (0, 255, 0), 5)
    
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 1:
        break
cap.release()
cv2.destroyAllWindows()