import cv2 # openCV library, referred to as cv2 in python code
import numpy as np # numpy for mathematical operations
from scipy import stats
import tkinter as tk # tk will be used to represent normal widgets
from tkinter import ttk # ttk is used for specific themed widgets
import tkinter.messagebox
from plyer import notification # Used to trigger notifications in user's computer
import time
from tkinter import *


cap = cv2.VideoCapture(0) # Choosing the default webcam (Cameras are ordered by default, 0 represents the first camera in line, and is usually the built-in camera of the laptop/desktop the user is using.)

root = tk.Tk() # The root window
# An initial message box that explains the whole procedure of this program/application
tk.messagebox.showinfo(title="Introductory information", 
                       message="""Hello. Welcome to Posturiser.
                       \nPosturiser is an AI that detects if you are sitting with correct posture or not. It uses AI and computer vision for this. It warns you if you sit with incorrect posture, so that you can fix your posture.
                       \nUpon fixing your posture, the warning will go away.
                       \nAfter clicking the 'Okay' button below, you will be asked to make a choice to confirm how you would like to be notified of your incorrect posture.
                       \nThe 'Notification Tray' option means sitting with incorrect posture will cause the AI to send you an actual notification at the bottom right of your screen. This notification will also show up in your notifications tab too. 
                       \nChoosing the 'Alert Box' option will mean you will get an alert message like the one you are reading right now to alert you of your incorrect posture. Clicking okay will make it disappear.
                       \nAfter making a choice, you will then have to sit with your correct, perfect posture for 5 seconds. In these 5 seconds, the AI will save and remember your correct sitting posture. 
                       \nDuring these 5 seconds, please look straight into the webcam, as this will guarantee the best results.
                       \nThe AI will start monitoring your posture, and, every second, it will use your current posture against what it recorded in those initial 5 seconds to determine if you're sitting with correct posture or not.
                       \nNot sitting with correct posture will cause you to be notified with your method of notification.""")

def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=2, search_width = 20): # Calculating contrast of a pixel in a picture, with contrast referring to brightness.
    y = int(y) # y and x are usually given as floats (32 bit decimal values), converted to integer as they will be used as index values for arrays/lists
    x = int(x)
    columns = img_gray[y:y+h, int(x-search_width/2):int(x+search_width/2)] # Selecting an area of pixels, around the pixel, that we are going to consider in our calculation for the contrast of this pixel
    column_average = columns.mean(axis=1) # Calculating the average of the intensity (brightness) of all pixels in this area
    try:
        gradient = np.gradient(column_average, 3) # Calculates the "gradient", essentially the average difference between the intensities of the pixels that we are considering
    except(Exception): return None
    gradient = np.absolute(gradient) # using the absolute value of the gradient
    max_indices = np.argpartition(gradient, -top_values_to_consider)[-top_values_to_consider:] # determining the indices (position) of the top 2 HIGHEST values (basically looking for the most intense changes in intensity)
    max_values = gradient[max_indices] # selecting the top 2 highest values (most intense change in intensity)
    if(max_values.sum() < top_values_to_consider): return None # return none if no large gradient exists - there are no significant changes in intensity in our considered area, so probably no shoulder in the range
    weighted_indices = (max_indices * max_values) # calculates the weighted indices by multiplying each index (max_indices) by its corresponding gradient value (max_values). assigns weight to each index based on the magnitude of the gradient at each index
    weighted_average_index = weighted_indices.sum() / max_values.sum() # average index within selected column area
    try:
        index = int(weighted_average_index) # if index is appropriate, it is returned, otherwise return none
    except(Exception): return None # if weighted_average_index is an invalid value and causes errors, then just return none
    index = y + index # making relative to the whole picture
    return index

def detect_shoulder(img_gray, face, direction, x_scale=0.75, y_scale=0.75):
    x_face, y_face, w_face, h_face = face # define face components, x,y position and width and height of face box
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
    lines = []
    for index in range(len(x_positions)):
        lines.append((x_positions[index], y_positions[index]))
    # extract line of best fit from lines
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions,y_positions) # slope and intercept are the only values we need, 
        # however functions also return other values too
        # calculates linear least-squares regression for 2 sets of data. allows us to get values for slope and y-intercept.
        line_y0 = int(x_positions[0] * slope + intercept)
        line_y1 = int(x_positions[-1] * slope + intercept) # determine the y-coordinates of the first and second endpoints of the line.
        line = [(x_positions[0], line_y0), (x_positions[-1], line_y1)]
        # line should be a list of 2 tuples, each being a 2 differet point in a line
        # coords in index 0 represent starting point of line, coords in index 1 represent ending point of line
    except(ValueError):
        return None
    # calculating the mean y-coordinate of the endpoints of the line to approximate the shoulder line's position within the image
    value = np.array([line[0][1], line[1][1]]).mean()
    # return rectangle and positions
    return line, lines, rectangle, value


# Check if the webcam is opened correctly
if not cap.isOpened(): # if there is a problem, inform the user
    tk.messagebox.showwarning(title="ERROR", message="Cannot open the webcam. Please check if your webcam is working.")
    exit() # Stop execution of the program

def close_window(): # A function that is called when the user has selected their option
    global selected_option # The selected option needs to be tracked, used as a global variable here. 
    selected_option = selected_option_var.get() # warning the user to be ready
    tk.messagebox.showwarning(parent=root, title = "Get ready", message = f"""We're going to measure your correct posture, sit straight and face your camera.
                              \nWe are going to alert you using the {selected_option}.""")
    root.after(0, lambda: root.destroy()) # to prevent a third blank window from appearing

options = ["Notification Tray", "Alert Box"]
selected_option_var = tk.StringVar() # creating a string variable for use in the gui
selected_option_var.set(options[0])  # Default selected option
dropdown_menu = ttk.Combobox(root, values=options, textvariable=selected_option_var)
dropdown_menu.pack(pady=10) # creating dropdown menu and setting position within its window
selected_option = selected_option_var.get() # fetch actual value
button_select = tk.Button(root, text="Select", command=close_window) # Binds select button to actual root window with options
button_select.pack(pady=10) # set position
root.geometry("200x100") # set default size of the window
root.title("Select your preferred method of notification!")

root.wait_window() # Make it wait for the user to select some input


shoulder_y_values = [] # array that contains multiple y-coordinate positions of where the users' shoulders be when using correct posture
i = 0
while i < 5: # only run 5 times and get an average from the 5 snapshots
    try:
        time.sleep(1)
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_AREA)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Our classifier file that contains data about how to 
        # scan faces (which is generated)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting inputted image to grayscale
        faces = face_cascade.detectMultiScale(frame, 1.05, 3) # using cascading classifier to detect face
        left_shoulder = detect_shoulder(grayscale, faces[0], "left") 
        right_shoulder = detect_shoulder(grayscale, faces[0], "right") # storing info about both shoulders
        if left_shoulder == None or right_shoulder == None: continue # retry if we couldnt record either shoulder
        left_shoulder_rectangle = left_shoulder[2] # rectangles/areas representing area taken up by each shoulder
        right_shoulder_rectangle = right_shoulder[2]
        left_shoulder_x, left_shoulder_y = left_shoulder[2][0], left_shoulder[2][1] # Left shoulder coordinates
        right_shoulder_x, right_shoulder_y = right_shoulder[2][0], right_shoulder[2][1] # Right shoulder coordinates
        average = (left_shoulder_y + right_shoulder_y)/2 # average y-position of both shoulders to get final average value for y-position of user's shoulders
        shoulder_y_values.append(average)
        i += 1
    except(Exception): # if we've run into an error, then we make it retry at getting a snapshot of the y-position of the user's shoulders.
        i-=1 # retrying infinitely until 5 recordings made
    
average_correct_y_height = 0 # Calculating mean/average of correct heights/y-positions of users shoulders.
for value in shoulder_y_values: average_correct_y_height += value
average_correct_y_height /= 5 
# warning user
tk.messagebox.showwarning(title="You will now be monitored", message="""We have recorded your correct posture. We will now start monitoring you. 
                        \nPlease proceed with your tasks.""")

cool_down = 0 # Amount of checks (done every second) before the user is sent a warning again. usually, the user needs some time to adjust back into correct posture.
while True:
    time.sleep(1) # Wait for a little time between each run to not clog up memory.
    ret, frame = cap.read() # Reading the actual picture/image currently seen by the selected webcam
    frame = cv2.resize(frame, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_AREA)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Our inbuilt xml document that contains data about how to scan faces (which is generated )
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale image
    faces = face_cascade.detectMultiScale(frame, 1.03, 4, minSize=(100, 100)) # detect faces
    
    try: # detecting shoulders and shoulder y-position just like above
        left_shoulder = detect_shoulder(grayscale, faces[0], "left") 
        right_shoulder = detect_shoulder(grayscale, faces[0], "right")
        if left_shoulder == None or right_shoulder == None: continue
        left_shoulder_rectangle = left_shoulder[2]
        right_shoulder_rectangle = right_shoulder[2]
        left_shoulder_x, left_shoulder_y = left_shoulder[2][0], left_shoulder[2][1] # Left shoulder coordinates
        right_shoulder_x, right_shoulder_y = right_shoulder[2][0], right_shoulder[2][1] # Right shoulder coordinates
        average_shoulder_height = (left_shoulder_y + right_shoulder_y)/2
        shoulder_slouch_diff = average_correct_y_height - average_shoulder_height # calculating the difference between current y-position of user's shoulders and the supposed correct y-position of user's shoulders 
    except(Exception): # retry if run into some failure
        continue



    # Determine slouching by checking the reference line and current shoulder line against a threshold
    slouch_threshold = 30  # Threshold value, represents MAXIMUM difference that correct required y-position of shoulders and the user's current y-position of shoulders can have for it to still be classified as "good posture"
    if abs(shoulder_slouch_diff) > slouch_threshold: # shoulder slouch diff is usually a negative value, so have to get absolute value for it to make sure program detects incorrect posture properly.
        if selected_option == "Alert Box": # if user wanted alert boxes, notify them as such
            if cool_down == 0: # if theyve run out of cool down time, its time to alert them
                tk.messagebox.showwarning(title = "ALERT", message = f"FIX YOUR POSTURE RIGHT NOW")
                cool_down = 5 # reset cool down as they can now have time to readjust
            else: cool_down -= 1 # if theyre already on a cool-down, decrease by 1
        elif selected_option == "Notification Tray": # if user wanted notifications, notify them as such
            if cool_down == 0:
                notification.notify(
                    title = 'Posturiser',
                    message = 'CORRECT YOUR POSTURE!',
                    app_icon = None,
                    timeout = 0.5,
                )
                cool_down = 5
            else: cool_down -= 1
    else: # if user has good posture, reset cool down
        cool_down = 5

mainloop() # triggering tkinter ui windows