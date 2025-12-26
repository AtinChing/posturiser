import cv2
import numpy as np
from scipy import stats
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
from plyer import notification
import time
from tkinter import *

def view_image(i):
    cv2.imshow('view', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_warning():
    tk.messagebox.showwarning(title="Warning", message= "CORRECT YOUR POSTURE RIGHT NOW!")
cap = cv2.VideoCapture(0)

root = tk.Tk()
tk.messagebox.showinfo(title="Introductory information", message="Hello. Welcome to Posturiser.\nPosturiser is an AI that detects if you are sitting with correct posture or not. It uses AI and computer vision for this. It warns you if you sit with incorrect posture, so that you can fix your posture.\nUpon fixing your posture, the warning will go away.\nAfter clicking the 'Okay' button below, you will be asked to make a choice to confirm how you would like to be notified of your incorrect posture.\nThe 'Notification Tray' option means sitting with incorrect posture will cause the AI to send you an actual notification at the bottom right of your screen. This notification will also show up in your notifications tab too. \nChoosing the 'Alert Box' option will mean you will get an alert message like the one you are reading right now to alert you of your incorrect posture. Clicking okay will make it disappear.\nAfter making a choice, you will then have to sit with your correct, perfect posture for 5 seconds. In these 5 seconds, the AI will save and remember your correct sitting posture. During these 5 seconds, please look straight into the webcam, as this will guarantee the best results.\nThe AI will start monitoring your posture, and, every second, it will use your current posture against what it recorded in those initial 5 seconds to determine if you're sitting with correct posture or not.\nNot sitting with correct posture will cause you to be notified with your method of notification.")
def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=2, search_width = 20): # Calculating difference between 2 pixels in inputted picture.
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
    except(Exception): return None # Using e-handling to ensure that a non-null value is returned
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
    value = np.array([line[0][1], line[1][1]]).mean()
    # return rectangle and positions
    return line, lines, rectangle, value


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

def close_window():
    global selected_option
    selected_option = selected_option_var.get()
    tk.messagebox.showwarning(parent=root, title = "Get ready", message = f"We're going to measure your correct posture, sit straight and face your camera.\n We are going to alert you using the {selected_option}.")
    root.after(0, lambda: root.destroy())

options = ["Notification Tray", "Alert Box"]
selected_option_var = tk.StringVar()
selected_option_var.set(options[0])  # Default selected option
dropdown_menu = ttk.Combobox(root, values=options, textvariable=selected_option_var)
dropdown_menu.pack(pady=10)
selected_option = selected_option_var.get()
button_select = tk.Button(root, text="Select", command=close_window) # Binds select button to actual root window with options
button_select.pack(pady=10)
root.geometry("200x100")
root.title("Select your preferred method of notification!")

root.wait_window() # Make it wait for the user to select some input


shoulder_y_values = []
i = 0
while i < 5: # original part where we measure
    try:
        time.sleep(1)
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_AREA)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Our inbuilt xml document that contains data about how to 
        #scan faces (which is generated )
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
        average = (left_shoulder_y + right_shoulder_y)/2
        shoulder_y_values.append(average)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cv2.line(frame, (0, int(average)), (frame.shape[1], int(average)), (0, 0, 255), 2)
        cv2.imshow('Input', frame)
        i += 1
    except(Exception):
        i-=1
    

average_correct_y_height = 0 
for value in shoulder_y_values: average_correct_y_height += value
average_correct_y_height /= 5 # Calculating mean/average of heights.
tk.messagebox.showwarning(title="You will now be monitored", message="We have recorded your correct posture. We will now start monitoring you. Please proceed with your tasks.")

cool_down = 0 # Amount of iters before the user is sent a warning again. usually, the user needs some time to adjust back into correct posture.
while True:
    time.sleep(0.5) # Wait for a little time between each run to not clog up memory.
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_AREA)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Our inbuilt xml document that contains data about how to scan faces (which is generated )
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 10, 350)
    i_blurred = cv2.GaussianBlur(grayscale, ksize = (29,1), sigmaX=0)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi/180.0, threshold=1, minLineLength=1, maxLineGap=20)
    faces = face_cascade.detectMultiScale(frame, 1.03, 4, minSize=(100, 100))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
    
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
        cv2.line(frame, (0, int(average_correct_y_height)), (frame.shape[1], int(average_correct_y_height)), (0, 255, 0), 2) # green, correct height
        cv2.line(frame, (0, int(average_shoulder_height)), (frame.shape[1], int(average_shoulder_height)), (0, 0, 255), 2) # red, current height
        cv2.imshow('Input', frame)
    except(Exception):
        continue



    # Determine slouching by checking the reference line angle against a threshold
    slouch_threshold = 30  # Adjust the threshold as needed
    if abs(shoulder_slouch_diff) > slouch_threshold: # shoulder slouch diff is usually a negative value, so have to get absolute value for it to make sure program detects incorrect posture properly.
        print("Bad posture")
        if selected_option == "Alert Box": 
            if cool_down == 0:
                tk.messagebox.showwarning(title = "ALERT", message = f"FIX YOUR POSTURE RIGHT NOW")
                cool_down = 5
            else: cool_down -= 1
        elif selected_option == "Notification Tray": 
            if cool_down == 0:
                notification.notify(
                    title = 'Posturiser',
                    message = 'CORRECT YOUR POSTURE!',
                    app_icon = None,
                    timeout = 0.5,
                )
                cool_down = 5
            else: cool_down -= 1
        #print(f'correct level is {average_correct_y_height} \n current level is {average_shoulder_height} \n difference is {shoulder_slouch_diff}')
    else:
        print("Good posture")
        cool_down = 5
        #print(f'correct level is {average_correct_y_height} \n current level is {average_shoulder_height} \n difference is {shoulder_slouch_diff}')


    
    
    
    c = cv2.waitKey(1)
    if c == 1:
        break
mainloop()
cap.release()
cv2.destroyAllWindows()