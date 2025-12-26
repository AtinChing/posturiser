import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

def show_alert():
    selected_option_value = selected_option.get()
    messagebox.showinfo("Alert", f"Selected Option: {selected_option_value}")

# Create the main window
root = tk.Tk()
root.title("Alert Box with Dropdown Menu")

# Create and set up the Combobox
options = ["Option 1", "Option 2", "Option 3"]
selected_option = tk.StringVar(value=options[0])
combo = ttk.Combobox(root, values=options, textvariable=selected_option)
combo.pack(pady=10)

# Create a button to trigger the alert
button_show_alert = tk.Button(root, text="Show Alert", command=show_alert)
button_show_alert.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()