import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from tkinter import messagebox
import kociemba
import serial

def initialize_serial_connection():
    # Find available ports
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"Port: {port.device} - {port.description}")
    try:
        ser = serial.Serial('COM6', 115200)  # Replace 'COM6' with the correct port if necessary
        print("Serial port opened successfully.")
        return ser
    except serial.SerialException as e:
        messagebox.showerror("Serial Error", f"Could not open serial port: {e}")
        return None

def display_result(state):
    # Calculate the image dimensions based on the length of the solution string
    text_width = 20 * len(state)  # Estimate width based on average character width
    img_width = max(text_width, 300)  # Ensure minimum width of 300 pixels
    img_height = 300

    # Create a black image
    result_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Write the Rubik's Cube state on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_size = cv2.getTextSize(state, font, font_scale, font_thickness)[0] #getTexsize resultat = tuble ((taille),valeur de posision base ligne)
    text_x = (result_img.shape[1] - text_size[0]) // 2
    text_y = (result_img.shape[0] + text_size[1]) // 2
    cv2.putText(result_img, state, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Display the image
    cv2.imshow("Rubik's Cube State Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class WebcamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Rubik Cube Application")
        self.video_capture = cv2.VideoCapture(0)
        self.current_image = None
        self.canvas_width = 480
        self.canvas_height = 480
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height,bg="#5784BA")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        # Create labels to explain each button
    
        label_instruction=tk.Label(window, text="The order of capture ", justify='left', bg=window['bg'], fg='#FE277E', font=('Arial', 20, 'bold'))
        label_instruction.place(x=750, y=40)
        label_text =    label_text = """
1) Capture the up face with the << white >> center cube ( top face orange centre)


2) Capture the right face with the << blue >> center cube ( top face white centre)


3) Capture the front face with the << red >> center cube (top face white centre)


4) Capture the down face with the << yellow >> center cube ( top face red centre)


5) Capture the left face with the << green >> center cube ( top face white centre)


6) Capture the back face with the << orange >> center cube ( top face white centre)
"""

        label = tk.Label(window, text=label_text, justify='left', bg=window['bg'], fg='#FFFFFF', font=('Arial', 12, 'bold'))
        label.place(x=580, y=100)

        self.capture_button = tk.Button(window, text="Capture", command=self.capture_image, width=15,  bg="#47EAD0", font=('Arial', 12))
        self.capture_button.grid(row=1, column=0, padx=5, pady=5)

        self.solve_button = tk.Button(window, text="Solve Cube", command=self.solve_cube, width=15, state="disabled",  bg="#FE277E", font=('Arial', 12))
        self.solve_button.grid(row=1, column=1, padx=5, pady=5)

        self.remove_last_button = tk.Button(window, text="Remove Last Matrix", command=self.remove_last_matrix, width=15, bg="#47EAD0", font=('Arial', 12))
        self.remove_last_button.grid(row=1, column=2, padx=5, pady=5)

        self.exit_button = tk.Button(window, text="Exit", command=self.exit_program, width=15, bg="#47EAD0", font=('Arial', 12))
        self.exit_button.grid(row=2, column=2, padx=5, pady=5)
        
        self.restart_button = tk.Button(window, text="Restart", command=self.restart, width=15, bg="#47EAD0", font=('Arial', 12))
        self.restart_button.grid(row=2, column=0, padx=5, pady=5)
        
        self.cube_states = []  # Initialize cube_states list
        self.captured_images = []  # Initialize list to store captured images
        self.update_webcam()

    def preprocess_image(self, frame):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Convert the frame to LAB color space
        lab_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
        
        # Apply Histogram Equalization to the luminance channel
        lab_frame[0,:,:] = cv2.equalizeHist(lab_frame[0,:,:])
        
        # Convert the frame back to BGR color space
        equalized = cv2.cvtColor(lab_frame, cv2.COLOR_Lab2BGR)
        
        # Sharpening filter
        sharpen_filter = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        
        sharp_image = cv2.filter2D(equalized, -1, sharpen_filter)
        
        # Adjust brightness and contrast
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 10  # Brightness control (0-100)
        adjusted_image = cv2.convertScaleAbs(sharp_image, alpha=alpha, beta=beta)

        return adjusted_image

    def get_square_colors(self, frame):
        processed_frame = self.preprocess_image(frame)

        # Convert the image to the HLS color space
        hls = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HLS)

        # Define the lower and upper bounds for each color in HLS
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([80, 255, 255])

        lower_yellow = np.array([22,146,80]) 
        upper_yellow = np.array([70 ,255, 255])

        lower_white = np.array([0, 200, 0])
        upper_white = np.array([180, 255, 255])

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])

        lower_orange = np.array([10, 50, 50])
        upper_orange = np.array([25, 255, 255])

        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Define color names
        color_names = ['G', 'Y', 'W', 'R', 'O', 'B']

        # Divide the image into a 3x3 grid
        rows, cols, _ = frame.shape
        grid_size = min(rows, cols) // 3

        # Initialize a 3x3 matrix to store the colors
        colors_matrix = [['' for _ in range(3)] for _ in range(3)]

        # Iterate through each sub-square of the 3x3 grid
        for i in range(3):
            for j in range(3):
                sub_roi = hls[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size]

                # Determine the dominant color in each sub-square
                mask_green = cv2.inRange(sub_roi, lower_green, upper_green)
                mask_yellow = cv2.inRange(sub_roi, lower_yellow, upper_yellow)
                mask_white = cv2.inRange(sub_roi, lower_white, upper_white)
                mask_red = cv2.inRange(sub_roi, lower_red, upper_red)
                mask_orange = cv2.inRange(sub_roi, lower_orange, upper_orange)
                mask_blue = cv2.inRange(sub_roi, lower_blue, upper_blue)

                masks = [mask_green, mask_yellow, mask_white, mask_red, mask_orange, mask_blue]

                max_color_index = np.argmax([np.sum(mask) for mask in masks])

                colors_matrix[i][j] = color_names[max_color_index]

        return colors_matrix

    def update_webcam(self):
        ret, frame = self.video_capture.read()
        if ret:
            colors_matrix = self.get_square_colors(frame)

            # Draw grid lines
            rows, cols, _ = frame.shape
            grid_size = min(rows, cols) // 3
            color = (186, 132, 87)
            for i in range(1, 3):
                cv2.line(frame, (0, i * grid_size), (cols, i * grid_size), color, 2)
                cv2.line(frame, (i * grid_size, 0), (i * grid_size, rows), color, 2)

            # Display color names in the center of each square
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1  # Increase font scale for bigger text
            font_thickness = 2  # Increase font thickness for bold text
            text_color =  (10, 15, 12)  # BGR equivalent of #CA3C66
            for i in range(3):
                for j in range(3):
                    text_size = cv2.getTextSize(colors_matrix[i][j], font, font_scale, font_thickness)[0]
                    text_x = j * grid_size + (grid_size - text_size[0]) // 2
                    text_y = i * grid_size + (grid_size + text_size[1]) // 2
                    cv2.putText(frame, colors_matrix[i][j], (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.current_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            self.window.after(18, self.update_webcam)

    def capture_image(self):
        # Check if the number of captured images is less than 6
        if len(self.captured_images) < 6:
            self.solve_button.config(state="disabled")
            ret, frame = self.video_capture.read()
            if ret:
                colors_matrix = self.get_square_colors(frame)
                print("Rubik's Cube Colors:")
                for i, row in enumerate(colors_matrix):
                    print(row)
                    for j, color in enumerate(row):
                        # Replace colors with corresponding letters
                        if color == 'R':
                            colors_matrix[i][j] = 'F'
                        elif color == 'W':
                            colors_matrix[i][j] = 'U'
                        elif color == 'G':
                            colors_matrix[i][j] = 'L'
                        elif color == 'B':
                            colors_matrix[i][j] = 'R'
                        elif color == 'O':
                            colors_matrix[i][j] = 'B'
                        elif color == 'Y':
                            colors_matrix[i][j] = 'D'

                # Add the Rubik's Cube state to the list
                self.cube_states.append(''.join([''.join(row) for row in colors_matrix]))
                # Create a new frame for the captured image
                captured_frame = tk.Frame(self.window, width=240, height=240, bd=1, relief=tk.RIDGE)
                captured_frame.grid(row=4, column=len(self.captured_images), padx=10, pady=10, sticky='e')

                # Convert the captured frame to ImageTk format and display it in the frame
                captured_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                resized_image = captured_image.resize((140, 140))
                captured_photo = ImageTk.PhotoImage(image=resized_image)
                captured_label = tk.Label(captured_frame, image=captured_photo)
                captured_label.image = captured_photo
                captured_label.pack()

                # Store the captured image and frame for later use
                self.captured_images.append((captured_image, captured_frame))
                if len(self.captured_images) == 6:
                    self.solve_button.config(state="normal")
        else:
            # If the number of captured images exceeds 6, display a message box
            messagebox.showinfo("Limit Exceeded", "You have already captured 6 images. Cannot capture more.")

    def restart(self):
        # Destroy captured frames and delete associated images
        for image, frame in self.captured_images:
            frame.destroy()
            del image
        self.captured_images.clear()

        # Clear cube states
        self.cube_states.clear()

        # Clear canvas
        self.canvas.delete("all")

        # Restart webcam update
        self.update_webcam()

    def remove_last_matrix(self):
        if len(self.captured_images) > 0:
            # Remove the last captured image and canvas from the GUI
            last_captured_image, last_captured_frame = self.captured_images.pop()
            self.cube_states.pop()
            last_captured_frame.destroy()
            print("Last capture removed.")

    def solve_cube(self):
        if len(self.cube_states) == 6: 
            concatenated_state = ''.join(self.cube_states)
            print(f"\nRubik's Cube State List : ('{concatenated_state}')")

            # Use kociemba to solve the Rubik's Cube
            solution = kociemba.solve(concatenated_state)
            print(f"\nKociemba Solution: {solution}")

            # Convert the notation from U', F', R', D', L', B' to U3, F3, R3, D3, L3, B3
            converted_solution = solution.replace("U'", "U3").replace("F'", "F3").replace("R'", "R3").replace("D'", "D3").replace("L'", "L3").replace("B'", "B3")
            display_result(converted_solution)
            print(f"Converted Solution: {converted_solution}")

            ser.write(converted_solution.encode())

            # Clear the list for the next set of six matrices
            self.cube_states.clear()

            # Update the Solve Cube button font to bold

    def exit_program(self):
        if messagebox.askokcancel(title="Quit?", message="Do you really want to quit?"):
            self.window.destroy()

# Create the main application window
fenetre = tk.Tk()
fenetre.geometry('1450x800')
fenetre.title('Rubik Cube Interface')
fenetre['bg'] = '#0C0F0A'
fenetre.resizable(height=False, width=False)

# Initialize the WebcamApp class with the main window
app = WebcamApp(fenetre)

# Start the Tkinter event loop
fenetre.mainloop()