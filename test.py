import cv2
import numpy as np

def get_square_colors(frame):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for each color
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 30, 255])

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_orange = np.array([3, 100, 100])
    upper_orange = np.array([20, 255, 255])

    lower_blue = np.array([110, 50, 50])
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
            sub_roi = hsv[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size]

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

            # Draw rectangles around the sub-squares in the 3x3 grid
            x, y, w, h = j * grid_size, i * grid_size, grid_size, grid_size
            color = (0, 255, 0)  # Green color for rectangles
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Add the name of the color to each square (in black)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(colors_matrix[i][j], font, font_scale, font_thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(frame, colors_matrix[i][j], (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return colors_matrix

# Open a connection to the camera (you may need to change the camera index)
cap = cv2.VideoCapture(1)

cube_states = []  # List to store Rubik's Cube states

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    # Get colors of each sub-square in the 3x3 grid
    colors_matrix = get_square_colors(frame)

    # Display the image
    cv2.imshow("Rubik's Cube", frame)

    # Check if the 'c' key is pressed
    key = cv2.waitKey(0)
    if key & 0xFF == ord('c'):
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
        cube_states.append(''.join([''.join(row) for row in colors_matrix]))

    # Print the concatenated character list as a string in the desired format after six matrices
    if len(cube_states) == 6:
        concatenated_state = ''.join(cube_states)
        print(f"\nRubik's Cube State List: ('{concatenated_state}')")

        # Use kociemba to solve the Rubik's Cube

        cube_states = []  # Clear the list for the next set of six matrices

    # Exit the loop if the 'q' key is pressed
    elif key & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()