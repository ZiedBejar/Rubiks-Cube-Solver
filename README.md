# Rubik's Cube Color Recognition and Solver Application

### Overview
This application provides a graphical interface for capturing and processing images of a Rubik's Cube, identifying each face's colors, and solving the cube using the Kociemba algorithm. The solution is displayed graphically, allowing users to visualize and understand the Rubik's Cube state and its solution steps.

The code utilizes the following technologies:
- **Tkinter**: For creating the GUI.
- **OpenCV**: For video capture and image processing.
- **NumPy**: For handling matrix and numerical operations.
- **kociemba library**: To calculate the Rubik's Cube solution.
- **Serial Communication**: To interface with external hardware if required.

## Features
- **Webcam Integration**: Capture images from a connected webcam to detect the Rubik's Cube colors.
- **Color Detection and Recognition**: Identifies colors on each face of the Rubik's Cube using predefined color ranges.
- **Cube Solving**: Computes the cube solution and displays it to the user.
- **Serial Communication**: Establishes a serial connection to control external devices if necessary.

## Requirements
- **Python 3.x**
- **Tkinter**: (comes pre-installed with Python)
- **OpenCV**: `pip install opencv-python`
- **Pillow**: `pip install pillow`
- **NumPy**: `pip install numpy`
- **kociemba**: `pip install kociemba`
- **PySerial**: `pip install pyserial`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/ZiedBejar/Rubiks-Cube-Solver.git
   
