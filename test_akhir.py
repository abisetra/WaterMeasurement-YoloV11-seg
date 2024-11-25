import numpy as np
import cv2
import torch
from ultralytics import YOLO
import argparse
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parsing command-line arguments
parser = argparse.ArgumentParser(description="Water Level Detection with YOLOv8-Seg")
parser.add_argument("input_video", type=str, help="The path of the input video.")
parser.add_argument("--model_path", type=str, default="best.pt", help="Path to the trained YOLO segmentation model.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to save the video detection results.")
parser.add_argument("--save_roi", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save grayscale ROI images.")
parser.add_argument("--save_csv", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to save water level data to a CSV file.")
parser.add_argument("--top_left", type=str, default="806,70", help="Top-left corner of the ROI (x,y).")
parser.add_argument("--top_right", type=str, default="975,68", help="Top-right corner of the ROI (x,y).")
parser.add_argument("--bottom_left", type=str, default="808,641", help="Bottom-left corner of the ROI (x,y).")
parser.add_argument("--bottom_right", type=str, default="984,641", help="Bottom-right corner of the ROI (x,y).")
parser.add_argument("--top_cm", type=float, default=400.0, help="Top of the measurement scale in centimeters.")
parser.add_argument("--bottom_cm", type=float, default=0.0, help="Bottom of the measurement scale in centimeters.")
args = parser.parse_args()

# Parse the ROI corner points
top_left = tuple(map(int, args.top_left.split(',')))
top_right = tuple(map(int, args.top_right.split(',')))
bottom_left = tuple(map(int, args.bottom_left.split(',')))
bottom_right = tuple(map(int, args.bottom_right.split(',')))

# Calculate ROI parameters
roi_x = top_left[0]
roi_y = top_left[1]
roi_w = top_right[0] - top_left[0]
roi_h = bottom_left[1] - top_left[1]

# Define measurement range
top_cm = args.top_cm
bottom_cm = args.bottom_cm

# Conversion function from pixels to centimeters
def convert_px_to_cm(y_px, top_px, bottom_px, top_cm, bottom_cm):
    if top_px >= bottom_px:
        print("Warning: top_px should be less than bottom_px.")
    cm_per_px = (bottom_cm - top_cm) / (bottom_px - top_px)
    cm_value = top_cm + (y_px - top_px) * cm_per_px
    return cm_value

# Load the YOLO model
model = YOLO(args.model_path)

# Open video input file
vid = cv2.VideoCapture(args.input_video)
if not vid.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
video_frame_cnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(vid.get(cv2.CAP_PROP_FPS))

# Prepare video writer for saving the detection results
if args.save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('Water_Level_Detection_Result.mp4', fourcc, video_fps, (video_width, video_height))

# Create folders for ROI images if needed
if args.save_roi and not os.path.exists("roi_images"):
    os.makedirs("roi_images")

# Ensure PyTorch uses GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Variables for water level tracking
last_valid_water_level_cm = None
last_valid_adjusted_water_level = None
water_levels = []
frame_numbers = []

# Initialize live plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot([], [], marker='o', color='blue', label="Water Level (cm)")
ax.set_xlim(0, video_frame_cnt)
ax.set_ylim(bottom_cm, top_cm)
ax.set_xlabel("Frame Number")
ax.set_ylabel("Water Level (cm)")
ax.set_title("Live Water Level Detection")
ax.grid()
ax.legend()

def update_plot():
    """Update the live plot."""
    line.set_xdata(frame_numbers)
    line.set_ydata(water_levels)
    ax.set_xlim(0, max(frame_numbers) + 1 if frame_numbers else 1)
    ax.set_ylim(min(water_levels) - 10 if water_levels else bottom_cm, max(water_levels) + 10 if water_levels else top_cm)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Process each frame
for frame_idx in range(video_frame_cnt):
    ret, frame = vid.read()
    if not ret:
        print(f"Error: Unable to read frame {frame_idx}. Exiting.")
        break

    # Extract ROI from the frame
    region_of_interest = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), thickness=3)

    # Perform YOLO inference
    results = model(region_of_interest, device=device)
    masks = results[0].masks
    water_mask = None

    if masks is not None and len(masks.data) > 0:
        # Extract the water mask
        water_mask = masks.data[0]
        binary_mask = water_mask.cpu().numpy().astype(np.uint8)
        binary_mask = cv2.resize(binary_mask, (roi_w, roi_h))

        # Calculate the water level in pixels and centimeters
        nonzero_indices = np.where(binary_mask > 0)
        if len(nonzero_indices[0]) > 0:
            water_level_px = min(nonzero_indices[0])
            adjusted_water_level = roi_y + water_level_px
            water_level_cm = convert_px_to_cm(adjusted_water_level, roi_y, roi_y + roi_h, top_cm, bottom_cm)

            # Update water level data
            last_valid_water_level_cm = water_level_cm
            last_valid_adjusted_water_level = adjusted_water_level
            water_levels.append(water_level_cm)
            frame_numbers.append(frame_idx)

            # Draw the detection line
            cv2.line(frame, (roi_x, int(adjusted_water_level)), (roi_x + roi_w, int(adjusted_water_level)), (0, 255, 0), 4)
            cv2.putText(frame, f"Water Level = {water_level_cm:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # Retain the last valid water level if no detection
        if last_valid_adjusted_water_level is not None:
            cv2.line(frame, (roi_x, int(last_valid_adjusted_water_level)), (roi_x + roi_w, int(last_valid_adjusted_water_level)), (0, 255, 255), 4)
            cv2.putText(frame, f"Water Level = {last_valid_water_level_cm:.2f} cm (Retained)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Update live plot
    update_plot()

    # Show the frame with detection
    cv2.imshow("Water Level Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Save the frame to the output video
    if args.save_video:
        video_writer.write(frame)

# Final plot saving
plt.ioff()
plt.savefig("water_level_chart.png")

# Save water level data to CSV
if args.save_csv and water_levels:
    with open('water_levels.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Water Level (cm)"])
        csv_writer.writerows(zip(frame_numbers, water_levels))

# Release resources
vid.release()
if args.save_video:
    video_writer.release()
cv2.destroyAllWindows()
