import cv2

# Function to handle mouse click events
def onclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: X = {x}, Y = {y}")

# Load the video
video_path = "malam2.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a window to display the video
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", onclick)  # Connect the mouse click event

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display the current frame
    cv2.imshow("Video", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
