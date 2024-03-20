import cv2

# Open the camera
phoneCam = 2  # Change this to the appropriate camera index if needed
cap = cv2.VideoCapture(phoneCam)

# Initialize variables to store max and min x, y coordinates
max_x = max_y = float('-inf')
min_x = min_y = float('inf')

# Read frames from the camera
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Update max and min x, y coordinates
    max_x = max(max_x, width)
    max_y = max(max_y, height)
    min_x = min(min_x, 0)
    min_y = min(min_y, 0)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop on pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the maximum and minimum x, y coordinates
print("Max X:", max_x)
print("Min X:", min_x)
print("Max Y:", max_y)
print("Min Y:", min_y)
