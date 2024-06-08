import cv2
import numpy as np
import tensorflow as tf
import os

def overlay_predictions(frames, ball_positions):
    '''
    Overlay the predicted ball positions onto every frame in the mp4 file

    args:
    frames (str): Path to the input mp4 file of a ping pong game
    ball_positions (tf.Tensor): Tensor of dimensions (batch_size, 3), where each row corresponds to (frame_number, xCoord, yCoord)

    returns: Path to the overlaid mp4 file
    '''
    # Open the input video
    cap = cv2.VideoCapture(frames)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(os.getcwd(), 'output_video_with_tracking.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Convert ball_positions tensor to numpy array for easier indexing
    ball_positions_np = ball_positions.numpy()
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find the ball position for the current frame
        ball_position = ball_positions_np[ball_positions_np[:, 0] == frame_idx]
        if ball_position.shape[0] > 0:
            x, y = ball_position[0, 1:].astype(int)
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Draw a green circle for the ball
        
        # Write the frame to the output video
        out.write(frame)
        
        frame_idx += 1
    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Output video saved at: {output_video_path}")
    return output_video_path

# Example usage
frames = 'input_video.mp4'
ball_positions = tf.constant([
    [0, 100, 200],
    [5, 150, 250],
    [10, 200, 300]
], dtype=tf.float32)

output_path = overlay_predictions(frames, ball_positions)
print(f"Output video saved at: {output_path}")
