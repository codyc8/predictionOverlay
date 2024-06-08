import cv2
import numpy as np
import tensorflow as tf
from inputProcessing import format_output

def overlay_predictions(frames, ball_positions, tracker_color=(0, 255, 0), trail_duration=0.5):
    '''
    Overlay the predicted ball positions onto every frame in the mp4 file, leaving a trail for the ball.

    args:
    frames (str): Path to the input mp4 file of a ping pong game
    ball_positions (tf.Tensor): Tensor of dimensions (batch_size, 3), where each row corresponds to (frame_number, xCoord, yCoord)
    tracker_color (tuple): BGR color values for the tracker
    trail_duration (float): Duration (in seconds) for which the ball's trail should be visible

    returns: Path to the overlaid mp4 file
    '''
    # Open the input video
    cap = cv2.VideoCapture(frames)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = 'output_video_with_tracking.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Convert ball_positions tensor to numpy array for easier indexing
    ball_positions_np = ball_positions.numpy()

    # Calculate the number of frames the trail should last
    trail_frames = int(trail_duration * fps)

    # Initialize a list to store recent ball positions
    recent_positions = []

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Find the ball position for the current frame
        ball_position = ball_positions_np[ball_positions_np[:, 0] == frame_idx]
        if ball_position.shape[0] > 0:
            x, y = ball_position[0, 1:].astype(int)
            recent_positions.append((frame_idx, x, y))

        # Remove old positions outside the trail duration
        recent_positions = [pos for pos in recent_positions if frame_idx - pos[0] <= trail_frames]

        # Overlay the recent positions with decreasing opacity
        for i, (f_idx, x, y) in enumerate(recent_positions):
            alpha = (trail_frames - (frame_idx - f_idx)) / trail_frames
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), 10, tracker_color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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
frames = 'data/game_1.mp4'
ball_positions = 'data/game_1_ball_markup.json'
ball_positions = format_output(ball_positions)

# Specify the tracker color (e.g., Red) and trail duration (e.g., 0.5 seconds)
tracker_color = (0, 0, 255)
trail_duration = 0.5

output_path = overlay_predictions(frames, ball_positions, tracker_color, trail_duration)
print(f"Output video saved at: {output_path}")
