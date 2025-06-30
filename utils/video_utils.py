import cv2 

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames =[]
    while True:
        ret, frame = cap.read()
        if not ret: ## If the video ends break the loop
            break 
        frames.append(frame)  
    return frames 



def save_video(output_video_frames, output_video_path, reference_video_path=None):
    if not output_video_frames:
        print("No frames to save.")
        return
    
    # get fps from reference video if provided
    fps = 30  # default fallback
    if reference_video_path:
        cap = cv2.VideoCapture(reference_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    # Get frame size from first frame
    height, width, _ = output_video_frames[0].shape
    frame_size = (width, height)

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    try:
        for frame in output_video_frames:
            out.write(frame)
    except Exception as e:
        print(f"Error writing video: {e}")
    finally:
        out.release()
        print(f"Video saved successfully at {output_video_path} with {fps:.2f} fps.")

        