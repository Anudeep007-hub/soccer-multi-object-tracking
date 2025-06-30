## pip imports
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import cv2
import numpy as np
import os 
import datetime 
import time

## Local imports
from utils import read_video, save_video 
from tracker import Tracker
from AssignTeam import TeamAssigner


os.makedirs("output_videos", exist_ok=True)
os.makedirs("input_videos", exist_ok=True)
os.makedirs("stubs", exist_ok=True)
os.makedirs("models", exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
OUTPUT_VIDEO_PATH = os.path.join("output_videos", f"output_video_{timestamp}.mp4")
INPUT_VIDEO_PATH = os.path.join("input_videos", "15sec_input_720p.mp4")
MODEL_PATH = os.path.join("models", "best.pt")
EMBEDDINGS_PATH = os.path.join("stubs", "track_pickle.pkl")

def run_yolo_botsort_tracking(model_path, video_path):
    """
    Runs YOLO detection with BoT-SORT tracking and returns after showing the frames.
    """
    model = YOLO(model_path)

    results = model.track(
        source=video_path,
        tracker="botsort.yaml",
        conf=0.3,
        iou=0.3,
        show=False,
        stream=True , # yields generator of results
        save=True
    )

    for result in results:
        frame = result.orig_img.copy()

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            track_id = int(box.id) if box.id is not None else 0

            cx = int((x1 + x2) / 2)
            cy = int(y2)

            # draw a circle
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # draw ID text
            cv2.putText(frame, f'ID {track_id}', (cx - 10, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("BoT-SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main_2(tracker_model):
    video_frames = read_video(INPUT_VIDEO_PATH)
    
    # Initialize tracker
    tracker = Tracker(MODEL_PATH, tracker_model)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=EMBEDDINGS_PATH)
    
    # Assign Player Teams 
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks["players"][frame_num][player_id]['team'] = team
            tracks["players"][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
            
    # Draw output 
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Save video
    save_video(output_video_frames, OUTPUT_VIDEO_PATH)
    
    

if __name__ == "__main__":
    byte_tracker_instance = sv.ByteTrack()
    
    deep_sort_instance = DeepSort(
        max_age=60,                 
        n_init=3,                   
        max_cosine_distance=0.2,   
        embedder="mobilenet",        
        half=True,                  
        bgr=True
    )
    
    s = input("Choose one\n1-ByteTrack, 2-DeepSORT, 3-BotSORT\n ") 
    
    # make sure logs folder exists
    os.makedirs("logs", exist_ok=True)

    start_time = time.time()

    if s == '1':
        main_2(byte_tracker_instance)
        tracker_name = "ByteTrack"
    elif s == '2':
        main_2(deep_sort_instance)
        tracker_name = "DeepSort"
    elif s == '3':
        run_yolo_botsort_tracking(MODEL_PATH, INPUT_VIDEO_PATH)
        tracker_name = "BoT-SORT"
    else:
        print("Enter a valid input!!")
        tracker_name = None

    end_time = time.time()
    time_taken = end_time - start_time

    if tracker_name:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"logs/{tracker_name}--elapsed_{timestamp}.log"
        with open(log_filename, "w") as log_file:
            log_file.write(f"{tracker_name} execution completed in {time_taken:.2f} seconds.\n")

    print(f"{tracker_name} took {time_taken:.2f} seconds.") if tracker_name else None
