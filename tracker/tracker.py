from ultralytics import YOLO 
import cv2
import pandas as pd
import supervision as sv
import numpy as np
import pickle
import os

from utils import get_bbox_width, get_center_of_bbox, get_foot_position

class Tracker:
    def __init__(self,model_path, tracker):
        self.model = YOLO(model_path)
        self.tracker = tracker
        
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
        
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    
    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.6)
            detections += detections_batch  
        return detections
    
    def get_object_tracks(self,frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as f:
                    tracks = pickle.load(f)  
                return tracks
            except Exception as e:
                print(e)
            finally:
                f.close()
                
        detections = self.detect_frames(frames) 
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names 
            cls_name_inv = {v:k for k,v in cls_names.items()} 
            # print(cls_names)
            # print('---------------')
            
            ## Convert to supervision Detection format 
            detection_supervision = sv.Detections.from_ultralytics(detection) 
            print(detection_supervision)
            
            ## Convert GoalKeepr to player, later using embeddings we will identify him correctly
            ## Else remove this below flow loop 
            # for object_ind, class_id in enumerate(detection_supervision.class_id):
            #     if cls_names[class_id] == 'goalkeeper':
            #         detection_supervision.class_id[object_ind] = cls_name_inv["player"]
            # # Track obejcts
            
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision )
            
            ## Initalize the  tracks to append it in dict format
            tracks["players"].append({}) 
            tracks["referees"].append({}) 
            tracks["ball"].append({}) 
            
            ## Adding it into the boxes
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist() 
                cls_id = frame_detection[3]
                track_id = frame_detection[4] 
                
                if cls_id == cls_name_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id == cls_name_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_name_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        
        if stub_path:
            ## Trying it in try-catch block to ensure the safety of file
            try:
                with open(stub_path, "wb") as f:
                    pickle.dump(tracks, f) 
                
            except Exception as e:
                print(e) 
            finally:
                f.close()
                
        return tracks 
    
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, y_center = get_center_of_bbox(bbox) 
        width = get_bbox_width(bbox) 
        
        cv2.ellipse(
            frame, 
            center=(x_center,y2), 
            axes=(int(width), int(0.35*width)),
            angle=0.0 ,
            startAngle=-45 ,
            endAngle=235,
            color=color, 
            thickness=2 ,
            lineType=cv2.LINE_4 
        )
        
        reactangle_width = 40 
        reactangle_height = 20 
        x1_rect = x_center-reactangle_width//2
        x2_rect = x_center +reactangle_width//2 
        y1_react = (y2 - reactangle_height//2)+15 
        y2_rect = (y2 + reactangle_height//2)+15
        if track_id:
            cv2.rectangle(frame, (int(x1_rect), int(y1_react)), (int(x2_rect),int(y2_rect)), color, cv2.FILLED)
            
            x1_text = x1_rect+12 
            if track_id>99:
                x1_text -= 10 
            cv2.putText(
                frame, 
                f"{track_id}",
                (int(x1_text), int(y1_react+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        
        return frame
    
    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox) 
        
        traingle_points = np.array([
            [x,y], 
            [x-10, y-20], 
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [traingle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [traingle_points], 0, color, 2)
        return frame

    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() ## To not broke change the original one 
            
            player_dit = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num] 
            
            ## Draw players  
            
            for track_id, player in player_dit.items(): ## For player
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color ,track_id) 
                
            for track_id, referee in referee_dict.items(): ## For referee
                frame = self.draw_ellipse(frame, referee['bbox'], (0,255,0) ,track_id) 
                
            ## Draw ball
            for track_id, ball in ball_dict.items(): ## For ball
                frame = self.draw_ellipse(frame, ball['bbox'], (255,0,0) ,track_id) 
                frame = self.draw_traingle(frame, ball["bbox"], (255,0,0))
                
            # frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

                
            output_video_frames.append(frame) 
            
        return output_video_frames
                