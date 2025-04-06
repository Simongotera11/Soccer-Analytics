import pickle
import os
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
import supervision as sv
from utils import *

class Tracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    def add_position_to_tracks(sekf,tracks):
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
        
    def detect_frames(self, frames):
        """Detect objects in video frames in batches."""
        detections = []
        for i in range(0, len(frames), 20):
            batch_predictions = self.model.predict(frames[i:i+20], conf=0.1)
            detections.extend(batch_predictions)  # Store all 
            
        return detections
            
    def get_object_tracks(self, frames,read_from_stub= False,stub_path= None):
        """Process object detections and apply tracking."""
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open (stub_path,'rb') as f:
                tracks = pickle.load(f)
            
            return tracks
        detections = self.detect_frames(frames)
        
        tracks ={
            'players':[],
            'referees':[],
            'ball':[]
        }

        for frame_num, detection in enumerate(detections):
            # Ensure each `detection` is a valid object, not a list

            cls_names = detection.names
            print(cls_names)
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            
            # Convert to supervision format
            d_sv = sv.Detections.from_ultralytics(detection)
            
            # Convert Goalkeeper to player
            for object_indx, id in enumerate(d_sv.class_id):
                if cls_names[id] == "goalkeeper":
                    d_sv.class_id[object_indx] = cls_names_inv["player"]
                    
            
            detection_with_tracks =self.tracker.update_with_detections(d_sv)
            
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            
            for frame in detection_with_tracks:
                bbox = frame[0].tolist()
                cls_id = frame[3]
                track_id = frame[4]
                
                
                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox":bbox}
                    
            for frame in d_sv:
                bbox = frame[0].tolist()
                cls_id = frame[3]        
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox":bbox}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
                
        return tracks
    
    def draw_elipse(self,frame,bbox,color,id=None):
        y2 = int(bbox[3])
        x_c,_ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame,center=(x_c,y2),axes=(int(width),int(0.35*width)),angle=0.0,startAngle=-45,endAngle=235,
                    color=color,thickness=2,lineType=cv2.LINE_4)
        
        rect_width = 40
        rect_height =20
        x1_rect = x_c -rect_width //2
        x2_rect = x_c + rect_width //2 
        y1_rect = (y2 -rect_height) +15
        y2_rect = (y2 + rect_height) +15
        
        if id is not None:
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color=color, 
                          thickness=cv2.FILLED)

            x1_text = x1_rect+12
            if id >99:
                x1_text -=10
                
            cv2.putText(frame,f"{id}",(int(x1_text),int(y1_rect+15)), 
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0))
            
        
        return frame
    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1])
        x,_=get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x,y],[x-10,y-20],[x+10,y-20]
        ])
        
        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)
        return frame
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
        
    def draw_annotations(self,video_frames,tracks,team_ball_control):
        output_video_frames =[]
        for  frame_num,frame in enumerate(video_frames):
            
            frame = frame.copy()
            
            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            
            # Draw Player
            for id,player in player_dict.items():
                frame = self.draw_elipse(frame,player['bbox'],player.get('team_color',(0,0,255)),id)
                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame,player['bbox'],(0,0,255))
                
            # Draw Ball
            for id,ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball['bbox'],(0,255,0))
            
            #Draw ball possesion
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)
            
                
            output_video_frames.append(frame)
        return output_video_frames
            
            

        
