
#processing the video as per frames.
#performing alignments (normalized) of the faces to get "clean" visual data.
#calculating the Action Units (AUs)
#saving the single compressed file as .npz files

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import mediapipe as mp #Google Mediapipe Face Mesh Topology

#we need to perform frame resampling and synchronization so that the face pixels and attributes are confined to one particular region.

#visual input for ViT (target): (15, 224, 224, 3)
# we are extracting 15 equi-spaced frames per video sample.

class VideoProcessor():
    def __init__(self, target_size = (224,224), num_frames = 32):
        self.target_size = target_size
        self.num_frames = num_frames
        
        #AFFINE ALIGNMENT:
        #initializing MediaPipe Face Mesh:
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, #video: 15 frames
            max_num_faces=1, #one face in a frame
            refine_landmarks=True, # Critical for accurate eye/iris tracking
            min_detection_confidence=0.5
        )

    def align_and_crop(self, frame, landmarks):
        """
        Rotates and scales the image so eyes are perfectly horizontal and centered.
        This removes head tilt/pose, leaving only the emotion.
        """
        # Get coordinates for Left and Right eyes (Average of corners for stability)
        left_eye = np.mean(landmarks[[33, 133]], axis=0)
        right_eye = np.mean(landmarks[[362, 263]], axis=0)
        
        # 1. Calculate Angle to rotate (Level the eyes)
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) #theta
        
        # 2. Calculate Scale (Normalize distance to camera)
        current_dist = np.sqrt(dX**2 + dY**2)
        desired_dist = self.target_size[0] * 0.35 # Eyes always 35% of image width apart
        scale = desired_dist / (current_dist + 1e-6) #S
        
        # 3. Get Rotation Matrix
        eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2) #C
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale) #shape: M(2x3)
        
        # 4. Adjust Translation (Center the face in the frame)
        # Move eyes to: X=50%*height of the image, Y=35%*width of the image
        #standards used in a lot of models: VGGFace, ArcFace
        
        t_x = self.target_size[0] * 0.5 #standard
        t_y = self.target_size[1] * 0.35 #standard
        M[0, 2] += (t_x - eyes_center[0]) #t_x - c_x
        M[1, 2] += (t_y - eyes_center[1]) #t_Y - c_y
        
        # Apply Affine Transform
        aligned_face = cv2.warpAffine(frame, M, self.target_size, flags=cv2.INTER_CUBIC) 
        return aligned_face

    def process_video(self, video_path):
        #streaming the video frame by frame
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0: 
            cap.release() #empties the video
            return None

        # Sample exactly 'num_frames' indices evenly spaced in the total frames.
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames_out = []
        last_valid_frame = None
        current_frame_idx = 0
        
        #writing a loop to find the 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if current_frame_idx in indices:
                
                # Convert BGR (OpenCV) to RGB (ViT Standard)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #getting back the 468 points/indices/landmarks
                results = self.face_mesh.process(frame_rgb) 
                
                if results.multi_face_landmarks:
                    h, w, _ = frame.shape # (h,w,channel)
                    
                    # Extract landmarks as (x, y) coordinates
                    lms_3d = results.multi_face_landmarks[0].landmark
                    lms_2d = np.array([(lm.x * w, lm.y * h) for lm in lms_3d])
                    
                    # Extracting Aligned and cropped image
                    aligned_img = self.align_and_crop(frame_rgb, lms_2d)
                    
                    frames_out.append(aligned_img)
                    last_valid_frame = aligned_img
                    
                elif last_valid_frame is not None:
                    # Face lost? Forward Fill
                    frames_out.append(last_valid_frame)
                else:
                    # Start of video and no face? Black frame (rare)
                    frames_out.append(np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))
            
            current_frame_idx += 1
            if len(frames_out) == self.num_frames:
                break
                
        cap.release()
        
        # Pad if video ended too early
        if len(frames_out) < self.num_frames:
            while len(frames_out) < self.num_frames:
                if last_valid_frame is not None:
                    frames_out.append(last_valid_frame)
                else:
                    return None 
                    
        return np.array(frames_out) # Shape: (15, 224, 224, 3) (4D)

def main():
    #creating a parser:
    parser = argparse.ArgumentParser(description="Pure Video Processing")
    parser.add_argument(
        "--input_csv", 
        type=str, 
        default="data/interim/metadata.csv"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed/video"
    )
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True) 
    #checks whether there is still the output path else creates one.
    
    df = pd.read_csv(args.input_csv)
    
    #creating a video-only dataset out of the complete metadata.
    # video_df = df[df['Sample_path'].str.endswith('.mp4')].copy()
    
    video_df = df[df['Sample_path'].astype(str).str.split('/').str[-1].str.match(r'^01-.*\.mp4$')].copy()
    
    #initializing the processor.
    processor = VideoProcessor(num_frames=32)
    processed_paths = []
    
    print(f"Processing {len(video_df)} videos...")
    
    #creating a progress bar:
    for _, row in tqdm(video_df.iterrows(), total=len(video_df)):
        raw_file = Path(row['Sample_path'])
        save_path = output_path / f"{raw_file.stem}.npy"
        
        try:
            video_tensor = processor.process_video(raw_file)
            
            if video_tensor is not None:
                # Save as a standard Numpy array
                np.save(save_path, video_tensor)
                processed_paths.append(str(save_path))
            else:
                processed_paths.append(None)
                
        except Exception as e:
            print(f"Error processing {raw_file.name}: {e}")
            processed_paths.append(None)
            
    video_df['video_processed_path'] = processed_paths
    video_df.to_csv("data/interim/video_metadata.csv", index=False)
    print(f"Processing Complete. Data saved to {args.output_dir}")

if __name__ == "__main__":
    main()