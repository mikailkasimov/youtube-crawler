import os
import torch.nn.functional as F
from collections import defaultdict
import torch
import sys
import cv2
import numpy as np
sys.path.append("/home/kasimov/notebooks/LegislatorImagePipeline")
from PreprocessingPipeline import PreprocessingPipeline
from FeatureExtractionPipeline import FeatureExtractionPipeline


class FrameLabeller:
    def __init__(self):
        self.device="cuda" if torch.cuda.is_available() else "cpu"

    """
    Sample frames every frame_rate amount of seconds

    param: frame_rate, sample 1 frame every frame_rate seconds
    param: video, mp4 file path of the video

    return: list of frames
    """
    def get_frames(self, video, interval):
        frames = []
        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)  # Number of frames to skip

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            if frame_count % frame_interval == 0:
                frames.append(frame)
            frame_count += 1
        cap.release()
        return frames


    """
    Create facenet embeddings for one frame

    param: frame, ndarray of image 

    return: N x D, N facenet embeddings of dimension D for N faces
    """
    def generate_embeddings(self, frame, preprocessor=None, extractor=None):
        preprocessor = preprocessor or PreprocessingPipeline()
        extractor = extractor or FeatureExtractionPipeline()

        normalized_images = preprocessor.preprocess_multiple(frame)
        embeddings = extractor.batch_extract_features(normalized_images)  # batched

        return embeddings

    """
    Create facenet embeddings for multiple frames (batched)

    param: frames: List of frames (np.ndarray)

    return: (embeddings, frame_indices)
    """
    def batch_generate_embeddings(self, frames, preprocessor=None, extractor=None):
        preprocessor = preprocessor or PreprocessingPipeline()
        extractor = extractor or FeatureExtractionPipeline()

        all_faces = []
        frame_indices = []

        for i, frame in enumerate(frames):
            faces = preprocessor.preprocess_multiple(frame)
            all_faces.extend(faces)
            frame_indices.extend([i] * len(faces))

        if not all_faces:
            return None, []

        embeddings = extractor.batch_extract_features(all_faces)
        return embeddings, frame_indices


    """
    Assigns a label (nearest centroid) to each face embedding using cosine similarity.

    param: embeddings: torch.Tensor of shape (N, 512) — face embeddings
    param: frame_indices: List[int] of length N — mapping each face to a frame
    param: centroids: torch.Tensor of shape (K, 512) — known identity centroids
    
    return: Dictionary of in format {frame: [centroid for each face]} for each frame
    """
    def label_faces(self, embeddings, frame_indices, centroids):
        if embeddings is None or len(frame_indices) == 0 or len(embeddings) == 0:
            return [], []

        #convert to tensor
        embeddings = torch.from_numpy(embeddings).float()
        centroids = torch.from_numpy(centroids).float()

        #Note: facenet embeddings SHOUULD already be normalized!
        #this is here as an extra precaution, but can be commented
        #out if we are using facenet and want performance
        embeddings = F.normalize(embeddings, p=2, dim=1)
        centroids = F.normalize(centroids, p=2, dim=1)
        
        # Compute cosine similarity matrix between each embedding and each centroid
        similarity_matrix = torch.matmul(embeddings, centroids.T)
        # For each embedding, find the index of the centroid with highest similarity
        scores, labels = torch.max(similarity_matrix, dim=1)


        #convert to return in format
        #{frame1: {
        #    label: [idx1, idx2, ...]
        #    score: [score1, score2, ...]
        #    }
        #frame2: {
        #   ...
        #   }
        #}
        
        grouped = defaultdict(list)
        for frame, label, score in zip(frame_indices, labels, scores):
            grouped[frame].append({"label": label.item(), "score": score.item()})
        
        return dict(grouped)


    #Full pipeline
    """ 
    Full pipeline that extracts embeddings out of all frames and labels them

    param: video: file path of mp4 file to be processed
    param: centroids: N x D, N centroids of dimension D
    param: interval to collect 1 frame per interval seconds

    return: labels in format {frameNumber: [labelIndex for each face detected]} for each frame
    """
    def process(self, video, centroids, interval): 
        #Step 1: generate the frames
        frames = self.get_frames(video, interval=interval)
        #step 2: generate embeddings of all frames
        embeddings, frame_indices = self.batch_generate_embeddings(frames)
        #step 3: label embeddings accords to centroids (using cosine similarity metrid)
        labels = self.label_faces(embeddings, frame_indices, centroids)

        #clear gpu cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return labels


