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
    def __init__(self, data_points, preprocessor=None, extractor=None):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = preprocessor or PreprocessingPipeline()
        self.extractor = extractor or FeatureExtractionPipeline()
        self.data_points = data_points   # <--- centroids used for classification

    def get_frames(self, video, interval):
        """
        Sample frames every frame_rate amount of seconds

        param: frame_rate, sample 1 frame every frame_rate seconds
        param: video, mp4 file path of the video

        return: list of frames
        """
        frames = []
        frames_index = []
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
                frames_index.append(frame_count)
            frame_count += 1
        cap.release()
        return frames, frames_index

    def get_frame_generator(self, video, interval, batch_size):
        """
        Sample frames every frame_rate amount of seconds and yields every batch_size

        param: frame_rate, sample 1 frame every frame_rate seconds
        param: video, mp4 file path of the video

        return: list of frames
        """
        frames = []
        frames_index = []
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_count % frame_interval) == 0:
                frames.append(frame)
                frames_index.append(frame_count)
            frame_count += 1
            if len(frames) == batch_size:
                yield frames, frames_index
                frames = []
                frames_index = []
        #yield leftover frames that did not full batch
        if frames:
            yield frames, frames_index
        cap.release()

    def generate_embeddings(self, frame):
        """
        Create facenet embeddings for one frame

        param: frame, ndarray of image 
        return: N x D, N facenet embeddings of dimension D for N faces
        """
        normalized_faces = self.preprocessor.preprocess(frame)
        embeddings = self.extractor.extract_features(normalized_images)
        return embeddings

    def batch_generate_embeddings(self, frames):
        """
        Create facenet embeddings for each frame

        param: frame, ndarray of image 
        return: List of N x D np.darrays, N facenet embeddings of dimension D for N faces
        """
        preprocessed_faces_per_frame = self.preprocessor.batch_preprocess(frames)
        embeddings_per_frames = self.extractor.batch_extract_features(preprocessed_faces_per_frame)
        return embeddings_per_frames


    def label_faces(self, embeddings, frame_indices):
        """
        Assigns a label (nearest centroid) to each face embedding using cosine similarity.

        param: embeddings: torch.Tensor of shape (N, 512) — face embeddings
        param: frame_indices: List[int] of length N — mapping each face to a frame
        param: centroids: torch.Tensor of shape (K, 512) — known identity centroids
        
        return: Dictionary of in format {frame: [centroid for each face]} for each frame
        """
        if embeddings is None or len(frame_indices) == 0 or len(embeddings) == 0:
            return {}

        embeddings = torch.from_numpy(embeddings).float()
        data_points = torch.from_numpy(self.data_points).float()

        embeddings = F.normalize(embeddings, p=2, dim=1)
        data_points = F.normalize(data_points, p=2, dim=1)

        similarity_matrix = torch.matmul(embeddings, data_points.T)
        scores, labels = torch.max(similarity_matrix, dim=1)

        grouped = defaultdict(list)
        for frame, label, score in zip(frame_indices, labels, scores):
            grouped[frame].append({"label": label.item(), "score": score.item()})

        return dict(grouped)

    #Full pipeline
    def process(self, video_path, interval, batch_size=16):
        """
        Full pipeline for a video using batches of frames.

        param: video_path: path to mp4
        param: interval: seconds between frames
        param: batch_size: number of frames per batch

        return: Dictionary {frame_number: [label dicts]} for all frames
        """
        all_labels = {}  # accumulate labels for all frames
        # Use frame generator
        for frames_batch, frames_index_batch in self.get_frame_generator(video_path, interval, batch_size):
            # Step 1: Preprocess faces for all frames in this batch
            preprocessed_faces_per_frame = self.preprocessor.batch_preprocess(frames_batch)
            # Step 2: Extract embeddings for all frames (flattened internally)
            embeddings_per_frame = self.extractor.batch_extract_features(preprocessed_faces_per_frame)

            # Step 3: Flatten embeddings and create frame mapping
            flat_embeddings = []
            flat_frame_indices = []
            for idx, frame_embeddings in enumerate(embeddings_per_frame):
                for emb in frame_embeddings:
                    flat_embeddings.append(emb)
                    flat_frame_indices.append(frames_index_batch[idx])
            if not flat_embeddings:
                continue  # no faces detected in this batch
            flat_embeddings = np.stack(flat_embeddings)
            # Step 4: Label all embeddings at once
            batch_labels = self.label_faces(flat_embeddings, flat_frame_indices)
            # Step 5: Merge into global labels
            for frame_num, labels_list in batch_labels.items():
                if frame_num not in all_labels:
                    all_labels[frame_num] = labels_list
                else:
                    all_labels[frame_num].extend(labels_list)
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_labels
                
                
