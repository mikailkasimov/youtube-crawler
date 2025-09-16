import yt_dlp
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def sanitize_filename(title):
    title = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', title)
    title = re.sub(r'[“”]', '"', title)  # Normalize quotes
    return title.strip()

""" 
Downloads youtube video to output_dir

param: video_link: link to youtube video
param: output_dir, directory to specify output to
return: path of video
"""
def download_yt_video(video_link, output_dir="."):
    # First, extract video info to get the title
    ydl = yt_dlp.YoutubeDL({'quiet': True})
    info = ydl.extract_info(video_link, download=False)
    title = sanitize_filename(info['title'])

    output_path = os.path.abspath(output_dir)
    full_path = os.path.join(output_path, f"{title}.mp4")

    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': full_path,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_link])

    return full_path

""" 
Same functionality as download_yt_video, except that its performed on
a list

param: link_list: list of links to youtube videos
param: output_dir, directory to specify each output to
return: list of paths to downloaded videos
"""
def download_yt_video_list(link_list, output_dir="."):
    video_paths = []
    for link in link_list:
        path = download_yt_video(link)
        if path:
            video_paths.append(path)
    return video_paths

""" 
Gets video links and from YoutubeAPI response and stores them in a list

param: results: Response from a YoutubeAPI Search request
return: List of youtube links

"""
def get_video_links(results):
    base_url = "https://www.youtube.com/watch?v="
    links = []
    for item in results:
        video_id = item.get("id", {}).get("videoId")
        if video_id:
            links.append(f"{base_url}{video_id}")
    return links


""" 
Loads embeddings from directory
The function expects the file structure:

|Embeddings_Directory
|----|Rep_Harriet_M_Hageman.npy
|----|Sen_Chuck_Schumer.npy
|----|...

In each .npy file, the shape is (N, D), with N representing the amount
of embeddings for that senator, and D being the dimension of the embedding (512 for facenet).

Warning: There are a different amount of embeddings for each legislator, so the list cannot be a
reshaped as a true nparray of dimension 3 (cube)!

param: embeddings_directory: directory of embeddings

return: N x M_i x D list, where N represents the number of unique legislators,
M_i is the number of embeddings for the current legislator, and D is the dimension
of the embeddings (facenet is 512).

return: label_map Dict{}, where the key is the numerical idx of the legislator, and 
the value is the file name of the .npy file
"""
def load_embeddings(embeddings_directory, omitSuffix=""):
    all_embeddings = []
    label_map = {}
    for i,file in enumerate(os.listdir(embeddings_directory)):
        filepath = os.path.join(embeddings_directory, file)
        vectors = np.load(filepath)
        all_embeddings.append(vectors)
        label_map[i] = file.removesuffix(omitSuffix)

    return all_embeddings, label_map

""" 
Calculates centroids for a list of N legislators

params: N x M x D list, where N represents the number of unique legislators,
M is the number of embeddings for the current legislator, and D is the dimension
of the embeddings (facenet is 512).

returns: N x D nparray, where N represents the number of unique legislator, and
         D is the dimension of the corresponding centroid (facenet is 512)
"""
def calculate_centroids(all_embeddings):
    centroids = []
    for vectors in all_embeddings:
        centroid = np.mean(vectors, axis=0)
        centroids.append(centroid)
    return np.stack(centroids)




def plot_similarity_from_labels(labels_dict, label_map):
    all_names = []
    all_scores = []

    for frame_id, entries in labels_dict.items():
        for entry in entries:
            label_idx = entry["label"]
            score = entry["score"]
            name = centroid_key[label_idx][:-19]  # strip _Face_Embeddings or similar suffix
            all_names.append(name)
            all_scores.append(score * 100)  # convert to %

    if not all_names:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.scatter(all_names, all_scores, alpha=0.7, color='darkgreen')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Detected Person")
    plt.ylabel("Cosine Similarity (%)")
    plt.title("Face Recognition Similarity Scores")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
