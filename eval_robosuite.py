import os
import imageio

def save_video(video_path_name, datas, fps=20):
    if not video_path_name.endswith(".mp4"):
        video_path_name += ".mp4"
    
    video_writer = imageio.get_writer(video_path_name, fps=fps)
    # print(f"number of eps: {len(datas)}")
    for data in datas:
        for img in data:
            video_writer.append_data(img[::-1])
            
    video_writer.close()
        
    