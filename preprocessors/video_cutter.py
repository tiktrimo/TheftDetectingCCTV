from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import xmltodict
from os import listdir, makedirs, path
from os.path import isfile, join
FRAME = 10 * 1 / 30 # 30fps

def cut_videos(source_path, save_path):
    if not path.exists(save_path): makedirs(save_path)
    
    videos= [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".MP4" in f]
    
    
    for progress, video in enumerate(videos):
        curr_xml = open(f"{source_path}/{video[:-4]}.XML", "r").read().rstrip()
        dict_type = xmltodict.parse(curr_xml)

        action_starts = []
        action_ends = []
        for track in dict_type["annotations"]["track"]:
            if "start" in track["@label"]:
                if isinstance(track["box"], list): action_starts.append(int(track["box"][0]["@frame"]))
                else : action_starts.append(int(track["box"]["@frame"]))
            if "end" in track["@label"]:
                if isinstance(track["box"], list): action_ends.append(int(track["box"][0]["@frame"]))
                else : action_ends.append(int(track["box"]["@frame"]))
                


        for index in range(min(len(action_starts), len(action_ends), 1)):
            clip = VideoFileClip(f"{source_path}/{video}", )
            clip = clip.subclip(int(action_starts[index] * FRAME), int(action_ends[index] * FRAME))
            clip.write_videofile(f"{save_path}/{video[:-4]}.mp4", threads = 8, fps=24)
            print(f"{source_path}->{save_path} {progress}/{len(videos)}\n")
            # ffmpeg_extract_subclip(f"{source_path}/{video}", 
            #                     int(action_starts[index] * FRAME), 
            #                     int(action_ends[index] * FRAME), 
            #                     targetname=f"{save_path}/{video[:-4]}.mp4")




cut_videos("./source/val/stealing", "./cvs/val/stealing")
cut_videos("./source/val/others", "./cvs/val/others")
cut_videos("./source/train/stealing", "./cvs/train/stealing")
cut_videos("./source/train/others", "./cvs/train/others")


# if not path.exists("./cvs/val/stealing"): makedirs("./cvs/val/stealing")
# if not path.exists("./cvs/val/others"): makedirs("./cvs/val/others")

# path1 = "./CVS/val/stealing/"
# path2 = "./CVS/val/others/"
# videos1= [f for f in listdir(path1) if isfile(join(path1, f)) and ".MP4" in f]
# videos2= [f for f in listdir(path2) if isfile(join(path2, f)) and ".MP4" in f]

# for video1 in videos1:
#     curr_xml = open(f"{path1}/{video1[:-4]}.XML", "r").read().rstrip()
#     dict_type = xmltodict.parse(curr_xml)

#     action_starts = []
#     action_ends = []
#     for track in dict_type["annotations"]["track"]:
#         if "start" in track["@label"]:
#             action_starts.append(int(track["box"][0]["@frame"]))
#         if "end" in track["@label"]:
#             action_ends.append(int(track["box"][0]["@frame"]))


#     for index in range(len(action_starts)):
#         ffmpeg_extract_subclip("C_3_12_1_BU_DYA_07-31_16-15-01_CA_RGB_DF2_M1.mp4", 
#                             int(action_starts[index] * FRAME), 
#                             int(action_ends[index] * FRAME), 
#                             targetname=f"C_3_12_1_BU_DYA_07-31_16-15-01_CA_RGB_DF2_M1_{index}.mp4")
        
