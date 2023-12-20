from os import listdir, replace, rmdir
from os.path import isfile, join

path1 = "./cvs/val/stealing/"
path2 = "./cvs/val/others/"

videos1= [f for f in listdir(path1) if isfile(join(path1, f))]
videos2= [f for f in listdir(path2) if isfile(join(path2, f))]

conf_file = open("./cvs/action_small_val_video.txt", "w")
for video1 in videos1:
    replace(f"{path1}{video1}",f"./cvs/val/{video1}")
    conf_file.write(f"{video1} 1\n")
for video2 in videos2:
    replace(f"{path2}{video2}", f"./cvs/val/{video2}")
    conf_file.write(f"{video2} 0\n")
conf_file.close()

rmdir(path1)
rmdir(path2)




path1 = "./cvs/train/stealing/"
path2 = "./cvs/train/others/"

videos1= [f for f in listdir(path1) if isfile(join(path1, f))]
videos2= [f for f in listdir(path2) if isfile(join(path2, f))]

conf_file = open("./cvs/action_small_train_video.txt", "w")
for video1 in videos1:
    replace(f"{path1}{video1}",f"./cvs/train/{video1}")
    conf_file.write(f"{video1} 1\n")
for video2 in videos2:
    replace(f"{path2}{video2}", f"./cvs/train/{video2}")
    conf_file.write(f"{video2} 0\n")
conf_file.close()

rmdir(path1)
rmdir(path2)



