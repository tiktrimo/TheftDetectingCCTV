from os import listdir
from os.path import isfile, join

path1 = "./cvs/val/"
path2 = "./cvs/train/"

videos1= [f for f in listdir(path1) if isfile(join(path1, f))]
videos2= [f for f in listdir(path2) if isfile(join(path2, f))]

videos = videos1 + videos2

# True if there is no duplicates
print(len(videos) == len(set(videos)))
