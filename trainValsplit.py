import os

train = open('filelists/train.txt', 'w')
test = open('filelists/val.txt', 'w')

base = '../common_dataset/'

folders = ['Preaching_Face_detected']

for folder in folders:
    videos = os.listdir(os.path.join(base, folder))
    for i, video in enumerate(videos):
        # images = [x for x in os.listdir(os.path.join(base, folder,video)) if '.jpg' in x]

        # if len(images) > 5:
        thingToWrite = os.path.join(folder, video) + '\n'
        if i % 10 == 0:
            test.write(thingToWrite)
        else:
            train.write(thingToWrite)

train.close()
test.close()