'''
Created on Dec 11, 2018

Transforms manually annotated pictures from AffectNet dataset to csv format by doing the follwing steps:
- Open mapping csv
- For each photo with a face:
    crop face
    resize to 48x48
    convert to grayscale
    write pixel values to result csv

@author: Vladimir Petkov
'''


import csv
import numpy as np
from os import getenv
from PIL import Image

AFFECT_NET_ROOT = getenv("AFFECT_NET_ROOT")
MAPPING_CSV = getenv("MAPPING_CSV")

def convert_to_feely_emotion_mapping(affect_net_code):
#  feely codes (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
#  affect net codes (0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face)
    affect_to_feely_codes = {0:6, 1:3, 2:4, 3:5, 4:2, 5:1, 6:0}
    return affect_to_feely_codes[affect_net_code]

def get_face_pixels(path, face_x, face_y, face_width, face_height):
    img = Image.open(path)
    area = (face_x, face_y, face_x + face_width, face_y + face_height)
    processed_img = img.crop(area).resize((48, 48)).convert('L')
    pixels = list(processed_img.getdata())
    return pixels

def main():
    mapping_file = open(MAPPING_CSV, 'rb')
    reader = csv.DictReader(mapping_file)
    rows = list(reader)
    total_rows = len(rows)

    result_file = open('result.csv', 'w')
    writer = csv.DictWriter(result_file, ['emotion', 'pixels'])
    writer.writeheader()
    i = 0
    for row in rows:
        try:
            i+=1
            if i%100 ==0:
                print i, "/", total_rows 
            emotion = int(row['expression'])
            if emotion > 6:
                continue # no face here
            image_path = row['subDirectory_filePath']
            face_width = int(row['face_width'])
            face_height = int(row['face_height'])
            face_x = int(row['face_x'])
            face_y = int(row['face_y'])
            pixels = get_face_pixels('/'.join([AFFECT_NET_ROOT, image_path]), face_x, face_y, face_width, face_height)
            emotion_code = str(convert_to_feely_emotion_mapping(emotion))
            pixels_str = ' '.join(map(str, pixels))
            writer.writerow({'emotion':emotion_code, 'pixels': pixels_str})
        except:
            print 'ERROR'
            continue
    
    mapping_file.close()
    result_file.close()

if __name__ == "__main__":
    main()
