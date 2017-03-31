import pandas as pd
import os

from PIL import Image
from gesture import Gesture

#user_number = list(range(3, 8)) + [9, 10]
user_number = [3, 4]#, 5, 6]
def fetch_pixels():
    '''
    Returns a tuple of pixels(inputs) and output labels
    '''
    gestures = []
    for user in user_number:
        path_to_csv = 'Dataset/user_' + str(user) + '/user_' + str(user) + '_loc.csv'
        print(path_to_csv)

        path_to_images = 'Dataset/user_' + str(user) + '/'
        list_of_images = os.listdir(path_to_images)
        list_of_images = filter(lambda x: '.csv' not in x, list_of_images)

        for image in list_of_images:
            im = Image.open(path_to_images + image)
            pix = im.load()
            rows, cols = im.size

            pixels = list()
            for row in range(rows):
                for col in range(cols):
                    for p in pix[row, col]:
                        pixels.append(p/256)
            gesture = Gesture(image[0], pixels)
            gesture.generate_output()
            gestures.append(gesture)
    return gestures
