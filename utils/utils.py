import os
from random import randint


def rename_files():
    path = '/Users/bernardoalmeida/Documents/Tese/mask_detection/yes'
    files = os.listdir(path)
    for index, file in enumerate(files):
        r = randint(0, 999999)
        name = str(r) + ".jpg"
        os.rename(os.path.join(path, file), os.path.join(path, name))