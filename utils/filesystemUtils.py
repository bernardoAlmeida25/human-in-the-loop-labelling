import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
import glob
from excpetions.NoMatchingFilesException import NoMatchingFilesException
import os


def show_select_folder_dialog():
    root = tk.Tk()
    root.withdraw()
    mb.showwarning(title='Select Directory', message='Select the same directory as previous selected in LabelImg')
    directory = fd.askdirectory()
    return directory


def validate_directory(directory, extension):
    path = directory + '/' + '*.' + extension
    if len(glob.glob(path)) == 0:
        raise NoMatchingFilesException(path=path, extension=extension)
    else:
        return path


def create_dest_folders():
    dest_directory = os.path.join(os.path.dirname(os.getcwd()), "mask_detection")
    if os.path.exists(dest_directory):
        os.chdir(dest_directory)
        os.mkdir("mask")
        os.mkdir("no_mask")
    else:
        os.mkdir(dest_directory)
        os.chdir(dest_directory)
        os.mkdir("mask")
        os.mkdir("no_mask")
    return dest_directory
