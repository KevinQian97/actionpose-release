import torch.utils.data as data
from avi_r import AVIReader
from PIL import Image
import os
import torch
import numpy as np
from numpy.random import randint
import cv2
import json
import decord
from decord import VideoReader
from decord import cpu, gpu
import torchvision
import gc
import math
decord.bridge.set_bridge('torch')
from torchvision.io import read_video,read_video_timestamps
from tools.load_skel import read_xyz
class VideoRecord_NTU(object):
    def __init__(self, dic, name, root_path, enlarge_rate=0.):
        self._data = dic
        self._path = name
        self._prop = name.split("/")[-1].strip("_rgb.avi")
        self._root_path = root_path
        self.enlarge_rate = enlarge_rate

    def frames(self):
        vr = VideoReader(self._path)
        trans = torchvision.transforms.ToPILImage(mode='RGB')
        images = []
        for idx in range(len(vr)):
            images.append(trans(vr[idx].permute(2,0,1)).convert('RGB'))
        return images

    def joints(self):
        path = os.path.join(self._root_path,self._prop+".skeleton")
        return read_xyz(path)

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._prop

    @property
    def num_frames(self):
        return self._data["annotations"]["num_frames"]

    @property
    def label(self):
        return self._data["annotations"]["label"]

    @property
    def start(self):
        return 0


class VideoRecord_PIP(object):
    def __init__(self, dic, name, root_path, enlarge_rate=0.):
        self._data = dic
        self._path = name
        self._prop = name
        self._root_path = root_path
        self.enlarge_rate = enlarge_rate

    def frames(self):
        vr = VideoReader(self._path)
        trans = torchvision.transforms.ToPILImage(mode='RGB')
        images = []
        for idx in range(len(vr)):
            images.append(trans(vr[idx].permute(2,0,1)).convert('RGB'))
        return images

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._prop

    @property
    def num_frames(self):
        return self._data["num_frames"]

    @property
    def label(self):
        return self._data["label"]

    @property
    def start(self):
        return 0