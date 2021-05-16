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

decord.bridge.set_bridge('torch')

# modify the loader to video loader with json labels
class VideoRecord(object):
    def __init__(self, dic, name, root_path, enlarge_rate=0.1):
        self._start = dic["annotations"]["start"]
        self._bbox = dic["annotations"]["bbox"]
        self._num_frames = dic["annotations"]["end"]-dic["annotations"]["start"]
        self._conf = torch.Tensor([round(prob) for prob in dic["annotations"]["conf"]])
        self._path = name.split("_")[0]+".avi"
        self._prop = name
        self._root_path = root_path
        self._enlarge_rate = enlarge_rate
    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._prop

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def label(self):
        return self._conf
    
    @property
    def start(self):
        return self._start

    @property
    def bbox(self):
        return self._bbox
    

# modify the loader to video loader
# class VideoRecord(object):
#     def __init__(self, row):
#         self._data = row

#     @property
#     def path(self):
#         return self._data[0]+'.mp4'

#     @property
#     def num_frames(self):
#         return int(self._data[1])

#     @property
#     def label(self):
#         return int(self._data[2])

# original image loader
# class VideoRecord(object):
#     def __init__(self, row):
#         self._data = row

#     @property
#     def path(self):
#         return self._data[0]

#     @property
#     def num_frames(self):
#         return int(self._data[1])

#     @property
#     def label(self):
#         return int(self._data[2])
def get_frames(root_path,vid_name,start,num_frames,bbox,enlarge_rate):
        cap = AVIReader(os.path.join(root_path,vid_name))
        cap.seek(start)
        height = cap.height
        width = cap.width
        x0,y0,x1,y1 = bbox
        enlarge_x = (x1-x0)*enlarge_rate
        enlarge_y = (y1-y0)*enlarge_rate
        x0 = int(max(0,x0-enlarge_x))
        x1 = int(min(width,x1+enlarge_x))
        y0 = int(max(0,y0-enlarge_y))
        y1 = int(min(height,y1+enlarge_y))
        
        images = []
        for frame in cap.get_iter(num_frames):      
            images.append(torch.from_numpy(frame.numpy('rgb24')).float()[y0:y1,x0:x1])   
        cap.close()
        return images

class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False, all_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()


# also modify it to video loader
    def _load_image(self, frames, directory, idx, start):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                if idx >= len(frames):
                    idx = len(frames)-1
                return [frames[idx]]
            except Exception:
                print('error loading video:{} whose idx is {}'.format(os.path.join(self.root_path, directory),start+idx))
                return [frames[0]]
# parse_list for json input
    def _parse_list(self):
        tmp = json.load(open(self.list_file,"r"))["database"]
        items = tmp.keys()
        self.video_list = [VideoRecord(tmp[item],item,self.root_path) for item in items]
        print('video number:%d' % (len(self.video_list)))

# parse_list for txt input
    # def _parse_list(self):
    #     # check the frame number is large >3:
    #     tmp = [x.strip().split(' ') for x in open(self.list_file)]
    #     if not self.test_mode or self.remove_missing:
    #         tmp = [item for item in tmp if int(item[1]) >= 3]
    #     self.video_list = [VideoRecord(item) for item in tmp]

    #     if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
    #         for v in self.video_list:
    #             v._data[1] = int(v._data[1]) / 2
    #     print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments))
            return offsets

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)-1 for x in range(self.num_segments)])
            return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)


    def get(self, record, indices):
        # print(indices)
        images = list()
        frames = get_frames(record._root_path,record._path,record.start,record.num_frames,record.bbox,record._enlarge_rate)
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(frames,record.name,p,record.start)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        images = torch.stack(images,dim=0)
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


# if __name__ == "__main__":
#     path = "/home/lijun/datasets/meva/videos/2018-03-07.17-35-01.17-40-01.hospital.G436.avi"
#     frames = []
#     cap = AVIReader(path)
#     cap.seek(64)
#     x0,y0,x1,y1 = [1728,288,1920,673]
#     for frame in cap.get_iter(64):
#         frame = frame.numpy()[y0:y1,x0:x1]
#         frame = torchvision.transforms.ToPILImage(mode='RGB')(frame.transpose(2,0,1)).convert('RGB')
#         frames.append(frame)
