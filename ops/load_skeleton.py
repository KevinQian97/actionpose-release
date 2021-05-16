import os
import sys
import pickle
import numpy as np
import torch

from ops.st_gcn import ST_GCN_18

max_body = 2
num_joint = 25
max_frame = 300

def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key,
                                    f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key,
                                        f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data


# if __name__ == '__main__':
#     # Build model
#     model = ST_GCN_18(in_channels=3, num_class=60, graph_cfg={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}, dropout=0.5)

#     # Skeleton file path which looks like '/mnt/hdd/shengcao/NTU/nturgb+d_skeletons/S001C001P001R001A001.skeleton'
#     root = '/mnt/hdd/shengcao/NTU'      # Change this to yours
#     video = 'S001C001P001R001A001'
#     file_path = os.path.join(root, 'nturgb+d_skeletons', '{}.skeleton'.format(video))

#     # Load skeleton data
#     data = read_xyz(file_path)
#     print(data.shape)                                   # 3 x T x V x M

#     # Forward pass
#     inputs = torch.FloatTensor(data).unsqueeze(0)
#     print(inputs.shape)                                 # B x 3 x T x V x M
#     output = model(inputs)
#     print(output.shape)                                 # B x 60

#     # If you want features instead of class scores
#     _, features = model.extract_feature(inputs)
#     print(features.shape)                               # B x 256 x T' x V x M  (Note T' < T)
