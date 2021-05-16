import pickle
import json
import os
import random
import decord
from decord import VideoReader
from decord import cpu, gpu
base_path = "/mnt/data/ntu/nturgb+d_rgb"
out_path = "/home/kevinq/datasets/ntu"
vid_path = base_path

def gen_splits(trn_rate=0.8):
    trn_list = []
    tst_list = []
    labels = {}
    vids = os.listdir(vid_path)
    print("Totally {} videos".format(len(vids)))
    for vid in vids:
        label = int(vid.strip("_rgb.avi").split("A")[-1])
        if label not in labels:
            labels[label] = [vid]
        else:
            labels[label].append(vid)
    for k,v in labels.items():
        loc = int(trn_rate*len(v))
        for i in range(len(v)):
            if i <= loc:
                trn_list.append(v[i])
            else:
                tst_list.append(v[i])
    print("Totally {} vids for training".format(len(trn_list)))
    print("Totally {} vids for testing".format(len(tst_list)))
    return trn_list,tst_list,list(labels.keys())


def main():
    jname_trn = os.path.join(out_path,"str_trn_anno.json")
    jname_tst = os.path.join(out_path,"str_tst_anno.json")
    error_file = open("./errors.txt","w")
    error_list = []
    trn_list,tst_list,label_list = gen_splits()
    new_dict = {}
    new_dict["labels"] = []
    for cat in label_list:
        new_dict["labels"].append(cat)
    new_dict["database"] = {}
    for loc in trn_list:
        loc = os.path.join(vid_path,loc)
        try:
            vr = VideoReader(loc)
        except:
            print("{} loading error".format(loc))
            error_list.append(loc)
            error_file.write(loc+"\n")
            continue
        num_frames = len(vr)
        label = new_dict["labels"].index(int(loc.strip("_rgb.avi").split("A")[-1]))
        act_dict = {"annotations":{"label":label,"num_frames":num_frames,"start":0,"end":num_frames-1}}
        new_dict["database"][loc] = act_dict
    json_str = json.dumps(new_dict,indent=4)
    with open(jname_trn, 'w') as save_json:
        save_json.write(json_str)    
        
    new_dict = {}
    new_dict["labels"] = []
    for cat in label_list:
        new_dict["labels"].append(cat)
    new_dict["database"] = {}
    for loc in tst_list:
        loc = os.path.join(vid_path,loc)
        try:
            vr = VideoReader(loc)
        except:
            print("{} loading error".format(loc))
            error_list.append(loc)
            error_file.write(loc+"\n")
            continue
        num_frames = len(vr)
        label = new_dict["labels"].index(int(loc.strip("_rgb.avi").split("A")[-1]))
        act_dict = {"annotations":{"label":label,"num_frames":num_frames,"start":0,"end":num_frames-1}}
        new_dict["database"][loc] = act_dict
    json_str = json.dumps(new_dict,indent=4)
    with open(jname_tst, 'w') as save_json:
        save_json.write(json_str) 

main()
        