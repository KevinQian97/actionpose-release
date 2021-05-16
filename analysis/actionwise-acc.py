import json
import os
import numpy as np
import csv

js_rgb = json.load(open("../pred_rgb.json","r"))
js_joints = json.load(open("../pred_joints.json","r"))
js_fuse = json.load(open("../pred_fuse.json","r"))

acc_dict = {"rgb":[0]*60,"joints":[0]*60,"fuse":[0]*60}
count = [0]*60
for k,v in js_rgb.items():
    gt = v["gt"]
    pred = np.argmax(np.asarray(v["pred"]))
    if pred==gt:
        acc_dict["rgb"][gt]+=1
    count[gt]+=1
for i in range(60):
    acc_dict["rgb"][i]/=count[i]

count = [0]*60
for k,v in js_joints.items():
    gt = v["gt"]
    pred = np.argmax(np.asarray(v["pred"]))
    if pred==gt:
        acc_dict["joints"][gt]+=1
    count[gt]+=1
for i in range(60):
    acc_dict["joints"][i]/=count[i]

count = [0]*60
for k,v in js_fuse.items():
    gt = v["gt"]
    pred = np.argmax(np.asarray(v["pred"]))
    if pred==gt:
        acc_dict["fuse"][gt]+=1
    count[gt]+=1
for i in range(60):
    acc_dict["fuse"][i]/=count[i]

with open("./actionwise.csv",'w') as f:
    csv_write = csv.writer(f)
    csv_write.writerow(acc_dict["rgb"])
    csv_write.writerow(acc_dict["joints"])
    csv_write.writerow(acc_dict["fuse"])


print(acc_dict["rgb"])
print(acc_dict["joints"])
print(acc_dict["fuse"])