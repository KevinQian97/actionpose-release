import os
import json
import numpy as np
js_rgb = json.load(open("../pred_rgb.json","r"))
js_joints = json.load(open("../pred_joints.json","r"))
js_fuse = json.load(open("../pred_fuse.json","r"))


error_dict = {}
for k,v in js_joints.items():
    name = k.split("/")[-1].strip("_rgb.avi")
    gt = int(v["gt"])
    pred = int(np.argmax(np.asarray(v["pred"])))
    if name in error_dict:
        error_dict[name]["joints"] = pred
    elif pred!=gt:
        error_dict[name] = {"gt":gt,"joints":pred}

for k,v in js_rgb.items():
    name = k.split("/")[-1].strip("_rgb.avi")
    gt = v["gt"]
    pred = int(np.argmax(np.asarray(v["pred"])))
    if name in error_dict:
        error_dict[name]["rgb"] = pred
    elif pred!=gt:
        error_dict[name] = {"gt":gt,"rgb":pred}

for k,v in js_fuse.items():
    name = k.split("/")[-1].strip("_rgb.avi")
    gt = v["gt"]
    pred = int(np.argmax(np.asarray(v["pred"])))
    if name in error_dict:
        error_dict[name]["fusion"] = pred
    elif pred!=gt:
        error_dict[name] = {"gt":gt,"fusion":pred}

json_str = json.dumps(error_dict,indent=4)
with open("./error.json", 'w') as save_json:
    save_json.write(json_str) 