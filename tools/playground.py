import json
import os

# org_js = json.load(open("/data/yijunq/datasets/STR/str_all_anno_virat.json","r"))
# jname = "/data/yijunq/datasets/STR/str_all_anno_viratdong.json"
# new_dict = {"labels":[],"database":{}}
# new_dict["labels"] = org_js["labels"]
# base_vid_path = "/data/yijunq/datasets/STR/videos"
# for k,v in org_js["database"].items():
#     cat = k.split("/")[-2]
#     vid = k.split("/")[-1]
#     combk = os.path.join(base_vid_path,cat,vid)
#     new_dict["database"][combk] = v

# json_str = json.dumps(new_dict,indent=4)
# with open(jname, 'w') as save_json:
#     save_json.write(json_str)  

# path = "/data/yijunq/datasets/VIRAT/videos"
# vids = os.listdir(path)
# for vid in vids:
#     if vid == "STR":
#         continue
#     src = os.path.join(path,vid)
#     new_vid = ".".join()

path = "/home/kevinq/datasets/STR"
js = json.load(open(os.path.join(path,"str_trn_anno.json"),"r"))
jname = "/home/kevinq/datasets/STR/str_trn_stab_anno.json"
new_dict = {}
new_dict["labels"]= js["labels"]
new_dict["database"] = {}
for k,v in js["database"].items():
    comps = k.split("/")
    comps[2] = "share"
    comps[3] = "datasets"
    comps[4] = "PIP175k"
    comps[5] = "pip_175k_stabilized"
    new_k = "/".join(comps)
    new_dict["database"][new_k] = v

json_str = json.dumps(new_dict,indent=4)
with open(jname, 'w') as save_json:
    save_json.write(json_str) 