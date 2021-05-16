import numpy as np
import torch
import json
import os
from ops.merger import OverlapCubeMerger
from ops.cube import ActivityTypeMEVA,CubeActivities,ActivityTypeVIRAT
import decord
from decord import VideoReader

def get_obj_types(event):
    toks = event.split("_")
    if "person" in toks and "vehicle" in toks :
        return ["Vehicle","Person"]
    elif "person" in toks:
        return ["Person"]
    elif "vehicle" in toks:
        return ["Vehicle"]
    else:
        print (event)
        raise RuntimeError("unknown events")

def get_file_index(filesProcessed):
    new_dict = {}
    for f in filesProcessed:
        new_dict[f]={"framerate": 30.0, "selected": {"0": 1, "9000": 0}}
    return new_dict

def get_activity_index(activities):
    new_dict = {}
    for act in activities:
        new_dict[act] = {"objectTypes":get_obj_types(act)}
    return new_dict

def getoutput(vid_names,video_pred_topall,video_prob_topall,event_dict,args,pred_frames = 64,filter_rate = 0.4):
    new_dict = {}
    for name, pred_all, prob_all in zip(vid_names, video_pred_topall,video_prob_topall):
        if args.dataset in ["MASK","DET","MEVA"]:
            video = name.split("_")[0]
        else:
            video = "_".join(name.split("_")[:-1])
        prop = name.split("_")[-1]
        if video not in list(new_dict.keys()):
            new_dict[video] = {}
            for event in event_dict:
                new_dict[video][event] = []
            new_dict[video]["props"] = []
        new_dict[video]["props"].append(prop)
        for i in range(len(pred_all)):
            pred = pred_all[i]
            event = event_dict[int(pred)]
            prob = prob_all[i]
            new_dict[video][event].append(str(prob))
    videos = list(new_dict.keys())
    database = json.load(open(args.val_list,"r"))["database"]
    props = os.listdir(os.path.join(args.prop_path,args.tst_name))
    merger = OverlapCubeMerger()
    activities = []
    files = []
    for prop in props:
        vid_name = prop.strip(".csv")
        if vid_name+args.vid_type not in files:
            files.append(vid_name+args.vid_type )
    for vid_name in videos:
        cube_tensor = torch.zeros(len(new_dict[vid_name]["props"])*len(event_dict),9)
        for i in range(len(new_dict[vid_name]["props"])):
            prop_name = vid_name+"_"+new_dict[vid_name]["props"][i]
            t0 = float(database[prop_name]["annotations"]["start"])
            t1 = float(database[prop_name]["annotations"]["end"])
            tid = int(database[prop_name]["tid"])
            x0,y0,x1,y1 = database[prop_name]["annotations"]["bbox"]
            for j in range(len(event_dict)):
                score = float(new_dict[vid_name][event_dict[j]][i])
                cube_tensor[i*len(event_dict)+j] = torch.tensor([tid,j+1,score,t0,t1,x0,y0,x1,y1])
        if args.dataset in ["MEVA","MASK","DET"]:
            cube_acts = CubeActivities(cube_tensor, vid_name+args.vid_type , ActivityTypeMEVA)
        else:
            cube_acts = CubeActivities(cube_tensor, vid_name+args.vid_type , ActivityTypeVIRAT)
        filtered_acts = merger(cube_acts)
        acts = filtered_acts.to_official()
        for act in acts:
            act["activityID"] = event_dict.index(act["activity"])
            activities.append(act)
    res_dict = {"filesProcessed":files,"activities":[]}
    file_dict = get_file_index(res_dict["filesProcessed"])
    eve_dict = get_activity_index(event_dict)
    sum_frame = 0
    fnames = file_dict.keys()
    for vid in fnames:
        vr = VideoReader(os.path.join(args.root_path,vid))
        sum_frame+=len(vr)
    loc = int(sum_frame/pred_frames*filter_rate)
    scores = [0]*len(event_dict)
    rank_lst = [[]]*len(event_dict)
    for act in activities:
        rank_lst[act["activityID"]].append(act["presenceConf"])
    for i in range(len(event_dict)):
        rank_lst[i].sort()
        if loc >=len(rank_lst[i]):
            loci = len(rank_lst[i])-1
        else:
            loci = loc
        scores[i] = rank_lst[i][-loci]
    
    for act in activities:
        if act["presenceConf"]>= scores[act["activityID"]]:
            res_dict["activities"].append(act)

    return res_dict,file_dict,eve_dict

def get_eventdict(args):
    if args.dataset in ["DET","MASK","MEVA"]:
        f = open("./labels_det.txt","r")
    elif args.dataset in ["VIRAT"]:
        f = open("./labels_virat.txt","r")
    events = f.readlines()
    event_dict = []
    for event in events:
        event_dict.append(event.strip())
    print(event_dict)
    return event_dict

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_nll(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_bce(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # print(output.size())
    # print(target.size())
    inner_target = []
    inner_output = []
    batch_size = target.size(0)
    for i in range(batch_size):
        if torch.max(target[i])>0:
            inner_output.append(output[i])
            inner_target.append(target[i])
    if len(inner_target)==0:
        return [torch.tensor(-1).cuda(),torch.tensor(-1).cuda()],0
    inner_output = torch.stack(inner_output)
    inner_target = torch.stack(inner_target)
    inner_target = torch.argmax(inner_target,dim=1)
    

    _, pred = inner_output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(inner_target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, len(inner_target)