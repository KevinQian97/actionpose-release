import os
import json
import shutil
import csv
from multiprocessing import Pool

data_folder = "/data/diva/kf1_vod2sfix/kf1_tst/prop_gen/images/validation"
out_data_folder = "/data/yijunq/kf1_test_TSM_prop"
prop_path = "/data/diva/kf1_vod2sfix/kf1_tst/prop_gen/props"
f = open("/data/yijunq/MEVA_TSM_LABEL/test_videofolder.txt","w")

def get_prop_move(vid,prop_id):
    json_file = json.load(open(os.path.join(prop_path,vid,"annotation",os.path.splitext(vid)[0],"actv_id_type.json"),"r"))
    if prop_id not in list(json_file.keys()):
        print(prop_id)
        print(vid)
        raise RuntimeError("prop not in json_file")
    start_frame = json_file[prop_id]["start_frame"]
    return int(start_frame)



def relocate_vids(vid):
    print(vid)
    vid_path = os.path.join(data_folder,vid)
    frames = os.listdir(vid_path)
    new_vid_name = vid.split("_")[0]+".avi"
    prop_id = vid.split("_")[1]

    if len(frames)<10:
        print("frames less than 10")
        return
    if len(frames)<90:
        frames.sort()
        mov = get_prop_move(new_vid_name,prop_id)
        start = str(mov)
        end = str(mov+len(frames))
        split_name = start+"_"+end+"_"+prop_id
        # if not os.path.exists(os.path.join(out_data_folder,new_vid_name,split_name)):
        #     os.makedirs(os.path.join(out_data_folder,new_vid_name,split_name))
        f.write(os.path.join(new_vid_name,split_name)+" "+str(len(frames))+" 0"+"\n")
        # for i in range(len(frames)):
        #     frame = frames[i]
        #     new_frame = "img_"+str(i+1).zfill(5)+".jpg"
        #     # new_frame = "img_"+frame.split("_")[1]
        #     shutil.copy(os.path.join(data_folder,vid,frame),os.path.join(out_data_folder,new_vid_name,split_name,new_frame))

    else:
        frames.sort()
        print(frames)
        epocs = len(frames) // 90
        mov = get_prop_move(new_vid_name,prop_id)
        for epoc in range(epocs):
            if len(frames)-epoc*90<30:
                break
            start = str(epoc*90+mov)
            end = str(epoc*90+mov+min(90,len(frames)-epoc*90))
            split_name = start+"_"+end+"_"+prop_id
            # if not os.path.exists(os.path.join(out_data_folder,new_vid_name,split_name)):
            #     os.makedirs(os.path.join(out_data_folder,new_vid_name,split_name))
            f.write(os.path.join(new_vid_name,split_name)+" "+str(min(90,len(frames)-epoc*90))+" 0"+"\n")
            # for i in range(epoc*90,min(len(frames),(epoc+1)*90)):
            #     frame = frames[i]
            #     new_frame = "img_"+str(i-epoc*90+1).zfill(5)+".jpg"
            #     # new_frame = "img_"+frame.split("_")[1]
            #     shutil.copy(os.path.join(data_folder,vid,frame),os.path.join(out_data_folder,new_vid_name,split_name,new_frame))
    return

def main():
    # n_jobs=48
    # pool = Pool(n_jobs)
    vids = os.listdir(data_folder)
    # pool.map(relocate_vids, vids)
    # pool.close()
    for vid in vids:
        relocate_vids(vid)

main()