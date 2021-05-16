import os
import json
import shutil
import csv
from multiprocessing import Pool

data_folder = "/data/yijunq/MEVA_prop"
annot_path = "/data/yijunq/MEVA_kf1_ibm_umd_neg_split_with_labels.json"
out_data_folder = "/data/yijunq/MEVA_TSM_PROP"
out_annot_path = "/data/yijunq/MEVA_TSM_LABEL"
json_file = json.load(open(annot_path,"r"))
event_dict = json_file["labels"]
print(len(event_dict))
print(event_dict)
database = json_file["database"]
vids = list(database.keys())
print(len(vids))
train_csv = open(os.path.join(out_annot_path,"train.csv"),"w")
train_writer = csv.writer(train_csv)
val_csv = open(os.path.join(out_annot_path,"validation.csv"),"w")
val_writer = csv.writer(val_csv)
train_writer.writerow(["label","youtube_id","time_start","time_end","split"])
val_writer.writerow(["label","youtube_id","time_start","time_end","split"])





def relocate_vids(vid):
    print(vid)
    vid_path = os.path.join(data_folder,vid,vid)
    frames = os.listdir(vid_path)
    label = database[vid]["annotations"]["label"]
    subset = database[vid]["subset"]
    if len(frames)<10:
        print("frames less than 10")
        return
    if len(frames)<90:
        frames.sort()
        if not os.path.exists(os.path.join(out_data_folder,label,vid+"_1")):
            os.makedirs(os.path.join(out_data_folder,label,vid+"_1"))
        if subset == "training":
            train_writer.writerow([label,vid,1,len(frames),"train"])
        elif subset == "validation":
            val_writer.writerow([label,vid,1,len(frames),"validate"])
        else:
            print(subset)
            raise RuntimeError("find error subset")
        # for i in range(len(frames)):
        #     frame = frames[i]
        #     new_frame = "img_"+str(i+1).zfill(5)+".jpg"
        #     # new_frame = "img_"+frame.split("_")[1]
        #     shutil.copy(os.path.join(data_folder,vid,vid,frame),os.path.join(out_data_folder,label,vid+"_1",new_frame))


    else:
        frames.sort()
        print(frames)
        epocs = len(frames) // 90
        for epoc in range(epocs):
            if len(frames)-epoc*90<30:
                break
            if subset == "training":
                train_writer.writerow([label,vid,epoc*90+1,min(90,len(frames)-epoc*90),"train"])
            elif subset == "validation":
                val_writer.writerow([label,vid,epoc*90+1,min(90,len(frames)-epoc*90),"validate"])
            else:
                print(subset)
                raise RuntimeError("find error subset")
            # if not os.path.exists(os.path.join(out_data_folder,label,vid+"_"+str(epoc*90+1))):
            #     os.makedirs(os.path.join(out_data_folder,label,vid+"_"+str(epoc*90+1)))
            # for i in range(epoc*90,min(len(frames),(epoc+1)*90)):
            #     frame = frames[i]
            #     new_frame = "img_"+str(i-epoc*90+1).zfill(5)+".jpg"
            #     # new_frame = "img_"+frame.split("_")[1]
            #     shutil.copy(os.path.join(data_folder,vid,vid,frame),os.path.join(out_data_folder,label,vid+"_"+str(epoc*90+1),new_frame))
    return


def make_cat_folders(out_data_folder,event_dict):
    for event in event_dict:
        if not os.path.exists(os.path.join(out_data_folder,event)):
            os.makedirs(os.path.join(out_data_folder,event))
    return



def main():
    # make_cat_folders(out_data_folder,event_dict)
    # n_jobs=48
    # pool = Pool(n_jobs)
    # pool.map(relocate_vids, vids)
    # pool.close()
    for vid in vids:
        relocate_vids(vid)


main()


    


