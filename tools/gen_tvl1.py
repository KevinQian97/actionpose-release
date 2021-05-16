import os
from multiprocessing import Pool

RGB_path = "/data/UCF-101"
FLOW_path = "/data/UCF-101-Flow"
if not os.path.exists(FLOW_path):
    os.makedirs(FLOW_path)
folders = os.listdir(RGB_path)
def gen_optical(arg):
    vid,gpu_id = arg
    vid_name = vid.split("/")[-1].strip(".avi")
    directory = vid.split("/")[-2]
    print("Now processing VID:{}".format(vid_name))
    out = os.path.join(FLOW_path,directory,vid_name)
    if not os.path.exists(out):
        os.makedirs(out)
    call = "extract_gpu -f {} -x {} -y {} -i /dev/null -b 20 -t 1 -s 1 -d {} -w 0 -h 0 -o zip".format(vid,os.path.join(out,"flow_x"),os.path.join(out,"flow_y"),gpu_id)
    os.system(call)
    return
args = []
for folder in folders:
    vids = os.listdir(os.path.join(RGB_path,folder))
    for index,vid in enumerate(vids):
        args.append([os.path.join(RGB_path,folder,vid),index%4])

n_jobs = 40
pool = Pool(n_jobs)
print("Totally tackle {} videos".format(len(args)))
pool.map(gen_optical, args)
pool.close()


    

