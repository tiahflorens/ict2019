import os
import platform

PUMA = 'puma'
OBAMA = 'obama'


if platform.node() == PUMA: #
    EXE = '/home/peter/.conda/envs/alpha.torch/bin/python'
    VIDEOPATH = '/home/peter/dataset/gist'
    PATH_OUTPUT_ROOT = '/home/peter/dataset/alpha_gist'


else:
    EXE = '/home/peter/anaconda3/envs/alpha/bin/python'
    VIDEOPATH = '/home/peter/extra/dataset/gist'
    PATH_OUTPUT_ROOT = '/home/peter/extra/dataset/alpha_gist'




PY = 'video_demo.py'

OUTPATH = os.path.join(VIDEOPATH, 'output')
DATA = 'gist/'

if not os.path.exists(PATH_OUTPUT_ROOT):
    os.mkdir(PATH_OUTPUT_ROOT)

for r, d, f in os.walk(VIDEOPATH):
    # print(r,d,f)
    for dd in d:
        subdir = r.split(DATA)
        if len(subdir) == 1:
            tmp_path = os.path.join(PATH_OUTPUT_ROOT, dd)
        else:
            tmp_path = os.path.join(PATH_OUTPUT_ROOT, r.split(DATA)[1], dd)

        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

    for ff in f:
        in_video = os.path.join(r, ff)
        out_dir = os.path.join(PATH_OUTPUT_ROOT, r.split(DATA)[1])

        if ff.split('.')[1] in ['avi', 'mp4', 'asf', 'webm', 'swp']:
            print(f'{in_video} >> {out_dir}')
            COMMAND = f'{EXE} {PY} --video {in_video} --outdir {out_dir} --save_video'
            os.system(COMMAND)
