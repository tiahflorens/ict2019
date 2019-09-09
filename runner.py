import os
import platform

PUMA = 'puma'
OBAMA = 'obama'

if platform.node() == PUMA:  #
    EXE = '/home/peter/.conda/envs/py36/bin/python'
    VIDEOPATH = '/home/peter/dataset/gist'
    PATH_OUTPUT_ROOT = '/home/peter/dataset/alpha_gist'


else:
    EXE = '/home/peter/anaconda3/envs/py36/bin/python'
    VIDEOPATH = '/home/peter/extra/dataset/gist'
    PATH_OUTPUT_ROOT = '/home/peter/extra/dataset/alpha_gist'

PY = 'video_demo.py'

OUTPATH = os.path.join(VIDEOPATH, 'output')
DATA = 'gist/'

if not os.path.exists(PATH_OUTPUT_ROOT):
    os.mkdir(PATH_OUTPUT_ROOT)

BASE = '/home/peter/dataset/gist/org/mid2019'
SUBDIR = ['gta_dh_trial_2/trim_ohryong1.avi', 'gta_jh_trial_1/trim_ohryong1.avi', 'gta_jh_trial_2/trim_ohryong1.avi',
          'gta_jh_trial_1/trim_student1.avi', 'roaming_kym_trial1/student4.avi']

OUTDIR = '/home/peter/tmp/bbox_test'
for dd in SUBDIR:
    invideo = os.path.join(BASE, dd)
    print(invideo, os.path.exists(invideo))
    COMMAND = f'{EXE} {PY} --video {invideo} --outdir {OUTDIR} --save_video'
    # print(COMMAND)
    os.system(COMMAND)

    DIR_NAME = dd.split('/')[0]
    video_name = os.path.basename(invideo)
    C2 = f'mv {OUTDIR}/AlphaPose_{video_name} {OUTDIR}/{DIR_NAME}_{video_name}'
    # print(C2)
    os.system(C2)
    # quit()
