"""
download models to tune InterTrack
"""
import os
import os.path as osp

from huggingface_hub import hf_hub_download

# Download HDM checkpoints
ckpt_files = [
    'stage1.pth',
    'stage2.pth',
    'stage1-chair.pth',
    'stage2-chair.pth',
    'corrAE.pth'
]
exp_names = [osp.splitext(x)[0] for x in ckpt_files]
# Object pose prediction network
ckpt_files.append('TOPNet-5obj.pth')
exp_names.append('so3smpl_5obj-none-dtune-acc0.2')
ckpt_files.append('TOPNet-small-objs.pth')
exp_names.append('so3smpl_small-none-dtune-acc0.8')

for file, exp in zip(ckpt_files, exp_names):
    ckpt_file = hf_hub_download("xiexh20/HDM-models", file)
    outdir = f'outputs/{exp}/single'
    os.makedirs(outdir, exist_ok=True)
    os.system(f'ln -s {ckpt_file} {outdir}/checkpoint-latest.pth')
