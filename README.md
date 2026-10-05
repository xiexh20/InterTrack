# InterTrack (3DV'25)
Official implementation for 3DV'25 paper: InterTrack: Tracking Human Object Interaction without Object Templates
[Project Page](https://virtualhumans.mpi-inf.mpg.de/InterTrack/) | [ProciGen-video Dataset](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.B6BM5R) | [ArXiv](https://arxiv.org/abs/2408.13953) 


<p align="left">
<img src="./configs/procigen-video.gif" alt="teaser" width="80%"/>
</p>

### Previous works: 
- Template-free single frame reconstruction: [HDM](https://github.com/xiexh20/HDM).
- Template based: [CHORE](https://github.com/xiexh20/CHORE), [VisTracker](https://github.com/xiexh20/VisTracker). 

## Contents 
1. [Dependencies](#dependencies)
2. [Quick start](#quick-start)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)
7. [License](#license)

### TODO List
- [x] Demo code. 
- [x] Human registration as standard alone repo, see [fast-human-reg](https://github.com/xiexh20/fast-human-reg).
- [ ] Training. 
- [ ] Full BEHAVE evaluation. 

### Updates
- March 21, 2025, code released, hello world!  

## Dependencies
The code is tested on `torch=2.4.1+cu121, cuda12.1, debian11`. In general it should work with torch + cuda 12.1.
We recommend using anaconda environment:
```shell
conda create -n intertrack python=3.10
conda activate intertrack 
```
Required packages can be installed by:
```shell
pip install -r pre-requirements.txt # Install pytorch and others
pip install -r requirements.txt     # Install pytorch3d from source
```

**SMPL body models**: We use SMPL-H (mano_v1.2) from [this website](https://mano.is.tue.mpg.de/download.php). 
Download and unzip to a local path and modify in `SMPL_MODEL_ROOT` in `lib_smpl/const.py`.
To use the [smplfitter](https://github.com/isarandi/smplfitter), we also need a `kid_template.npy` file, see [this doc](https://github.com/isarandi/smplfitter?tab=readme-ov-file#download-body-model-files).
To run our code, login to the AGORA project path, download the `smpl_kid_template.npy` via SMIL/SMIL-X template->SMIL(SMPL formate), and rename it to `kid_template.npy`. In the end, the file structure should be this:
```shell
SMPL_MODEL_ROOT
|--kid_template.npy
|--SMPLH_FEMALE.pkl
|--SMPLH_MALE.pkl
```


## Quick start
### Download checkpoints
```shell
python download_models.py
```

### Download demo data
We prepare two example sequences for quick start, one is captured by mobile phone and the other is from BEHAVE dataset. 
Download the packed file from [Edmond](https://edmond.mpg.de/file.xhtml?fileId=310951&version=3.0) and then do `unzip InterTrack-demo-data.zip -d demo-data `. 
Once downloaded, update the values of these paths: (use absolute path)
- The path to `demo-data` in `dataset.demo_data_path`, i.e. [this line](https://github.com/xiexh20/InterTrack/blob/main/configs/structured.py#L267).
- The path to packed data in `dataset.behave_packed_dir`, i.e. [this line](https://github.com/xiexh20/InterTrack/blob/main/configs/structured.py#L266), modify this to `demo-data/packed`.
- The path to SMPL assets, i.e. set `SMPL_ASSETS_ROOT` to `demo-data/assets` in `lib_smpl/const.py`.
  

### Run demo
```shell
# Run InterTrack on mobile phone sequence 
bash scripts/demo_phone.sh

# Run InterTrack on one behave sequence
bash scripts/demo_behave.sh
```
After running InterTrack on the behave sequence, you can evaluate the results with:
```shell
python eval/eval_separate.py -pr outputs/corrAE/single/opt-hoi-orighdm/pred -gt outputs/stage2/single/demo-stage2/gt -split configs/splits/demo-table.json
```
You should see some numbers like this: `All 679 images: hum_F-score@0.01m=0.3983  obj_F-score@0.01m=0.6754  H+O_F-score@0.01m=0.5647  CD=0.0257`

To run test on more BEHAVE sequences, you will need to download [this packed file](https://edmond.mpg.de/file.xhtml?fileId=310881&version=3.0) and update `behave_packed_dir` in `configs/structured.py` file. And prepare similar split files as `configs/splits/demo-seq-table-15fps.pkl` (for HDM recon. and optimization), and `configs/splits/demo-seq-table-15fps-video.pkl` (for object pose prediction). 
## Training
Coming soon... 

## Evaluation
Some notes regarding reproducing the results in paper:
- GT translation was used to run HDM (first stage in InterTrack). This is the same protocol as prior works PC2 and HDM. Because single view reconstruction has the inherent depth-scale ambiguity, with the unordered point cloud as output, it is difficult to align it with GT points to compute meaningful numbers. So we simply follow previous works to use GT translation, removing at least the depth ambiguity and focus on shape evaluation. If your work does not use GT translation, there are two possible ways to setup InterTrack for a potentially more fair comparison: Run InterTrack with estimated 2d translation, following [this script](https://github.com/xiexh20/InterTrack/blob/main/scripts/demo_phone.sh), and then you can do either 1). align with GT using the SMPL vertices, or 2). use pure ICP. 
- Pose estimation: we provide two pretrained checkpoints for object pose (step 4), see [download models](https://github.com/xiexh20/InterTrack/blob/main/download_models.py#L19-L22), the `TOPNet-5obj.pth` is for BEHAVE and InterCap chairs, tables and monitor, all other objects should use the other checkpoint `TOPNet-small-objs.pth`. 

Run test on the full behave dataset: coming soon...

### Evaluation without GT translation (InterCap, 21 sequences)
`eval/eval_rot_chamf.py` evaluates results obtained with estimated 2D translation (see `scripts/demo_phone.sh`). Predictions are brought back to metric camera space and aligned to GT with one similarity transform computed from the SMPL vertices of the first frame of each sequence. It reports Chamfer distance and F-score (human, object, combined), SMPL errors, object rotation and translation errors, acceleration and jitter errors, and hand contact accuracy.
```shell
# step 1, once: compute SMPL vertices and joints for prediction and GT, saved as checkpoint-<seq>_k1_smpl.pth
python eval/eval_rot_chamf.py --precompute -pr <output dir>/pred --packed_path <packed GT dir> --smpl_root <SMPL-H model dir>
# step 2: evaluation
python eval/eval_rot_chamf.py -pr <output dir>/pred --packed_path <packed GT dir> --icap_template_path <InterCap object templates dir> --smpl_root <SMPL-H model dir> -i intercap
```
`<output dir>` is the folder written by the joint optimization (`main_hoiopt_smplh.py`), containing `pred/` and `checkpoint-<seq>_k1.pth`. The sequences are listed in `configs/splits/intercap-seqs-21.json`; the frames (15fps), camera transforms and selected views are in `configs/splits/seqs`, `configs/splits/w2cs_all.pkl` and `configs/splits/selected-views-map.json`. The InterCap object templates are expected as `01.ply`, `02.ply`, ... in `--icap_template_path`.



## Citation
If you use the code, please cite: 
```
@inproceedings{xie2024InterTrack,
    title = {InterTrack: Tracking Human Object Interaction without Object Templates},
    author = {Xie, Xianghui and Lenssen, Jan Eric and Pons-Moll, Gerard},
    booktitle = {International Conference on 3D Vision (3DV)},
    month = {March},
    year = {2025},
}

@inproceedings{xie2023template_free,
    title = {Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Lenssen, Jan Eric and Pons-Moll, Gerard},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2024},
}

```

## Acknowledgements
This project leverages the following excellent works, we thank the authors for open-sourcing their code: 

* The [PyTorch3D](https://github.com/facebookresearch/pytorch3d) library. 
* The [diffusers](https://github.com/huggingface/diffusers) library. 
* The [pc2](https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion/tree/main) project.
* The [smplfitter](https://github.com/isarandi/smplfitter) library from [NLF](https://virtualhumans.mpi-inf.mpg.de/nlf/). 

## License
Please see [LICENSE](./LICENSE).
