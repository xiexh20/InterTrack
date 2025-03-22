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
The code is tested on `torch=1.12.1+cu121, cuda12.1, debian11`. We recommend using anaconda environment:
```shell
conda create -n intertrack python=3.8
conda activate intertrack 
```
Required packages can be installed by:
```shell
pip install -r pre-requirements.txt # Install pytorch and others
pip install -r requirements.txt     # Install pytorch3d from source
```

**SMPL body models**: We use SMPL-H (mano_v1.2) from [this website](https://mano.is.tue.mpg.de/download.php). 
Download and unzip to a local path and modify in SMPL_MODEL_ROOT in `lib_smpl/const.py`.



## Quick start
### Download checkpoints
```shell
python download_models.py
```

### Download demo data
We prepare two example sequences for quick start, one is captured by mobile phone and the other is from BEHAVE dataset. 
Download the packed file from [Edmond](https://edmond.mpg.de/file.xhtml?fileId=310951&version=3.0) and then do `unzip InterTrack-demo-data.zip -d demo-data `.
Update the path to `demo-data` in command line argument `dataset.demo_data_path`. 

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

To run test on more BEHAVE sequences, you will need to download [this packed file](https://edmond.mpg.de/file.xhtml?fileId=310881&version=3.0) and update `behave_packed_dir` in config file. 
## Training
Coming soon... 

## Evaluation
Run test on the full behave dataset: coming soon...


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
