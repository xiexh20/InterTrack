#!/bin/bash
set -e # exit if errors

# full pipeline to run InterTrack on the demo video data
# define colors for cmd output
GREEN=$'\e[0;32m'
RED=$'\e[0;31m'
NC=$'\e[0m'

split_file=${PWD}/configs/splits/demo-seq-phone-15fps.pkl
# step 1.1: run HDM stage 1
echo "${RED} InterTrack demo step 1/6: run HDM reconstruction ${NC}"
python main.py run.name=stage1-chair dataloader.batch_size=16 model.model_name=pc2-diff-ho-sepsegm \
model.predict_binary=True dataset.split_file=${split_file} \
run.job=sample dataset.type=behave-test run.save_name=demo dataset.test_transl_type=estimated-2d dataset.std_coverage=3.8 \
run.sample_save_gt=False run.diffusion_scheduler=ddim run.num_inference_steps=100  run.batch_start=0 run.batch_end=100
# step 1.2: run HDM stage 2
python main.py run.name=stage2-chair model=ho-attn model.attn_weight=1.0 model.attn_type=coord3d+posenc-learnable \
model.point_visible_test=combine dataset.split_file=${split_file} dataloader.batch_size=16 \
dataset.ho_segm_pred_path=${PWD}/outputs/stage1-chair/single/demo/pred \
run.job=sample run.sample_mode=interm-pred run.sample_noise_step=500 \
dataset.type=behave-attn-test run.save_name=demo-stage2 dataset.test_transl_type=estimated-2d dataset.std_coverage=3.8 \
run.sample_save_gt=False run.diffusion_scheduler=ddim run.num_inference_steps=100  run.batch_start=0 run.batch_end=100

hdm_out=${PWD}/outputs/stage2-chair/single/demo-stage2/pred
# Step 2: run CorrAE to obtain human correspondence
echo "${RED} InterTrack demo step 2/6: run CorrAE to obtain human correspondence ${NC}"
python main_corrae.py run.name=corrAE model=pvcnn-ae model.num_points=6890 dataset=behave dataset.fix_sample=True \
dataset.type=behave-attn-test dataloader.batch_size=32 run.freeze_feature_model=False dataset.split_file=${split_file} \
run.job=sample dataset.ho_segm_pred_path=${hdm_out} dataset.test_transl_type=estimated-2d \
run.sample_mode=interm-hum run.save_name=humae-phone  run.batch_start=0 run.batch_end=100

# Step 3: optimize human with one global shape and per-frame body pose
echo "${RED} InterTrack demo step 3/6: optimize human with one global shape and per-frame body pose ${NC}"
# Link the SMPLH results from CorrAE step to HDM out dir
ln -sf ${PWD}/outputs/corrAE/single/humae-phone/smplh ${PWD}/outputs/stage2-chair/single/demo-stage2/smplh
# Do optimization
python main_humopt_smpl.py run.name=corrAE model=pvcnn-ae model.num_points=6890 dataset=behave dataset.type=behave-attn-test \
dataset.test_transl_type=estimated-2d dataloader.batch_size=256 run.job=sample  run.max_steps=5000 logging.wandb=False \
run.save_name=opt-hum-orighdm model.hum_lw_cd=100 model.hum_lw_rigid=1000 model.hoi_lw_temp_h=1000 model.hum_opt_s=True model.hum_opt_t=True \
dataset.split_file=${split_file} dataset.ho_segm_pred_path=${hdm_out}

# Step 4: predict object pose
echo "${RED} InterTrack demo step 4/6: predict object pose using TOPNet ${NC}"
python main_avgrot.py run.name=so3smpl_5obj-none-dtune-acc0.2 model.model_name=so3smpl-reg run.mixed_precision=no \
model.so3_eps_scale=0.5 model.so3_rot_type=abs model.lw_rot_acc=0.2 model.so3_loss_type=rot+acc-l1 model.image_feature_model=dinov2_vitb14_tune \
model.pose_feat_dim=128 model.norm_layer=batch-emb model.smpl_cond_type=none dataset=behave dataset.type=behave-video-test \
dataloader.num_workers=2 dataloader.batch_size=16  dataset.window=1 dataset.clip_len=64 run.job=sample run.save_name=vl64-avg-predvis \
run.job=sample run.sample_save_gt=False dataset.split_file=${PWD}/configs/splits/demo-seq-phone-15fps-video.pkl \
dataset.ho_segm_pred_path=${PWD}/outputs/stage2-chair/single/demo-stage2/pred

# Step 5: optimize canonical object shape and per-frame object pose, this is the most time consuming part
echo "${RED} InterTrack demo step 5/6: optimize object shape and pose ${NC}"
python main_objopt_pts.py run.name=opt-obj model=pvcnn-ae dataset=behave dataset.type=behave-attn-test \
dataset.test_transl_type=estimated-2d dataset.std_coverage=3.8 \
dataset.ho_segm_pred_path=${PWD}/outputs/stage2-chair/single/demo-stage2/pred \
run.sample_mode=recon-rt dataloader.batch_size=64 run.job=sample model.obj_opt_t=True model.obj_opt_r=True model.obj_opt_s=True  \
dataset.pred_obj_pose_path=${PWD}/outputs/so3smpl_5obj-none-dtune-acc0.2/single/vl64-avg-predvis/metadata \
model.obj_lw_temp_t=1000 model.obj_lw_temp_r=50 model.obj_lw_temp_s=1000 model.obj_opt_occ_thres=0.5  run.max_steps=8000   \
logging.wandb=False dataloader.num_workers=2 model.obj_opt_noise=0.01 run.save_name=opt-nohum-a0.2-cdhoi \
dataset.split_file=${PWD}/configs/splits/demo-seq-phone-15fps.pkl

# Step 6: joint human + object optimization
echo "${RED} InterTrack demo step 6/6: optimize human and object together ${NC}"
# this still requires the corrAE to run latent， so the ckpt is needed.
python main_hoiopt_smplh.py run.name=corrAE model=pvcnn-ae model.num_points=6890 dataset=behave \
dataset.type=behave-attn-test dataset.test_transl_type=estimated-2d dataset.std_coverage=3.8 dataloader.batch_size=64 run.job=sample  \
run.max_steps=2500 logging.wandb=False  model.hum_lw_cd=100 model.hum_lw_rigid=1000 model.hoi_lw_temp_h=1000  model.hum_opt_t=False \
model.hoi_lw_cont=10. model.hum_opt_lat=True   model.obj_opt_t=False model.obj_opt_r=True model.obj_opt_s=False model.obj_opt_shape=False \
model.obj_lw_temp_t=1000 model.obj_lw_temp_r=50 model.obj_lw_temp_s=1000 model.obj_opt_occ_thres=0.8 \
dataset.pred_obj_pose_path=${PWD}/outputs/so3smpl_5obj-none-dtune-acc0.2/single/vl64-avg-predvis/metadata \
dataset.ho_segm_pred_path=${PWD}/outputs/stage2-chair/single/demo-stage2/pred  \
dataset.hoi_opt_obj_shape_path=${PWD}/outputs/opt-obj/single/opt-nohum-a0.2-cdhoi \
dataset.hoi_opt_hum_shape_path=${PWD}/outputs/corrAE/single/opt-hum-orighdm run.save_name=opt-hoi-orighdm \
dataset.split_file=${PWD}/configs/splits/demo-seq-phone-15fps.pkl

