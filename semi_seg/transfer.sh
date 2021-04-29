local_folder=runs
#
#rsync -azP --exclude "*/*.png"  --exclude "*/tra/*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0327/ \
#  "${local_folder}/0327/"
#  --exclude "*/*events.out.tfevents*" \
#rsync -azP  --exclude "*/*.png" --exclude "*/*.pth" \
#  --exclude "*/*.pth" \
#  root@jizong.buzz:/root/main/runs/0402_semi/ \
#  "${local_folder}/0402_semi_acdc/"

rsync -azP --exclude "*/*.png"  --exclude "*/tra/*/*.pth" --exclude "*/*.pth" \
  --exclude "*/patient*"   \
  root@jizong.buzz:/root/main/runs/cedar/0431* \
  "${local_folder}/cedar"

#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0326/githash_912bc30/acdc/random_seed_10/baseline \
#  "${local_folder}/ps/acdc"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0326/githash_d065b37/acdc/random_seed_10/sample_num_10/global_Conv5_Conv5_Conv5_1.0_0.5_0.1/contrast_on_partition_patient_cycle/self-paced/method_hard_hard_hard/loss_params*4.0_14.0*4.0_14.0*4.0_14.0 \
#  "${local_folder}/pretrain-self-finetune/acdc"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0326/githash_d065b37/acdc/random_seed_10/sample_num_10/global_Conv5_Conv5_Conv5_1.0_0.2_0.1/contrast_on_partition_patient_cycle/infonce \
#  "${local_folder}/pretrain-info-finetune/acdc"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0326/githash_d065b37/acdc/random_seed_10/sample_num_10/global_Conv5_Conv5_Conv5_1.0_0.2_0.1/contrast_on_partition_patient_cycle/infonce \
#  "${local_folder}/pretrain-info-finetune/acdc"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0326_semi/contrast_on_parition/self/githash_5d2e9c4/acdc/random_seed_10/checkpoint_null/mt/mt_0.1 \
#  "${local_folder}/mt/acdc"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0326_semi/contrast_on_all/self/githash_49d9567/acdc/random_seed_10/checkpoint_yes/mt/mt_0.5 \
#  "${local_folder}/pretrain-mt/acdc"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0326_semi/contrast_on_all/infonce/githash_bb1f1ec/acdc/random_seed_10/checkpoint_yes/infoncemt/info_0.01_mt_1.0 \
#  "${local_folder}/pretrain-info-mt-info/acdc"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0326_semi/contrast_on_all/self/githash_bb1f1ec/acdc/random_seed_10/checkpoint_yes/infoncemt/info_0.1_mt_0.5 \
#  "${local_folder}/pretrain-self-mt-self/acdc"

# prostate

#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0416_prostate/githash_c44b4dc/prostate/random_seed_20/baseline \
#  "${local_folder}/ps/prostate"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0416_prostate/githash_e7481ca/prostate/random_seed_40/sample_num_5/global_Conv5_1.0/contrast_on_partition/self-paced/method_soft/loss_params*3.0_60.0/type_inversesquare \
#  "${local_folder}/pretrain-self-finetune/prostate"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0416_prostate/githash_e7481ca/prostate/random_seed_30/sample_num_5/global_Conv5_1.0/contrast_on_partition/infonce \
#  "${local_folder}/pretrain-info-finetune/prostate"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0417_prostate_semi/semi_supervised/null_checkpoint/null/githash_a3c409b/prostate/random_seed_10/checkpoint_null/mt/mt_0.05 \
#  "${local_folder}/mt/prostate"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0417_prostate_semi/semi_supervised/contrast_on_all/self/githash_a3c409b/prostate/random_seed_10/checkpoint_yes/mt/mt_0.5 \
#  "${local_folder}/pretrain-mt/prostate"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0417_prostate_semi/semi_supervised/contrast_on_parition/infonce/githash_a3c409b/prostate/random_seed_10/checkpoint_yes/infoncemt/info_0.5_mt_0.8/ \
#  "${local_folder}/pretrain-info-mt-info/prostate"
#
#rsync -azP beluga:/home/jizong/scratch/Contrast-You/semi_seg/runs/0417_prostate_semi/semi_supervised/contrast_on_parition/self/githash_a3c409b/prostate/random_seed_10/checkpoint_yes/infoncemt/info_0.5_mt_0.8 \
#  "${local_folder}/pretrain-self-mt-self/prostate"


#rsync -azP --exclude "*/*.png"  --exclude "*/*.pth" \
#  --exclude "*/*.pth" --exclude "*/features/*"  \
#  shanxi1:/home/jizong/Contrast-You/semi_seg/runs/monitor \
#  "${local_folder}/0405_monitor/"

#rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0312_2/ \
#  "${local_folder}/0312_2/"

#rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0303_semi/ \
#  "${local_folder}/0303_semi/"

#rsync -azP  --exclude "*/*.png"  --exclude "tra/*/*.pth"  \
#  shanxi1:/home/jizong/Contrast-You/semi_seg/runs/0315 \
#  "${local_folder}/shanxi1/"
