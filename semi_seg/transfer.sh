local_folder=./runs
#
#rsync -azP --exclude "*/*.png"  --exclude "*/tra/*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0327/ \
#  "${local_folder}/0327/"
#  --exclude "*/*events.out.tfevents*" \
rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
  --exclude "*/*.pth" \
  root@jizong.buzz:/root/main/runs/0401_prostate/ \
  "${local_folder}/0401/"

#rsync -azP --exclude "*/*.png"  --exclude "*/*.pth" \
#  --exclude "*/*.pth" \
#  root@jizong.buzz:/root/main/runs/0401_acdc/ \
#  "${local_folder}/0401/"

#rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0312_2/ \
#  "${local_folder}/0312_2/"

#rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0303_semi/ \
#  "${local_folder}/0303_semi/"

#rsync -azP  --exclude "*/*.png"  --exclude "tra/*/*.pth"  \
#  shanxi1:/home/jizong/Contrast-You/semi_seg/runs/0315 \
#  "${local_folder}/shanxi1/"
