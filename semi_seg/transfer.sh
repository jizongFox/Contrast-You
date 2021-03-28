local_folder=./runs
#
rsync -azP  --exclude "*/*.png" --exclude "*/*.pth"  --exclude "*/tra/*/*.pth" \
  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0326_test/ \
  "${local_folder}/0326_test/"
#  --exclude "*/*events.out.tfevents*" \

#rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0312_2/ \
#  "${local_folder}/0312_2/"
#
#rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0303_semi/ \
#  "${local_folder}/0303_semi/"


#rsync -azP  --exclude "*/*.png"  --exclude "tra/*/*.pth"  \
#  shanxi1:/home/jizong/Contrast-You/semi_seg/runs/0315 \
#  "${local_folder}/shanxi1/"
