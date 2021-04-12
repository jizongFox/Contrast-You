local_folder=./runs
#
#rsync -azP --exclude "*/*.png"  --exclude "*/tra/*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0327/ \
#  "${local_folder}/0327/"
#  --exclude "*/*events.out.tfevents*" \
#rsync -azP  --exclude "*/*.png" --exclude "*/*.pth" \
#  --exclude "*/*.pth" \
#  root@jizong.buzz:/root/main/runs/0402_semi/ \
#  "${local_folder}/0402_semi_acdc/"

rsync -azP --exclude "*/*.png"  --exclude "*/*.pth" \
  --exclude "*/patient*"  \
  root@jizong.buzz:/root/main/runs/0412_prostate \
  "${local_folder}/"

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
