local_folder=./runs
#
#rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0304_new2/ \
#  "${local_folder}/0304_new2/"

#rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0303_semi/ \
#  "${local_folder}/0303_semi/"


rsync -azP  --exclude "*/*.png"   \
  shanxi1:/home/jizong/Contrast-You/semi_seg/runs/test_semi \
  "${local_folder}/shanxi1/"
