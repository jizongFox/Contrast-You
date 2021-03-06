local_folder=./runs
#
rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0306/ \
  "${local_folder}/0306/"
#
#rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
#  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0303_semi/ \
#  "${local_folder}/0303_semi/"


#rsync -azP  --exclude "*/*.png"   \
#  shanxi1:/home/jizong/Contrast-You/semi_seg/runs/test_semi \
#  "${local_folder}/shanxi1/"
