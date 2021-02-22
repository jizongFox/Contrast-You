local_folder=./runs

rsync -azP  --exclude "*/*.png"  --exclude "*/*.pth" \
  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0220/ \
  "${local_folder}/0220/"

#rsync -azP  --exclude "*/*.png"  \
#  shanxi1:/home/jizong/Contrast-You/semi_seg/runs/0218/test5 \
#  "${local_folder}/shanxi1/"
