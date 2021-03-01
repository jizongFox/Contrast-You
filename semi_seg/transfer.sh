local_folder=./runs

rsync -azP  --exclude "*/*.png"  --exclude "*/tra/*/*.pth" \
  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0228/ \
  "${local_folder}/0228/"

#rsync -azP  --exclude "*/*.png"  \
#  shanxi1:/home/jizong/Contrast-You/semi_seg/runs/0223 \
#  "${local_folder}/shanxi1/"
