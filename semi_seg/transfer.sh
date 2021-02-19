local_folder=./runs

rsync -azP  --exclude "*/*.png"  --exclude "*/tra/*/*.pth" \
  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0218/test4 \
  "${local_folder}/0218/"
