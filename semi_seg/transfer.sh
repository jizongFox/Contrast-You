local_folder=./runs

rsync -azP --exclude "*/*.pth" --exclude "*/*.png" --exclude "*/*.log" \
  beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0217/ \
  "${local_folder}/0217/"
