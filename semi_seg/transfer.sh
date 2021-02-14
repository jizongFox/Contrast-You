local_folder=./runs

rsync -azP --exclude "*/*.pth" --exclude "*/*.png" \
beluga:/lustre04/scratch/jizong/Contrast-You/semi_seg/runs/0214/ \
"${local_folder}/0214"
