import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--label_ratio", default=0.05, type=float)
parser.add_argument("-n", "--trainer_name", required=True, type=str)
parser.add_argument("-b", "--num_batches", default=500, type=int)
parser.add_argument("-s", "--random_seed", default=1, type=int)
parser.add_argument("-o", "--contrast_on", default="partition", type=str)
parser.add_argument("-w", "--reg_weight", default=0.0, type=float)
parser.add_argument("--save_dir", default=None, type=str)
parser.add_argument("-a", "--augment", default="simple", type=str)
parser.add_argument("-g", "--group_sample_num", default=6, type=int)
# parser.add_argument("--encoder_extractor", default="Conv5",type=str)
# parser.add_argument("--decoder_extractor", default="Up_conv3", type=str)
# parser.add_argument("--decoder_enable_grad_from", default="Up5", type=str)
parser.add_argument("--time", default=4, type=int)

args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio

# trainer_name="contrast" # or contrastMT
trainer_name = args.trainer_name
contrast_on = args.contrast_on
save_dir_main = args.save_dir if args.save_dir else "0801_main_contrast"
save_dir = f"{save_dir_main}/" \
           f"label_data_ration_{labeled_data_ratio}/" \
           f"{trainer_name}/" \
           f"contrast_on_{contrast_on}/" \
           f"augment_{args.augment}/" \
           f"group_sample_{args.group_sample_num}"

if trainer_name == "contrastMT":
    save_dir = save_dir + f"/reg_weight_{args.reg_weight:.2f}"

common_opts = f" Trainer.name={trainer_name} " \
              f" PretrainEncoder.group_option={contrast_on} " \
              f" RandomSeed={random_seed} " \
              f" Data.labeled_data_ratio={labeled_data_ratio} " \
              f" Data.unlabeled_data_ratio={1.0 - labeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Augment={args.augment} " \
              f" ContrastData.group_sample_num={args.group_sample_num}"

if trainer_name == "contrastMT":
    common_opts += f" FineTune.reg_weight={args.reg_weight} "

jobs = [
    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/baseline  "
    f"Trainer.train_encoder=False Trainer.train_decoder=False ",
    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder/conv5/enc_mlp  "
    f"Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.extract_position=Conv5",
    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder/conv5/enc_linear  "
    f"Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.extract_position=Conv5 "
    f"PretrainEncoder.ptype=linear ",

    # f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder/conv4  "
    #                                             f"Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.extract_position=Conv4",
    #
    # f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder/conv3  "
    #                                             f"Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.extract_position=Conv3",

    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder/encoder_conv5/decoder_conv3/freeze_encoder/dec_mlp "
    f"Trainer.train_encoder=True Trainer.train_decoder=True "
    f"PretrainEncoder.extract_position=Conv5 "
    f"PretrainDecoder.extract_position=Up_conv3 "
    f"PretrainDecoder.enable_grad_from=Up5 ",

    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder/encoder_conv5/decoder_conv2/freeze_encoder/dec_mlp "
    f"Trainer.train_encoder=True Trainer.train_decoder=True "
    f"PretrainEncoder.extract_position=Conv5 "
    f"PretrainDecoder.extract_position=Up_conv2 "
    f"PretrainDecoder.enable_grad_from=Up5 ",

    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder/encoder_conv5/decoder_conv3/enable_encoder/dec_mlp "
    f"Trainer.train_encoder=True Trainer.train_decoder=True "
    f"PretrainEncoder.extract_position=Conv5 "
    f"PretrainDecoder.extract_position=Up_conv3 "
    f"PretrainDecoder.enable_grad_from=Conv1 ",

    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder/encoder_conv5/decoder_conv2/enable_encoder/dec_mlp "
    f"Trainer.train_encoder=True Trainer.train_decoder=True "
    f"PretrainEncoder.extract_position=Conv5 "
    f"PretrainDecoder.extract_position=Up_conv2 "
    f"PretrainDecoder.enable_grad_from=Conv1 ",

    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder/encoder_conv5/decoder_conv3/freeze_encoder/dec_linear "
    f"Trainer.train_encoder=True Trainer.train_decoder=True "
    f"PretrainEncoder.extract_position=Conv5 "
    f"PretrainDecoder.extract_position=Up_conv3 "
    f"PretrainDecoder.enable_grad_from=Up5 "
    f"PretrainDecoder.ptype=linear ",

    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder/encoder_conv5/decoder_conv2/freeze_encoder/dec_linear "
    f"Trainer.train_encoder=True Trainer.train_decoder=True "
    f"PretrainEncoder.extract_position=Conv5 "
    f"PretrainDecoder.extract_position=Up_conv2 "
    f"PretrainDecoder.enable_grad_from=Up5 "
    f"PretrainDecoder.ptype=linear ",

    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder/encoder_conv5/decoder_conv3/enable_encoder/dec_linear "
    f"Trainer.train_encoder=True Trainer.train_decoder=True "
    f"PretrainEncoder.extract_position=Conv5 "
    f"PretrainDecoder.extract_position=Up_conv3 "
    f"PretrainDecoder.enable_grad_from=Conv1 "
    f"PretrainDecoder.ptype=linear ",

    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder/encoder_conv5/decoder_conv2/enable_encoder/dec_linear "
    f"Trainer.train_encoder=True Trainer.train_decoder=True "
    f"PretrainEncoder.extract_position=Conv5 "
    f"PretrainDecoder.extract_position=Up_conv2 "
    f"PretrainDecoder.enable_grad_from=Conv1 "
    f"PretrainDecoder.ptype=linear ",

]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="./", on_local=False, time=args.time)
for j in jobs:
    jobsubmiter.prepare_env(["source ./venv/bin/activate ", "export OMP_NUM_THREADS=1", ])
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)
