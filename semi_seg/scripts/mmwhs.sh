lr=0.0000020000
python main_finetune.py Data.name=mmwhsmr Trainer.num_batches=300 Arch.num_classes=5 Arch.input_dim=1 RandomSeed=10 Optim.lr=${lr} Trainer.max_epoch=60 Trainer.name=finetune Trainer.save_dir=0512_mmwhs/githash_45dcbf6/mmwhsmr/random_seed_10/baseline/lr_${lr}

lr=0.0000002
python main_finetune.py Data.name=mmwhsmr Trainer.num_batches=300 Arch.num_classes=5 Arch.input_dim=1 RandomSeed=10 Optim.lr=${lr} Trainer.max_epoch=60 Trainer.name=finetune Trainer.save_dir=0512_mmwhs/githash_45dcbf6/mmwhsmr/random_seed_10/baseline/lr_${lr}

lr=0.00000002
python main_finetune.py Data.name=mmwhsmr Trainer.num_batches=300 Arch.num_classes=5 Arch.input_dim=1 RandomSeed=10 Optim.lr=${lr} Trainer.max_epoch=60 Trainer.name=finetune Trainer.save_dir=0512_mmwhs/githash_45dcbf6/mmwhsmr/random_seed_10/baseline/lr_${lr}

lr=0.000000002
python main_finetune.py Data.name=mmwhsmr Trainer.num_batches=300 Arch.num_classes=5 Arch.input_dim=1 RandomSeed=10 Optim.lr=${lr} Trainer.max_epoch=60 Trainer.name=finetune Trainer.save_dir=0512_mmwhs/githash_45dcbf6/mmwhsmr/random_seed_10/baseline/lr_${lr}

lr=0.0000000002
python main_finetune.py Data.name=mmwhsmr Trainer.num_batches=300 Arch.num_classes=5 Arch.input_dim=1 RandomSeed=10 Optim.lr=${lr} Trainer.max_epoch=60 Trainer.name=finetune Trainer.save_dir=0512_mmwhs/githash_45dcbf6/mmwhsmr/random_seed_10/baseline/lr_${lr}
