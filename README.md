# GameTheoreticPruning
Experiments in Pruning using Game Theoretic Approaches by modeling pruning as a nim game

## Prep
conda create -n advprune python=3.7
conda activate advprune
pip install torch torchvision 


### Status
1. Baselines
    VGG16 cifar10 (done) 86.85 @1
    VGG16 cifar100 (done) 58.43 @1
    RESNET50 cifar10 (done) 86.89 @1
    RESNET50 cifar100 (done) 62.74 @1
    DPN92 cifar10 (done) 88.61 @1
    DPN92 cifar100  (done) 66.13 @1

2. Prune each baseline with 8 prunings
    DPN92 Cifar 10 


    python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method RANDOM > results/DPN92cifar100PruneRANDOM11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method L1 > results/DPN92cifar100PruneL111272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method POSITIVE > results/DPN92cifar100PrunePositive11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method NEGATIVE > DPN92cifar100PruneNegative11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method MAGNITUDE > DPN92cifar100PruneMAGNITUDE11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method MAGNITUDE+L1 > results/DPN92cifar100PruneMAGNITUDE+L111272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method MAGNITUDE+RANDOM > results/DPN92cifar100PruneMAGNITUDE+RANDOM11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method 1+RANDOM > results/DPN92cifar100Prune1+RANDOM11272020.txt






VGG16 Cifar10 L1 Norm 
[(0, 86.85), (0, 76.13), (0, 84.65), (0.05555764222756411, 91.82), (0.10185567741720084, 92.34), (0.15741331964476493, 92.38), (0.20366244845920134, 92.24), (0.2591222779363649, 92.7), (0.3053388025006677, 92.44), (0.3515553270649706, 92.87), (0.40698907314202726, 92.68), (0.4531729934561965, 92.69), (0.5085936978331997, 92.59), (0.5547776181473691, 92.83), (0.6008963299612714, 92.59), (0.656219221587874, 92.67), (0.7023216312767094, 92.21), (0.7576445229033121, 92.41), (0.8036165155916133, 92.73), (0.8587829068175749, 92.23), (0.9047548995058763, 92.48), (0.9047548995058763, 93.27), (0.9047548995058763, 93.34), (0.9047548995058763, 93.4), (0.9047548995058763, 93.27), (0.9047548995058763, 93.51), (0.9047548995058763, 93.41), (0.9047548995058763, 93.45), (0.9047548995058763, 93.53), (0.9047548995058763, 93.45), (0.9047548995058763, 93.4)]



VGG16 CIFAR100 L1 500 Epochs train
[(0, 51.94), (0, 55.5), (0, 51.83), (0.05555764222756411, 71.55), (0.10185567741720084, 72.18), (0.15741331964476493, 72.25), (0.20366244845920134, 72.62), (0.2591222779363649, 72.67), (0.3053388025006677, 72.23), (0.3515553270649706, 72.98), (0.40698907314202726, 72.3), (0.4531729934561965, 71.53), (0.5085936978331997, 72.55), (0.5547776181473691, 71.83), (0.6008963299612714, 72.41), (0.656219221587874, 71.79), (0.7023216312767094, 71.9), (0.7576445229033121, 71.39), (0.8036165155916133, 71.03), (0.8587829068175749, 71.24), (0.9047548995058763, 71.55), (0.9047548995058763, 73.5), (0.9047548995058763, 73.54), (0.9047548995058763, 73.77), (0.9047548995058763, 73.91), (0.9047548995058763, 73.79), (0.9047548995058763, 73.7), (0.9047548995058763, 73.6), (0.9047548995058763, 73.64), (0.9047548995058763, 73.91), (0.9047548995058763, 73.73)]

