echo("Starting pruning experiments DPN92 on CIFAR10")
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method RANDOM > results/DPN92cifar10PruneRANDOM11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method L1 > results/DPN92cifar10PruneL111272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method POSITIVE > results/DPN92cifar10PrunePositive11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method NEGATIVE > results/DPN92cifar10PruneNegative11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method MAGNITUDE > results/DPN92cifar10PruneMAGNITUDE11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method MAGNITUDE+L1 > results/DPN92cifar10PruneMAGNITUDE+L111272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method MAGNITUDE+RANDOM > results/DPN92cifar10PruneMAGNITUDE+RANDOM11272020.txt
python main.py --arch DPN92 --dataset cifar10 --load_name models/DPN92cifar10 --prune --prune_method L1+RANDOM > results/DPN92cifar10PruneL1+RANDOM11272020.txt
echo("Finished pruning experiments DPN92 on CIFAR10")
echo("Starting pruning experiments DPN92 on CIFAR100")
python main.py --arch DPN92 --dataset cifar100 --load_name models/DPN92cifar100 --prune --prune_method RANDOM > results/DPN92cifar100PruneRANDOM11272020.txt
python main.py --arch DPN92 --dataset cifar100 --load_name models/DPN92cifar100 --prune --prune_method L1 > results/DPN92cifar100PruneL111272020.txt
python main.py --arch DPN92 --dataset cifar100 --load_name models/DPN92cifar100 --prune --prune_method POSITIVE > results/DPN92cifar100PrunePositive11272020.txt
python main.py --arch DPN92 --dataset cifar100 --load_name models/DPN92cifar100 --prune --prune_method NEGATIVE > results/DPN92cifar100PruneNegative11272020.txt
python main.py --arch DPN92 --dataset cifar100 --load_name models/DPN92cifar100 --prune --prune_method MAGNITUDE > results/DPN92cifar100PruneMAGNITUDE11272020.txt
python main.py --arch DPN92 --dataset cifar100 --load_name models/DPN92cifar100 --prune --prune_method MAGNITUDE+L1 > results/DPN92cifar100PruneMAGNITUDE+L111272020.txt
python main.py --arch DPN92 --dataset cifar100 --load_name models/DPN92cifar100 --prune --prune_method MAGNITUDE+RANDOM > results/DPN92cifar100PruneMAGNITUDE+RANDOM11272020.txt
python main.py --arch DPN92 --dataset cifar100 --load_name models/DPN92cifar100 --prune --prune_method L1+RANDOM > results/DPN92cifar100PruneL1+RANDOM11272020.txt
echo("Finished pruning experiments DPN92 on CIFAR100")
echo("Starting pruning experiments VGG16 on CIFAR10")
python main.py --arch VGG16 --dataset cifar10 --load_name models/VGG16cifar10 --prune --prune_method RANDOM > results/VGG16cifar10PruneRANDOM11272020.txt
python main.py --arch VGG16 --dataset cifar10 --load_name models/VGG16cifar10 --prune --prune_method L1 > results/VGG16cifar10PruneL111272020.txt
python main.py --arch VGG16 --dataset cifar10 --load_name models/VGG16cifar10 --prune --prune_method POSITIVE > results/VGG16cifar10PrunePositive11272020.txt
python main.py --arch VGG16 --dataset cifar10 --load_name models/VGG16cifar10 --prune --prune_method NEGATIVE > results/VGG16cifar10PruneNegative11272020.txt
python main.py --arch VGG16 --dataset cifar10 --load_name models/VGG16cifar10 --prune --prune_method MAGNITUDE > results/VGG16cifar10PruneMAGNITUDE11272020.txt
python main.py --arch VGG16 --dataset cifar10 --load_name models/VGG16cifar10 --prune --prune_method MAGNITUDE+L1 > results/VGG16cifar10PruneMAGNITUDE+L111272020.txt
python main.py --arch VGG16 --dataset cifar10 --load_name models/VGG16cifar10 --prune --prune_method MAGNITUDE+RANDOM > results/VGG16cifar10PruneMAGNITUDE+RANDOM11272020.txt
python main.py --arch VGG16 --dataset cifar10 --load_name models/VGG16cifar10 --prune --prune_method L1+RANDOM > results/VGG16cifar10PruneL1+RANDOM11272020.txt
echo("Finished pruning experiments VGG16 on CIFAR10")
echo("Starting pruning experiments VGG16 on CIFAR100")
python main.py --arch VGG16 --dataset cifar100 --load_name models/VGG16cifar100 --prune --prune_method RANDOM > results/VGG16cifar100PruneRANDOM11272020.txt
python main.py --arch VGG16 --dataset cifar100 --load_name models/VGG16cifar100 --prune --prune_method L1 > results/VGG16cifar100PruneL111272020.txt
python main.py --arch VGG16 --dataset cifar100 --load_name models/VGG16cifar100 --prune --prune_method POSITIVE > results/VGG16cifar100PrunePositive11272020.txt
python main.py --arch VGG16 --dataset cifar100 --load_name models/VGG16cifar100 --prune --prune_method NEGATIVE > results/VGG16cifar100PruneNegative11272020.txt
python main.py --arch VGG16 --dataset cifar100 --load_name models/VGG16cifar100 --prune --prune_method MAGNITUDE > results/VGG16cifar100PruneMAGNITUDE11272020.txt
python main.py --arch VGG16 --dataset cifar100 --load_name models/VGG16cifar100 --prune --prune_method MAGNITUDE+L1 > results/VGG16cifar100PruneMAGNITUDE+L111272020.txt
python main.py --arch VGG16 --dataset cifar100 --load_name models/VGG16cifar100 --prune --prune_method MAGNITUDE+RANDOM > results/VGG16cifar100PruneMAGNITUDE+RANDOM11272020.txt
python main.py --arch VGG16 --dataset cifar100 --load_name models/VGG16cifar100 --prune --prune_method L1+RANDOM > results/VGG16cifar100PruneL1+RANDOM11272020.txt
echo("Finished pruning experiments VGG16 on CIFAR100")
echo("Starting pruning experiments RESNET50 on CIFAR10")
python main.py --arch RESNET50 --dataset cifar10 --load_name models/RESNET50cifar10 --prune --prune_method RANDOM > results/RESNET50cifar10PruneRANDOM11272020.txt
python main.py --arch RESNET50 --dataset cifar10 --load_name models/RESNET50cifar10 --prune --prune_method L1 > results/RESNET50cifar10PruneL111272020.txt
python main.py --arch RESNET50 --dataset cifar10 --load_name models/RESNET50cifar10 --prune --prune_method POSITIVE > results/RESNET50cifar10PrunePositive11272020.txt
python main.py --arch RESNET50 --dataset cifar10 --load_name models/RESNET50cifar10 --prune --prune_method NEGATIVE > results/RESNET50cifar10PruneNegative11272020.txt
python main.py --arch RESNET50 --dataset cifar10 --load_name models/RESNET50cifar10 --prune --prune_method MAGNITUDE > results/RESNET50cifar10PruneMAGNITUDE11272020.txt
python main.py --arch RESNET50 --dataset cifar10 --load_name models/RESNET50cifar10 --prune --prune_method MAGNITUDE+L1 > results/RESNET50cifar10PruneMAGNITUDE+L111272020.txt
python main.py --arch RESNET50 --dataset cifar10 --load_name models/RESNET50cifar10 --prune --prune_method MAGNITUDE+RANDOM > results/RESNET50cifar10PruneMAGNITUDE+RANDOM11272020.txt
python main.py --arch RESNET50 --dataset cifar10 --load_name models/RESNET50cifar10 --prune --prune_method L1+RANDOM > results/RESNET50cifar10PruneL1+RANDOM11272020.txt
echo("Finished pruning experiments RESNET50 on CIFAR10")
echo("Starting pruning experiments RESNET50 on CIFAR100")
python main.py --arch RESNET50 --dataset cifar100 --load_name models/RESNET50cifar100 --prune --prune_method RANDOM > results/RESNET50cifar100PruneRANDOM11272020.txt
python main.py --arch RESNET50 --dataset cifar100 --load_name models/RESNET50cifar100 --prune --prune_method L1 > results/RESNET50cifar100PruneL111272020.txt
python main.py --arch RESNET50 --dataset cifar100 --load_name models/RESNET50cifar100 --prune --prune_method POSITIVE > results/RESNET50cifar100PrunePositive11272020.txt
python main.py --arch RESNET50 --dataset cifar100 --load_name models/RESNET50cifar100 --prune --prune_method NEGATIVE > results/RESNET50cifar100PruneNegative11272020.txt
python main.py --arch RESNET50 --dataset cifar100 --load_name models/RESNET50cifar100 --prune --prune_method MAGNITUDE > results/RESNET50cifar100PruneMAGNITUDE11272020.txt
python main.py --arch RESNET50 --dataset cifar100 --load_name models/RESNET50cifar100 --prune --prune_method MAGNITUDE+L1 > results/RESNET50cifar100PruneMAGNITUDE+L111272020.txt
python main.py --arch RESNET50 --dataset cifar100 --load_name models/RESNET50cifar100 --prune --prune_method MAGNITUDE+RANDOM > results/RESNET50cifar100PruneMAGNITUDE+RANDOM11272020.txt
python main.py --arch RESNET50 --dataset cifar100 --load_name models/RESNET50cifar100 --prune --prune_method L1+RANDOM > results/RESNET50cifar100PruneL1+RANDOM11272020.txt
echo("Finished pruning experiments RESNET50 on CIFAR100")
echo("Done Pruning Experiments")