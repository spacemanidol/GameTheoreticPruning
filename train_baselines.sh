python main.py --prune --arch RESNET50 --dataset cifar100 --save_name models/resnet50cifar100 --train --eval --epochs 500
python main.py --prune --arch RESNET50 --dataset cifar10 --save_name models/DPN92cifar10 --train --eval --epochs 500
python main.py --prune --arch DPN92 --dataset cifar100 --save_name models/DPN92cifar100 --train --eval --epochs 500
python main.py --prune --arch DPN92 --dataset cifar10 --save_name models/DPN92cifar10 --train --eval --epochs 500
python main.py --prune --arch VGG16 --dataset cifar100 --save_name models/VGG16cifar100 --train --eval --epochs 500
python main.py --prune --arch VGG16 --dataset cifar10 --save_name models/VGG16cifar10 --train --eval --epochs 500