import os
import torchvision
from PIL import Image
from tqdm import tqdm

def save_cifar100_to_folders(root_dir='./cifar100_images'):
    # torchvision을 통해 데이터 다운로드
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)

    for split_name, dataset in [('train', trainset), ('val', testset)]:
        split_dir = os.path.join(root_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"Saving {split_name} data...")
        for i in tqdm(range(len(dataset))):
            img, label = dataset[i]
            class_name = dataset.classes[label]
            
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            img_path = os.path.join(class_dir, f'{i}.png')
            img.save(img_path)

if __name__ == '__main__':
    save_cifar100_to_folders()