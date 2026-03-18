import torch
import torch.nn.functional as F
import os
import argparse
from pathlib import Path
from torchvision import datasets
import torch.nn as nn

# main_adaptive에서 필요한 클래스 및 유틸리티 임포트
from main_adaptive import (
    HybridAdaptiveVisionTransformer, 
    AdaptiveMultiCropWrapper, 
    LossPredictor,
    get_args_parser
)
from main_dino import DataAugmentationDINO
import utils
from vision_transformer import DINOHead

def analyze_representation_trajectory(student, images, device):
    """
    공유 레이어(Shared Phase)의 각 반복(Recurrence) 스텝 간의 코사인 유사도를 측정하여
    게으른 항등 매핑(Lazy Identity Mapping) 현상을 분석합니다.
    """
    student.eval()
    with torch.no_grad():
        # images가 리스트인 경우 (Multi-crop) 첫 번째 글로벌 뷰만 사용
        if isinstance(images, list):
            img = images[0].to(device)
        else:
            img = images.to(device)

        # backbone 직접 호출하여 중간 특징값 추출
        # HybridAdaptiveVisionTransformer.forward는 (cls_intermediates, loss_intermediates) 반환
        # cls_intermediates: list of [B, D] (length: num_shared_recurrences)
        cls_intermediates, _ = student.backbone(img)
        
        num_steps = len(cls_intermediates)
        if num_steps < 2:
            print("Shared recurrences가 2 미만이라 분석이 불가능합니다.")
            return

        # 1. 스텝 간 코사인 유사도 측정 (Step l vs Step l+1)
        # 유사도가 1에 가까울수록 레이어 간 변화가 없음을 의미 (Identity Mapping 의심)
        sims = []
        for l in range(num_steps - 1):
            z_curr = cls_intermediates[l]   # [B, D]
            z_next = cls_intermediates[l+1] # [B, D]
            sim = F.cosine_similarity(z_curr, z_next, dim=-1).mean().item()
            sims.append(sim)
        
        # 2. 첫 스텝과 마지막 스텝의 유사도 측정 (Total representation shift)
        z_first = cls_intermediates[0]
        z_last = cls_intermediates[-1]
        total_shift_sim = F.cosine_similarity(z_first, z_last, dim=-1).mean().item()

        sim_str = " -> ".join([f"{s:.4f}" for s in sims])
        print(f"\n[Representation Trajectory Analysis]")
        print(f"Step-wise Cosine Similarity: {sim_str}")
        print(f"First-to-Last Similarity: {total_shift_sim:.4f}")
        
        # 3. 평균 변화량 (L2 Distance)
        dists = []
        for l in range(num_steps - 1):
            dist = torch.norm(cls_intermediates[l+1] - cls_intermediates[l], p=2, dim=-1).mean().item()
            dists.append(dist)
        dist_str = " -> ".join([f"{d:.4f}" for d in dists])
        print(f"Step-wise L2 Distance: {dist_str}\n")

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터셋 로드 (CIFAR100 관례에 따라 DataAugmentationDINO 사용)
    transform = DataAugmentationDINO(
        args.global_crops_scale, args.local_crops_scale, args.local_crops_number)
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    print(f"Data loaded from {args.data_path}: {len(dataset)} images.")

    # 2. 모델 빌드
    num_shared = args.num_shared_recurrences
    embed_dim = args.embed_dim

    backbone = HybridAdaptiveVisionTransformer(
        img_size=args.img_size, patch_size=args.patch_size,
        embed_dim=embed_dim, num_heads=args.num_heads,
        num_indep_layers=args.num_indep_layers,
        num_shared_recurrences=num_shared)
    
    heads = nn.ModuleList([
        DINOHead(embed_dim, args.out_dim, 
                 use_bn=args.use_bn_in_head, 
                 norm_last_layer=args.norm_last_layer if l == num_shared - 1 else True)
        for l in range(num_shared)])
    
    predictor = LossPredictor(embed_dim, num_shared)
    model = AdaptiveMultiCropWrapper(backbone, heads, predictor).to(device)

    # 3. 체크포인트 로드
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # MultiCropWrapper인 student 키에서 로드
        model.load_state_dict(checkpoint['student'])
        print(f"=> loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}', using random initialization")

    # 4. 분석 실행 (첫 번째 배치에 대해)
    images, _ = next(iter(data_loader))
    analyze_representation_trajectory(model, images, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Representation Analysis', parents=[get_args_parser()])
    args = parser.parse_args()
    run_experiment(args)
