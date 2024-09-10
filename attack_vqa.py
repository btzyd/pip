import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from datasets import vqa_dataset, save_result
from models import InstructBlip_PIP, Blip2_PIP
import utils
from attacks import PGD_Attack, CWL2AttackWithoutLabels


def main(args):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating vqa datasets")

    datasets = vqa_dataset(args.annotation_file, args.image_root)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()               
        sampler = torch.utils.data.DistributedSampler(datasets, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler = None

    data_loader = DataLoader(datasets, batch_size=1, num_workers=4, pin_memory=True, sampler=sampler, shuffle=False, collate_fn=None, drop_last=False)              

    print("Creating model")
    if args.lvlm=="iblip":
        model = InstructBlip_PIP(args.lvlm_root, args.lvlm_llm, args.attack_position)
    elif args.lvlm=="blip2":
        model = Blip2_PIP(args.lvlm_root, args.lvlm_llm, args.attack_position)


    model = model.to(device)   

    print("Start training")
    start_time = time.time()    

    model.eval()

    if args.attack_method=="pgd":
        attack = PGD_Attack(model, args.pgd_eps, args.pgd_lr, args.pgd_steps)
    elif args.attack_method=="cw":
        attack = CWL2AttackWithoutLabels(model, args.cw_max_iter, args.cw_lr, args.cw_const)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Attacking by {args.attack_method} on {args.attack_position}:"
    print_freq = 5
    
    vqa_result = []
    adv_process = []
    for n, (image_path, image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        

        image = image.to(device,non_blocking=True)
        answer_before_attack = model.predict(image, question)
        if args.attack_position=="clip":
            clean_featrue = model.get_clip(image)
            adv_image = attack(image, question, clean_featrue)   
        if args.attack_position=="llm":
            adv_image = attack(image, question, answer_before_attack)   
        answer_after_attack = model.predict(adv_image, question)          
        vqa_result.append({
            "question_id": int(question_id.numpy()[0]), 
            "answer": answer_after_attack
            })
        adv_process.append({
            "question_id": int(question_id.numpy()[0]),
            "question": question[0], 
            "answer_before_attack": answer_before_attack,
            "answer_after_attack": answer_after_attack,
            "attack_success": answer_before_attack!=answer_after_attack
        })
        torchvision.utils.save_image(image, os.path.join(args.output_image_dir, "{}_clean.png".format(image_path[0])))
        torchvision.utils.save_image(adv_image, os.path.join(args.output_image_dir, "{}_adv.png".format(image_path[0])))
    
    save_result(vqa_result, args.output_dir, 'vqa_result')
    save_result(adv_process, args.output_dir, 'adv_process')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) \


if __name__ == '__main__':
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    
    # default config
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    # lvlm model config
    parser.add_argument('--lvlm_root', required=True, type=str)
    parser.add_argument('--lvlm', required=True, type=str, choices=["blip2_flan-t5-xl", "blip2_flan-t5-xxl", "blip2_opt-2.7b", "blip2_opt-6.7b", "iblip_flan-t5-xl", "iblip_flan-t5-xxl", "iblip_vicuna-7b", "iblip_vicuna-13b"])

    # dataset config
    parser.add_argument('--annotation_file', required=True, type=str)
    parser.add_argument('--image_root', required=True, type=str)

    # attack config
    parser.add_argument('--attack_method', type=str, required=True, choices=["pgd", "cw"])
    parser.add_argument('--attack_position', type=str, required=True, choices=["clip", "llm"])
    
    # pgd attack confg
    parser.add_argument('--pgd_steps', default=20, type=int)
    parser.add_argument('--pgd_lr', default=2, type=int)
    parser.add_argument('--pgd_eps', default=8, type=int)

    # cw attack config
    parser.add_argument('--cw_max_iter', type=int, default=50)
    parser.add_argument('--cw_const', type=float, default=0.005)
    parser.add_argument('--cw_lr', type=float, default=0.01)

    args = parser.parse_args()

    annotation_type = args.annotation_file.replace(".json", "")
    if args.attack_method=="pgd":
        args.output_dir = f"./output_dir/{args.lvlm}/{args.attack_method}_{args.pgd_lr}_{args.pgd_eps}_{args.pgd_steps}_{args.attack_position}_{annotation_type}"
    else:
        args.output_dir = f"./output_dir/{args.lvlm}/{args.attack_method}_{args.cw_lr}_{args.cw_const}_{args.cw_max_iter}_{args.attack_position}_{annotation_type}"
    args.output_image_dir = os.path.join(args.output_dir, "image")
    Path(args.output_image_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(os.path.join(args.output_dir, "vqa_result.json")):
        args.lvlm, args.lvlm_llm = args.lvlm.split("_")

        args.pgd_lr = float(args.pgd_lr)/255
        args.pgd_eps = float(args.pgd_eps)/255

        main(args)
    else:
        print(f"Already exists attack result in {args.output_dir}")