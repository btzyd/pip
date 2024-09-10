# dataset="vqav2_ref_5000"
# lvlm="iblip_vicuna-7b"
# method="pgd"
# position="llm"
# for pgd_eps in "2" "4" "8" "16";do
#     python -m torch.distributed.run --nproc_per_node=8 attack_vqa.py --lvlm_root /root/huggingface_model --lvlm ${lvlm} --annotation_file ${dataset}.json --image_root ./image_dir/${dataset} --attack_method ${method} --attack_position ${position} --pgd_eps ${pgd_eps}
# done

dataset="vqav2_1000"
lvlm="iblip_vicuna-7b"
method="pgd"
position="llm"
for pgd_eps in "2" "4" "8" "16";do
    python -m torch.distributed.run --nproc_per_node=8 attack_vqa.py --lvlm_root /root/huggingface_model --lvlm ${lvlm} --annotation_file ${dataset}.json --image_root ./image_dir/${dataset} --attack_method ${method} --attack_position ${position} --pgd_eps ${pgd_eps}
done

# dataset="vqav2_ref_5000"
# lvlm="iblip_vicuna-7b"
# method="cw"
# position="llm"
# python -m torch.distributed.run --nproc_per_node=8 attack_vqa.py --lvlm_root /root/huggingface_model --lvlm ${lvlm} --annotation_file ${dataset}.json --image_root ./image_dir/${dataset} --attack_method ${method} --attack_position ${position}


# dataset="vqav2_ref_5000"
# method="pgd"
# position="llm"
# for lvlm in "blip2_flan-t5-xl" "blip2_flan-t5-xxl" "blip2_opt-2.7b" "blip2_opt-6.7b" "iblip_flan-t5-xl" "iblip_flan-t5-xxl" "iblip_vicuna-13b"; do
#     python -m torch.distributed.run --nproc_per_node=8 attack_vqa.py --lvlm_root /root/huggingface_model --lvlm ${lvlm} --annotation_file ${dataset}.json --image_root ./image_dir/${dataset} --attack_method ${method} --attack_position ${position}
# done

# for dataset in "vqav2_1000" "imagenet_1000"; do
#     for lvlm in "blip2_flan-t5-xl" "blip2_flan-t5-xxl" "blip2_opt-2.7b" "blip2_opt-6.7b"; do
#         for method in "pgd" "cw"; do
#             for position in "llm"; do
#                 python -m torch.distributed.run --nproc_per_node=8 attack_vqa.py --lvlm_root /root/huggingface_model --lvlm ${lvlm} --annotation_file ${dataset}.json --image_root ./image_dir/${dataset} --attack_method ${method} --attack_position ${position}
#             done
#         done
#     done
# done

# for dataset in "vqav2_1000" "imagenet_1000"; do
#     for lvlm in "iblip_flan-t5-xl" "iblip_flan-t5-xxl" "iblip_vicuna-7b" "iblip_vicuna-13b"; do
#         for method in "pgd" "cw"; do
#             for position in "llm" "clip"; do
#                 python -m torch.distributed.run --nproc_per_node=8 attack_vqa.py --lvlm_root /root/huggingface_model --lvlm ${lvlm} --annotation_file ${dataset}.json --image_root ./image_dir/${dataset} --attack_method ${method} --attack_position ${position}
#             done
#         done
#     done
# done