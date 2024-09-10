for lvlm in "blip2_flan-t5-xl" "blip2_flan-t5-xxl" "blip2_opt-2.7b" "blip2_opt-6.7b" "iblip_flan-t5-xl" "iblip_flan-t5-xxl" "iblip_vicuna-7b" "iblip_vicuna-13b"; do
    for dir in $(find ./output_dir/${lvlm} -mindepth 1 -maxdepth 1 -type d); do
        python -m torch.distributed.run --nproc_per_node=8 dump_attention.py --lvlm_root /root/huggingface_model --lvlm ${lvlm} --image_dir ${dir} --question_list ./question_list.json
    done
done
