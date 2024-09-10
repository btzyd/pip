for lvlm in "iblip_vicuna-7b"; do
    for dir in $(find ./output_dir/${lvlm} -mindepth 1 -maxdepth 1 -type d); do
        python -m torch.distributed.run --nproc_per_node=8 dump_attention.py --lvlm_root /root/huggingface_model --lvlm ${lvlm} --image_dir ${dir} --question_list ./question_list.json
    done
done
