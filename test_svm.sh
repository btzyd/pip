# for table 1
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_clip_vqav2_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm

# for table 3
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_clip_imagenet_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_imagenet_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm

# for table 4
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/cw_0.01_0.005_50_llm_vqav2_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/cw_0.01_0.005_50_llm_imagenet_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm

# for Table 5
python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_2_20_llm_vqav2_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm
python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_4_20_llm_vqav2_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm
python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_16_20_llm_vqav2_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm

# for Table 6
# for lvlm in "blip2_flan-t5-xl" "blip2_flan-t5-xxl" "blip2_opt-2.7b" "blip2_opt-6.7b" "iblip_flan-t5-xl" "iblip_flan-t5-xxl" "iblip_vicuna-7b" "iblip_vicuna-13b"; do
#     python test_svm.py --work_dir output_dir/${lvlm}/pgd_2_8_20_llm_vqav2_1000 --svm_dir output_dir/${lvlm}/pgd_2_8_20_llm_vqav2_ref_5000/svm
# done

# for Table 7
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_clip_vqav2_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm --svm_alarm_num 2 --svm_total_num 3
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm --svm_alarm_num 2 --svm_total_num 3
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_clip_imagenet_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm --svm_alarm_num 3 --svm_total_num 3
# python test_svm.py --work_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_imagenet_1000 --svm_dir output_dir/iblip_vicuna-7b/pgd_2_8_20_llm_vqav2_ref_5000/svm --svm_alarm_num 2 --svm_total_num 3
