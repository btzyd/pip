for dir in $(find ./output_dir/ -mindepth 2 -maxdepth 2 -name *vqav2_ref_5000 -type d); do
    python train_svm.py --work_dir ${dir} --question_index -1
done