## PIP: Detecting Adversarial Examples in Large Vision-Language Models via Attention Patterns of Irrelevant Probe Questions

This is a PyTorch implementation of the [PIP paper](https://arxiv.org/abs/2409.05076):

## Preparing the environment, code, data and model

1. Prepare the environment.

Creating a python environment and activate it via the following command.

```bash
cd PIP
conda create -n pip python=3.10
conda activate pip
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.31.0
pip install scikit-learn==1.3.2
```

2. Clone this repository.

```bash
git clone https://github.com/btzyd/pip.git
```

3. Prepare the dataset COCO, ImageNet and VQA v2.

In the annotation folder, we have generated 5,000 and 1,000 questions from [VQA v2](https://visualqa.org/download.html) to form [vqav2_ref_5000.json](annotation/vqav2_ref_5000.json) and [annotation/vqav2_1000.json](annotation/vqav2_1000.json). We also used ImageNet's labels to generate 1000 questions in the form of "Is there a/an {label}?", as [imagenet_1000.json](annotation/imagenet_1000.json).

As for images, you can get [COCO-val2014](http://images.cocodataset.org/zips/val2014.zip), [ImageNet(ILSVRC2012)](https://www.image-net.org/challenges/LSVRC/2012/index.php) from their official websites, respectively. We also provide the image files corresponding to our generated [vqav2_ref_5000](annotation/vqav2_ref_5000.json), [vqav2_1000](annotation/vqav2_1000.json) and [imagenet_1000](annotation/imagenet_1000.json), you can download [pip_image.zip](https://cloud.tsinghua.edu.cn/f/341402a2a1104a379b28/) and extract it to ``./image_dir/``. 

4. Download BLIP-2 and InstructBLIP Models

You can download the BLIP-2 and InstructBLIP models from the following HuggingFace links. Of course, you can also download the model while loading it in python code, though the download may be unstable. We recommend that you download the model first and then run the python code to load it from local directories.

- [instructblip-vicuna-7b](https://huggingface.co/Salesforce/instructblip-vicuna-7b)
- [instructblip-vicuna-13b](https://huggingface.co/Salesforce/instructblip-vicuna-13b)
- [instructblip-flan-t5-xl](https://huggingface.co/Salesforce/instructblip-flan-t5-xl)
- [instructblip-flan-t5-xxl](https://huggingface.co/Salesforce/instructblip-flan-t5-xxl)
- [blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [blip2-opt-6.7b](https://huggingface.co/Salesforce/blip2-opt-6.7b)
- [blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
- [blip2-flan-t5-xxl](https://huggingface.co/Salesforce/blip2-flan-t5-xxl)

## Run the PIP code

### Running the adversarial attack to generate adversarial examples

The adversarial attack can be executed with the following command.

```python
python -m torch.distributed.run --nproc_per_node=8 attack_vqa.py --lvlm_root /root/huggingface_model --lvlm iblip_vicuna-7b --annotation_file vqav2_ref_5000.json --image_root ./image_dir/vqav2_ref_5000 --attack_method pgd --attack_position llm
```

You can perform adversarial attacks on other datasets by replacing vqav2_ref_5000 with vqav2_1000 and imagenet_1000.

The meaning of the parameters are as follows:

- ``nproc_per_node``: Our code supports the use of multiple GPUs, specify the number of GPUs to use with this parameter.
- ``lvlm_root``: Specify the directory where models downloaded from hugging face are stored.
- ``lvlm``: Select the LVLM you want to use. Currently, a total of eight models are supported, including BLIP-2 and InstructBLIP, as described above. That is, ``blip2_flan-t5-xl``, ``blip2_flan-t5-xxl``, ``blip2_opt-2.7b``, ``blip2_opt-6.7b``, ``iblip_flan-t5-xl``, ``iblip_flan-t5-xxl``, ``iblip_vicuna-7b``, ``iblip_vicuna-13b``.
- ``annotation_file``: Specify the dataset to be used, three datasets are provided in our code, ``vqav2_ref_5000.json``, ``vqav2_1000.json`` and ``imagenet_1000.json``.
- ``image_root``: Specifies the directory where the image is located. If you downloaded the image from COCO and ImageNet website, you can specify the directory as val2014 for COCO and val for ImageNet. if you got the image from the zip package we provided, the directory is ``image_dir/vqav2_ref_5000``, ``image_dir/vqav2_1000`` and ``image_dir/imagenet_1000``.
- ``attack_method``: Methods for adversarial attacks, currently supporting ``pgd`` and ``cw``.
- ``attack_position``: The features of the adversarial attack, which currently support the attacks on CLIP output feature ``clip``, and the attacks on LLM output ``llm``.
- ``pgd_steps``, ``pgd_lr``, ``pgd_eps`` and ``cw_max_iter``, ``cw_const``, ``cw_lr``: Parameters of PGD and CW attacks. For PGD, ``pgd_lr=2`` represents an attack step of 2/255, and for CW, ``cw_lr=0.01`` represents an attack step of 2.25/255=0.01.

We also provide a script to perform adversarial attacks in batch.

```bash
bash attack_vqa.sh
```

### Extracting the cross-modal attention for clean examples and adversarial examples

We have provided 7 irrelevant probe questions in the [question_list.json](question_list.json), you can also modify this file as needed. 

The command to extract cross-modal attention is as follows, which will create the attention_map directory in the ``image_dir`` directory, where each file is the cross-modal attention of an image.

```python
python -m torch.distributed.run --nproc_per_node=8 dump_attention.py --lvlm_root /root/huggingface_model --lvlm iblip_vicuna-7b --image_dir ./output_dir/iblip_vicuna-7b/attack_llm_vqav2_ref_5000 --question_list ./question_list.json
```

The meaning of the parameters are as follows:

- ``image_dir``: The directory to store images of clean examples and adversarial examples.
- ``question_list``: Specify a list of irrelevant probe questions.

We also provide a script to perform adversarial attacks in batch.

```bash
bash dump_attention.sh
```

### Training an SVM using the cross-modal attention on vqav2_ref_5000

```python
python train_svm.py --work_dir output_dir/iblip_vicuna-7b/attack_llm_vqav2_ref_5000 --question_index -1
```

The meaning of the parameters are as follows:

- ``work_dir``: The directory to store cross-modal attention of clean examples and adversarial examples.
- ``question_index``: Specifies which question from [question_list.json](question_list.json) to train the SVM, if ``question_index=-1`` is specified, the corresponding SVMs are trained for all probe questions.

We also provide a script to perform adversarial attacks in batch.

```bash
bash train_svm.sh
```

### Testing an SVM using the cross-modal attention on vqav2_1000 and imagenet_1000

```python
python test_svm.py --work_dir output_dir/iblip_vicuna-7b/attack_llm_vqav2_1000 --svm_dir output_dir/iblip_vicuna-7b/attack_llm_vqav2_ref_5000/svm --svm_index 0
```

The meaning of the parameters are as follows:

- ``work_dir``: The directory to store cross-modal attention of clean examples and adversarial examples.
- ``svm_dir``: Specify the directory where the SVM is stored.
- ``svm_alarm_num``: Specifies how many SVMs alarm when the sample is considered to be an adversarial example, default is 1.
- ``svm_total_num``: Specify how many SVMs are used in total, default is 1.

```bash
bash test_svm.sh
```

## Citation

ACM Reference Format:
Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, and Yu Wang. 2024. PIP: Detecting Adversarial Examples in Large Vision-Language Models via Attention Patterns of Irrelevant Probe Questions. In Proceedings of the 32nd ACM International Conference on Multimedia (MM ’24), October 28-November 1, 2024, Melbourne, VIC, Australia. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3664647.3685510