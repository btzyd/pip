import os
import json
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from .data_utils import pre_question

class vqa_dataset(Dataset):
    def __init__(self, annotation_file, image_dir):
        self.annotation = json.load(open(os.path.join("annotation", annotation_file), 'r'))
        self.image_dir = image_dir
        self.transforms = transforms.Compose([transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path_name = self.annotation[index]["image_name"]
        question = pre_question(self.annotation[index]["question"])
        question_id = self.annotation[index]["question_id"]
        image_path = os.path.join(self.image_dir, image_path_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        return image_path_name.split(".")[0], image, question, question_id
    
class vqa_imagefolder_dump_attention(Dataset):
    def __init__(self, image_dir):
        self.image_dir = os.path.join(image_dir, "image")
        self.annotation = os.listdir(self.image_dir)
        self.transforms = transforms.Compose([transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        image_name = self.annotation[index]
        image_path_name = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path_name).convert('RGB')
        image = self.transforms(image)
        is_clean = 0 if "clean" in image_name else 1
        return image_name, image, is_clean
