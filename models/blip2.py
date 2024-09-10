import torch
import torch.nn as nn
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
from torchvision import transforms
from PIL import Image
import re
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput
from transformers.models.blip_2.configuration_blip_2 import Blip2Config

import os

class Blip2_PIP(nn.Module):
    def __init__(self, huggingface_root, lvlm_llm, attack_position=None):
        super(Blip2_PIP, self).__init__()
        assert lvlm_llm in ["opt-2.7b", "opt-6.7b", "flan-t5-xl", "flan-t5-xxl"], "lvlm_llm error"
        
        self.processor = Blip2Processor.from_pretrained(os.path.join(huggingface_root, "blip2-{}".format(lvlm_llm)))
        self.model = Blip2ForConditionalGeneration.from_pretrained(os.path.join(huggingface_root, "blip2-{}".format(lvlm_llm)))

        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)

        OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
        self.normalizer = transforms.Compose([transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),])
        '''
            To disable the following code in generate
            if self.config.text_config.architectures[0] == "LLaMAForCausalLM":
                outputs[outputs == 0] = 2
        '''
        
        self.model.config.text_config.architectures[0]="LLaMAForCausalLM_"

        self.lvlm_llm = lvlm_llm
        self.attack_position = attack_position

    def get_attention(self, x, question):
        x = torch.clamp(x, min=0, max=1)
        batch_size = x.shape[0]
        inputs_processor = self.processor(text="Question: {} Short answer:".format(question))

        inputs = dict(pixel_values=self.normalizer(x).to(self.device))
        inputs["input_ids"] = torch.tensor(inputs_processor.input_ids).view(batch_size, -1).to(self.device)
        outputs = self.model.generate(**inputs, output_attentions=True, output_hidden_states=True, output_scores=True, return_dict_in_generate=True)
        if "flan" in self.lvlm_llm:
            attention = torch.stack(outputs.cross_attentions[1], dim=0)
            attention = attention[:,0,:,0,:32].cpu()
        else:
            attention = torch.cat(outputs.attentions[1], dim=0).squeeze(2)[:,:,:32].cpu()
        
        return attention # [layer, head, token] 
        
    def predict(self, x, question):
        x = torch.clamp(x, min=0, max=1)
        batch_size = x.shape[0]
        inputs = dict(pixel_values=self.normalizer(x).to(self.device))
        inputs_processor_result = self.processor(text="Question: {} Short answer:".format(question))
        inputs["input_ids"] = torch.tensor(inputs_processor_result.input_ids).view(batch_size, -1).to(self.device)
        ids = self.model.generate(**inputs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


    def get_loss(self, x, question, answer):
        x = torch.clamp(x, min=0, max=1)
        batch_size = x.shape[0]
        inputs_processor_result = self.processor(text="Question: {} Short answer:".format(question))    
        inputs = dict(pixel_values=self.normalizer(x).to(self.device))
        inputs["input_ids"] = torch.tensor(inputs_processor_result.input_ids).view(batch_size, -1).to(self.device)
        label = torch.tensor(self.processor(text=answer).input_ids).view(batch_size, -1)
        inputs["labels"] = label.to(self.device)
        outputs = self.model(**inputs, return_dict=True)
        return outputs.loss
