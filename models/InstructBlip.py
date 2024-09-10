import torch
import torch.nn as nn
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
from torchvision import transforms
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers.models.instructblip.modeling_instructblip import InstructBlipForConditionalGenerationModelOutput
from transformers.models.instructblip.configuration_instructblip import InstructBlipConfig
import os


class InstructBlip_PIP(nn.Module):
    def __init__(self, huggingface_root, lvlm_llm, attack_position=None):
        super(InstructBlip_PIP, self).__init__()

        assert lvlm_llm in ["vicuna-7b", "vicuna-13b", "flan-t5-xl", "flan-t5-xxl"], "lvlm_llm error"
        
        self.processor = InstructBlipProcessor.from_pretrained(os.path.join(huggingface_root, "instructblip-{}".format(lvlm_llm)))
        self.model = InstructBlipForConditionalGeneration.from_pretrained(os.path.join(huggingface_root, "instructblip-{}".format(lvlm_llm)))
        
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
        if self.attack_position=="clip":
            self.loss_func = nn.MSELoss()

    def get_attention(self, x, question):
        x = torch.clamp(x, min=0, max=1)
        batch_size = x.shape[0]
        inputs_processor = self.processor(text="Question: {} Short answer:".format(question))

        inputs = dict(pixel_values=self.normalizer(x).to(self.device))
        inputs["input_ids"] = torch.tensor(inputs_processor.input_ids).view(batch_size, -1).to(self.device)
        inputs["qformer_input_ids"] = torch.tensor(inputs_processor.qformer_input_ids).view(batch_size, -1).to(self.device)
       
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
        inputs["qformer_input_ids"] = torch.tensor(inputs_processor_result.qformer_input_ids).view(batch_size, -1).to(self.device)
        ids = self.model.generate(**inputs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    
    def get_clip(self, x):
        x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x).to(self.device))
        clip_featrue = self.model.vision_model(**inputs).last_hidden_state
        return clip_featrue
     

    def get_loss(self, x , question, answer):

        x = torch.clamp(x, min=0, max=1)
        batch_size = x.shape[0]
        inputs_processor_result = self.processor(text="Question: {} Short answer:".format(question))
        inputs = dict(pixel_values=self.normalizer(x).to(self.device))
        inputs["input_ids"] = torch.tensor(inputs_processor_result.input_ids).view(batch_size, -1).to(self.device)
        inputs["qformer_input_ids"] = torch.tensor(inputs_processor_result.qformer_input_ids).view(batch_size, -1).to(self.device)
        if self.attack_position=="clip":
            adv_featrue = self.get_clip(x)
            answer.requires_grad_(False)
            return self.loss_func(answer, adv_featrue)
        elif self.attack_position=="llm":
            label = torch.tensor(self.processor(text=answer).input_ids).view(batch_size, -1)
            inputs["labels"] = label.to(self.device)
            outputs = self.model(**inputs, return_dict=True)
            return outputs.loss
