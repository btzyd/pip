import torch
from tqdm import tqdm
import torch.optim as optim

class CWL2AttackWithoutLabels():
    def __init__(self, model, cw_max_iter, cw_lr, cw_const):
        self.model = model
        self.cw_max_iter = cw_max_iter
        self.cw_const = cw_const
        self.cw_lr = cw_lr
        self.model.requires_grad_(False)


    def attack(self, images, question, answer):

        perturbation = self.cw_lr * torch.rand(images.shape).sign()
        perturbation = perturbation.to(images.device)
        perturbation.requires_grad=True
        
        optimizer = optim.Adam([perturbation], lr=self.cw_lr)
        images.requires_grad = False
        for _ in tqdm(range(self.cw_max_iter)):
            self.model.zero_grad()
            optimizer.zero_grad()
            image_adv = images + perturbation
            model_loss = self.model.get_loss(image_adv, question, answer)
            loss = -model_loss + self.cw_const * torch.norm(perturbation) ** 2
            loss.backward()
            optimizer.step()
            perturbation.data.clamp_(-1, 1)

        adv_images = images + perturbation
        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images
    
    def __call__(self, image, question, answer):
        image_adv = self.attack(image, question, answer)
        return image_adv