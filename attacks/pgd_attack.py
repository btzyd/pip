import torch
from tqdm import tqdm

class PGD_Attack():
    def __init__(self, model, pgd_eps, pgd_lr, pgd_steps):
        self.model = model
        self.pgd_eps = pgd_eps
        self.pgd_lr = pgd_lr
        self.pgd_steps = pgd_steps
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, image, question, answer):
        
        image_adv = image.detach().clone() + (self.pgd_lr * torch.rand(image.shape).sign()).to(image.device)

        for _ in tqdm(range(self.pgd_steps)):
            self.model.zero_grad()

            image_adv.requires_grad = True
            loss = self.model.get_loss(image_adv, question, answer)
            loss.backward()
            
            grad = image_adv.grad.detach().sign()
            image_adv = image_adv + self.pgd_lr * grad

            image_adv = image + torch.clamp(image_adv - image, min=-self.pgd_eps, max=self.pgd_eps)
            image_adv = image_adv.detach()
            image_adv = torch.clamp(image_adv, min=0, max=1)

        return image_adv

    def __call__(self, image, question, answer):
        image_adv = self.forward(image, question, answer)
        return image_adv