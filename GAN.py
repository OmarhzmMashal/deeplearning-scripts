import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer_fake = SummaryWriter("/home/omar/Desktop/images/runs/fake")
writer_real = SummaryWriter("/home/omar/Desktop/images/runs/real")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = torch.nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = torch.nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)
        return x


device='cuda'
# hyper paramaters
lr=0.001
epochs=10
batch_size=32
noise_dim = 64
step=0
# init models
fixed_noise = torch.randn((batch_size, noise_dim)).to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optim_gen = torch.optim.Adam(generator.parameters(), lr=lr)
optim_disc = torch.optim.Adam(discriminator.parameters(), lr=lr)
loss = nn.BCELoss()
# Download the MNIST Dataset
dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = transforms.ToTensor())

# for training
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = batch_size,
                                     shuffle = True)

for epoch in tqdm(range(epochs)):
    for batch_idx, (images_real, _) in enumerate(loader):

        images_real = images_real.view(-1, 28*28).to(device) # real images - x
        images_noise = torch.randn(batch_size, noise_dim).to(device) # noise images - z
        images_fake = generator(images_noise) # fake/constructed images G(z)

        # binary cross entropy: minmizes --> y_true log(y_pred) + (1 - y_true) log(1 - y_pred)

        # discriminator
        disc_images_real = discriminator(images_real).view(-1) # D(x)
        loss_disc_real = loss(disc_images_real, torch.zeros_like(disc_images_real)) # -> min (1 - log(D(x))) make it go to 1

        disc_images_fake = discriminator(images_fake).view(-1) # D(G(z))
        loss_disc_fake = loss(disc_images_fake, torch.ones_like(disc_images_real)) # -> min  log(D(G(z))) make it go to 0

        # max -> log(D(x)) + log(1-D(G(Z)))
        loss_disc_total = (loss_disc_real+loss_disc_fake)/2
        optim_disc.zero_grad()
        loss_disc_total.backward(retain_graph=True)
        optim_disc.step()

        # generator
        disc_images_fake2 = discriminator(images_fake).view(-1) # D(G(z))
        loss_gen = loss(disc_images_fake2, torch.zeros_like(disc_images_fake)) # min -> log(1 - D(G(z))) make it go to 1
        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} Loss D: {loss_disc_total:.4f}, loss G: {loss_gen:.4f}")
            with torch.no_grad():

                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                real = images_real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
print("Done")
