from GAN import *
from tqdm import tqdm
 
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters
    lr = 3e-4
    z_dim = 64
    image_dim = 28*28*1 # 784
    batch_size = 32
    num_epochs = 120

    disc = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    transforms = transforms1.Compose(
        [transforms1.ToTensor(), transforms1.Normalize((0.5,), (0.5,))]
    )
    dataset  = datasets.MNIST(root="dataset/", transform=transforms, download=True) # normalize the images between -1 and 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    writer_fake = SummaryWriter(f'runs/fake')
    writer_real = SummaryWriter(f'runs/real')
    step = 0

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(tqdm(loader)):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z))) <-> min -log(D(x)) - log(1 - D(G(z))
            noise = torch.randn(batch_size, z_dim).to(device) # gaussian noise is used to generate fake images because the gaussian distribution is the most common distribution in nature
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2 # Average the loss of real and fake images
            disc.zero_grad()
            lossD.backward(retain_graph=True) # Retain graph to avoid re-computation that is required for the generator
            opt_disc.step() # Update the weights of the discriminator

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)) <-> min -log(D(G(z)))
            output = disc(fake).view(-1) 
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step() # Update the weights of the generator

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "Mnist Real Images", img_grid_real, global_step=step
                    )
                    step += 1
    # save the model
    torch.save(gen.state_dict(), "gen.pth")
    torch.save(disc.state_dict(), "disc.pth")

def genrate_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 64
    image_dim = 28*28*1 # 784
    gen = Generator(z_dim, image_dim).to(device)
    gen.load_state_dict(torch.load("gen.pth"))
    gen.eval()
    noise = torch.randn(1, z_dim).to(device)
    img = gen(noise).reshape(1, 28, 28)
    img = img.detach().cpu().numpy()
    plt.imshow(img[0], cmap="gray")
    plt.show()

if __name__ == "__main__":
    #train()
    while True:
        genrate_image()