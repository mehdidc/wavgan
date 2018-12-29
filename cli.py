import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from transforms import Scale, Compose, PadTrim
from data import Dataset
from clize import run
def train(
        *,
        data_folder='data',
        nb=None,
        lr=1e-04, weight_decay=1e-04, beta1=0.5, beta2=.999, 
        batch_size=32, epochs=1000,
        input_dim=1,
        max_len=500,
        log_interval=50,
        cppn=True,
        cuda=False):
    # define the optimizers.
    if cppn:
        output_dim = 1
    else:
        output_dim = max_len
    generator = Generator(input_dim=input_dim, output_dim=output_dim)
    discriminator = Discriminator(
        input_dim=1, output_dim=1, input_size=max_len)
    if cuda:
        discriminator = discriminator.cuda()
        generator = generator.cuda()
    generator_optimizer = optim.Adam(
        generator.parameters(), lr=lr, betas=(beta1, beta2),
        weight_decay=weight_decay
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=lr, betas=(beta1, beta2),
        weight_decay=weight_decay
    )

    # prepare the model and statistics.
    generator.train()
    discriminator.train()
    epoch_start = 1
    transform = Compose([
        PadTrim(max_len=max_len),
        Scale(),
    ])
    dataset = Dataset(data_folder, transform=transform, nb=nb)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    nb_iter = 0
    for epoch in range(epoch_start, epochs+1):
        for batch_index, data in enumerate(dataloader):
            #for p in discriminator.parameters():
            #    p.data.clamp_(-0.1, 0.1)
            x = data
            x = x.cuda() if cuda else x
            discriminator.zero_grad()
            dreal = discriminator(x).mean()

            l = torch.randn(batch_size, generator.input_dim - 1)
            t = torch.linspace(-1, 1, discriminator.input_size)
            
            if cppn:
                z = torch.zeros(batch_size, discriminator.input_size, generator.input_dim)
                if cuda:
                    z = z.cuda()
                z[:, :, 1:] = l.view(l.size(0), 1, l.size(1)).expand(l.size(0), z.size(1), l.size(1))
                z[:, :, 0] = t
                z = z.contiguous()
                z_ = z.view(z.size(0) * z.size(1), -1)
                z_ = z_.contiguous()
                xfake = generator(z_)
                xfake = xfake.view(z.size(0), 1, z.size(1))
            else:
                z = torch.randn(batch_size, generator.input_dim)
                if cuda:
                    z = z.cuda()
                xfake = generator(z)
                xfake = xfake.view(xfake.size(0), 1, xfake.size(1))
            if nb_iter % 2 == 0:
                dfake = discriminator(xfake).mean()
                discr_loss = dfake - dreal 
                discr_loss.backward(retain_graph=True)
                discriminator_optimizer.step()

                generator.zero_grad()
                dfake = discriminator(xfake).mean()
                gen_loss = -dfake
                gen_loss.backward()
                generator_optimizer.step()
            if nb_iter % log_interval == 0:
                print(f'niter: {nb_iter:05d} gen_loss: {gen_loss.item():.4f} discr_loss: {discr_loss.item():.4f}')
                x = x.detach().cpu().numpy()
                xfake = xfake.detach().cpu().numpy()
                signal = x[0:3, 0].T
                fake_signal = xfake[0:3, 0].T
                fig = plt.figure()
                plt.plot(signal, color='blue', label='true')
                plt.plot(fake_signal, color='orange', label='fake')
                #plt.legend()
                plt.savefig('out.png')
                plt.close(fig)
            nb_iter += 1

if __name__ == '__main__':
    run(train)
