def create_mosaic(data, m=10, n=10, padding=1):

    ind = np.random.randint(0, len(data))
    img = data[ind][0]

    h, w = img.shape

    out = np.zeros((m * (h + padding) - padding, n * (w + padding) - padding))
    for i in range(m):

        i_start = i * h
        if i > 0:
            i_start += i * padding
        i_end = i_start + h

        for j in range(n):

            j_start = j * h
            if j > 0:
                j_start += j * padding
            j_end = j_start + w

            ind = np.random.randint(0, len(data))
            out[i_start:i_end, j_start:j_end] = data[ind][0].detach().numpy()

    return out


class NormalNLLLoss:
    def __call__(self, x, mu, log_var):
        dx = x - mu
        logli = 0.5 * (np.log(2 * np.pi) + log_var + dx * dx / log_var.exp())
        nll = logli.sum(1).mean()
        return nll


################################################################################
# Discriminator
################################################################################


class Generator(nn.Module):
    """
    Generator module that generates particles images from Gaussian noise
    using a convolutional neural network for upsampling.
    """

    def __init__(self, latent_dim, n_filters=32):
        """
        Arguments:
            latent_dim: The size of the noise vector used as input for the
                generator
            n_filters: Number of filter in the second-to-last layer.
        """

        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # 4 x 4 -> 8 x 8
            nn.ConvTranspose2d(latent_dim, n_filters * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # 8 x 8 -> 16 x 16
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(True),
            # 32 x 32 -> 32 x 32
            nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # Output
            nn.ConvTranspose2d(n_filters, n_filters, 4, 2, 0, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, 1, 3, 1, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)

    def generate(self, n=1):
        z = torch.randn(n, self.latent_dim, 1, 1, device=self.device)
        return self.forward(z)


class InfoGenerator(nn.Module):
    """
    Generator module that generates particles images from Gaussian noise
    using a convolutional neural network for upsampling.
    """

    def __init__(self, n_inc, n_cat_dim=10, n_con_dim=1, n_filters=32):
        """
        Arguments:
            latent_dim: The size of the noise vector used as input for the
                generator
            n_filters: Number of filter in the second-to-last layer.
        """

        super(InfoGenerator, self).__init__()
        self.n_inc = n_inc
        self.latent_dim = n_inc + n_cat_dim + n_con_dim
        self.n_cat_dim = n_cat_dim
        self.n_con_dim = n_con_dim
        self.main = nn.Sequential(
            # 4 x 4 -> 8 x 8
            nn.ConvTranspose2d(self.latent_dim, n_filters * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # 8 x 8 -> 16 x 16
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(True),
            # 32 x 32 -> 32 x 32
            nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # Output
            nn.ConvTranspose2d(n_filters, n_filters, 4, 2, 0, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, 1, 3, 1, 0, bias=False),
            nn.Tanh(),
        )

        # Keep track of class frequencies.
        self.frequencies = []
        freqs = torch.ones(self.n_cat_dim)
        freqs /= freqs.sum()
        self.update_frequencies(freqs)

    def update_frequencies(self, freqs):
        tot = freqs.sum()
        freqs = freqs.clamp(0.01 * tot, tot)
        self.frequencies += [freqs]
        freqs = freqs.div(freqs.sum())
        self.indices = torch.zeros(512, dtype=torch.long)
        i = 0
        for k, f in enumerate(freqs):
            j = i + int(f * 512)
            self.indices[i:j] = k
            i = j

    def forward(self, input):
        return self.main(input)

    def generate(self, n=1, cat=None, q=None):
        z = torch.zeros(n, self.latent_dim, 1, 1, device=self.device)
        z[:, : self.n_inc, 0, 0] = torch.randn(n, self.n_inc, device=self.device)

        if cat is None:
            inds = torch.randperm(512)
            inds = self.indices[inds[:n]].to(self.device) + self.n_inc
        else:
            inds = (
                cat * torch.ones((n,), device=self.device, dtype=torch.long)
                + self.n_inc
            )
        z[np.arange(0, n), inds, 0, 0] = 1.0

        if q is None:
            q = torch.rand(n, self.n_con_dim)
        else:
            q = q

        z[:, self.n_inc + self.n_cat_dim :, 0, 0] = q

        c = inds - self.n_inc
        return self.forward(z), c, q


################################################################################
# Discriminator
################################################################################


class Discriminator(nn.Module):
    """
    The discriminator that learns to distinguish synthetic particle imges
    from real ones.
    """

    def __init__(self, n_filters=32, output_filters=32):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 32 x 32 -> 16 x 16
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 16 -> 8 x 8
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 8 -> 4 x 4
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 x 4 -> 2 x 2
            nn.Conv2d(n_filters * 4, output_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_filters),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*8) x 4 x 4
            # 2 x 2 -> 1 x 1
            nn.Conv2d(output_filters, output_filters, 2, 1, 0, bias=False),
            nn.Conv2d(output_filters, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


class InfoDiscriminator(nn.Module):
    """
    The discriminator that learns to distinguish synthetic particle imges
    from real ones.
    """

    def __init__(self, n_cat_dim=10, n_con_dim=1, n_filters=32):
        super(InfoDiscriminator, self).__init__()

        self.n_cat_dim = n_cat_dim

        self.body = nn.Sequential(
            # 32 x 32 -> 16 x 16
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 16 -> 8 x 8
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 8 -> 4 x 4
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 x 4 -> 2 x 2
            nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (n_filters*8) x 4 x 4
        # 2 x 2 -> 1 x 1

        self.head_dis = nn.Sequential(
            nn.Conv2d(n_filters * 8, 1, 2, 1, 0, bias=False), nn.Sigmoid()
        )

        self.head_c = nn.Sequential(
            nn.Conv2d(n_filters * 8, 128, 2, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, n_cat_dim, 1, 1, 0, bias=False),
            nn.LogSoftmax(dim=1),
        )

        self.head_q = nn.Sequential(
            nn.Conv2d(n_filters * 8, 128, 2, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head_q_mean = nn.Conv2d(128, n_con_dim, 1, 1, 0, bias=False)
        self.head_q_var = nn.Conv2d(128, n_con_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        y = self.body(x)
        q = self.head_q(y)
        q_mean = self.head_q_mean(q)
        q_log_var = self.head_q_var(q)
        return self.head_dis(y), self.head_c(y), q_mean, q_log_var


class Gan:
    def __init__(
        self,
        latent_dim=100,
        n_filters_discriminator=32,
        n_filters_generator=32,
        features=32,
        device=None,
        optimizer="adam",
    ):

        self.latent_dim = latent_dim
        self.n_filters_discriminator = n_filters_discriminator
        self.n_filters_generator = n_filters_generator
        self.features = features
        self.device = device
        self.gan_type = "standard"
        self.optimizer = optimizer

        #
        # The device
        #

        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == "gpu" or device == torch.device("cuda:0"):
            self.device = torch.device("cuda:0")
        elif device == "cpu" or device == torch.device("cpu"):
            self.device = torch.device("cpu")
        else:
            raise Exception("Unknown device")

        self.generator = Generator(latent_dim, n_filters_generator)
        self.discriminator = Discriminator(
            n_filters=n_filters_discriminator, output_filters=features
        )
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.generator.device = self.device

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        for l in self.generator.children():
            weights_init(l)

            for l in self.discriminator.children():
                weights_init(l)

        self.generator_losses = []
        self.discriminator_losses = []
        self.image_list = []
        self.input_list = []

        #
        # Create optimizers
        #

        beta1 = 0.5
        lr_gen = 0.0002
        lr_dis = 0.0002
        if optimizer == "adam":
            self.optimizer_dis = torch.optim.Adam(
                self.discriminator.parameters(), lr=lr_dis, betas=(beta1, 0.999)
            )
            self.optimizer_gen = torch.optim.Adam(
                self.generator.parameters(), lr=lr_gen, betas=(beta1, 0.999)
            )
        elif optimizer == "sgd":
            self.optimizer_dis = torch.optim.SGD(
                self.discriminator.parameters(), lr=lr_dis
            )
            self.optimizer_gen = torch.optim.SGD(self.generator.parameters(), lr=lr_gen)
        else:
            raise Exception("Unknown optimizer type.")

        # Random input to track progress
        self.fixed_noise = torch.randn(
            64, self.generator.latent_dim, device=self.device
        )

        self.criterion = nn.BCELoss()

    def train(self, dataloader, lr_gen=0.0002, lr_dis=0.0002, noise=0.1):

        self.optimizer_gen.lr = lr_gen
        self.optimizer_dis.lr = lr_dis
        self.discriminator.to(self.device)
        self.generator.to(self.device)

        real_label = 0.9
        fake_label = 0
        iters = 0

        for i, data in enumerate(dataloader, 0):

            self.discriminator.zero_grad()

            real = data.to(self.device)
            real = real + noise * torch.randn(real.size(), device=self.device)
            real = torch.clamp(real, -1.0, 1.0)

            bs = real.size(0)
            label = torch.full((bs,), real_label, device=self.device)

            # Forward pass real batch through D
            output = self.discriminator(real).view(-1)
            err_dis_real = self.criterion(output, label)
            err_dis_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            fake = self.generator.generate(bs)
            fake = fake + noise * torch.randn(real.size(), device=self.device)
            fake = torch.clamp(fake, -1.0, 1.0)

            output = self.discriminator(fake.detach()).view(-1)
            label.fill_(fake_label)
            err_dis_fake = self.criterion(output, label)
            err_dis_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            err_dis = err_dis_real + err_dis_fake
            self.optimizer_dis.step()

            #
            # Train generator
            #

            self.generator.zero_grad()
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.discriminator(fake).view(-1)
            errG = self.criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()
            self.optimizer_gen.step()

            # Check how the generator is doing by saving G's output on fixed_noise
            self.fixed_noise.to(self.device)
            if iters % 500 == 0:
                with torch.no_grad():
                    fake = (
                        self.generator(self.fixed_noise.view(-1, 100, 1, 1))
                        .detach()
                        .cpu()
                    )
                    self.image_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                    )
                    self.input_list.append((real, fake))

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        i,
                        len(dataloader),
                        err_dis.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Save Losses for plotting later
            self.generator_losses.append(errG.item())
            self.discriminator_losses.append(err_dis.item())
            iters += 1

    def get_features(self, x):
        for l in next(self.discriminator.children()).children():
            if x.size()[-1] > 1:
                x = l(x)
            else:
                break
        return x

    def save(self, path):
        torch.save(
            {
                "latent_dim": self.latent_dim,
                "n_filters_discriminator": self.n_filters_discriminator,
                "n_filters_generator": self.n_filters_generator,
                "features": self.features,
                "device": self.device,
                "optimizer": self.optimizer,
                "discriminator_state": self.discriminator.state_dict(),
                "generator_state": self.generator.state_dict(),
                "discriminator_opt_state": self.optimizer_dis.state_dict(),
                "generator_opt_state": self.optimizer_gen.state_dict(),
                "discriminator_losses": self.discriminator_losses,
                "generator_losses": self.generator_losses,
                "gan_type": self.gan_type,
                "image_list": self.image_list,
                "input_list": self.input_list,
            },
            path,
        )

    @staticmethod
    def load(path):

        state = torch.load(path)
        print(state.keys())

        keys = [
            "latent_dim",
            "n_filters_discriminator",
            "n_filters_generator",
            "features",
            "device",
            "optimizer",
        ]
        kwargs = dict([(k, state[k]) for k in keys])

        if state["gan_type"] == "standard":
            gan = Gan(**kwargs)
        else:
            gan = WGan(**kwargs)

        try:
            gan.discriminator.load_state_dict(state["discriminator_state"])
            gan.optimizer_dis.load_state_dict(state["discriminator_opt_state"])
            gan.generator.load_state_dict(state["generator_state"])
            gan.optimizer_gen.load_state_dict(state["generator_opt_state"])
        except:
            pass

        gan.discriminator_losses = state["discriminator_losses"]
        gan.generator_losses = state["generator_losses"]
        gan.gan_type = state["gan_type"]
        gan.image_list = state["image_list"]
        gan.input_list = state["input_list"]

        return gan


class WGan(Gan):
    def __init__(
        self,
        latent_dim=100,
        n_filters_discriminator=32,
        n_filters_generator=32,
        features=32,
        device=None,
        optimizer="rmsprop",
        c=0.02,
        n_critic=5,
    ):

        super(WGan, self).__init__(
            latent_dim,
            n_filters_discriminator=n_filters_discriminator,
            n_filters_generator=n_filters_generator,
            features=features,
            device=device,
        )

        modules_dis = list(next(iter(self.discriminator.children())).children())
        print(modules_dis)
        self.discriminator.main = nn.Sequential(*modules_dis[:-1])
        self.discriminator.to(self.device)

        self.gan_type = "wasserstein"
        alpha = 0.00005

        self.optimizer = optimizer

        if not optimizer == "rmsprop":
            # raise Exception("Only rmsprop supported for WGAN.")
            pass

        self.optimizer_gen = torch.optim.RMSprop(self.generator.parameters(), lr=alpha)
        self.optimizer_dis = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=alpha
        )
        self.c = c
        self.n_critic = n_critic

    def train(self, dataloader, lr_gen=0.0002, lr_dis=0.0002, noise=0.1):

        self.optimizer_gen.lr = lr_gen
        self.optimizer_dis.lr = lr_dis

        self.discriminator.to(self.device)
        self.generator.to(self.device)

        iters = 0
        one = torch.FloatTensor([1])
        mone = one * -1
        one, mone = one.to(self.device), mone.to(self.device)
        bs = dataloader.batch_size

        for i, data in enumerate(dataloader, 0):

            # Turn on gradients
            for p in self.discriminator.parameters():
                p.requires_grad = True
            # Clip weights
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.c, self.c)
            # Remove old gradients
            self.discriminator.zero_grad()

            #
            # Real image
            #

            # Add noise to data
            real = data.to(self.device)
            real = real + noise * torch.randn(real.size(), device=self.device)
            real = torch.clamp(real, -1.0, 1.0)

            # Forward pass real batch through D
            output = self.discriminator(real).view(-1)
            e_real = output.mean()
            d_x = output.mean().item()

            #
            # Fake image
            #

            fake = self.generator.generate(real.size()[0])
            fake = fake + noise * torch.randn(real.size(), device=self.device)
            fake = torch.clamp(fake, -1.0, 1.0)

            output = self.discriminator(fake.detach()).view(-1)
            e_fake = output.mean()
            d_z_1 = e_fake.item()

            # Add the gradients from the all-real and all-fake batches
            e_dis = -e_real + e_fake
            e_dis.backward()
            self.optimizer_dis.step()

            #
            # Train generator
            #

            if (i % self.n_critic) == 0:
                self.generator.zero_grad()

                for p in self.discriminator.parameters():
                    p.requires_grad = False

                output = self.discriminator(fake).view(-1)
                e_fake_g = -output.mean()
                e_fake_g.backward()

                d_z_2 = -e_fake_g.item()
                self.optimizer_gen.step()

                self.generator_losses.append(e_fake_g.item())
                self.discriminator_losses.append(e_dis.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            self.fixed_noise.to(self.device)
            if iters % 500 == 0:
                with torch.no_grad():
                    fake = (
                        self.generator(self.fixed_noise.view(-1, 100, 1, 1))
                        .detach()
                        .cpu()
                    )
                    self.image_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                    )
                    self.input_list.append((real, fake))

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        i,
                        len(dataloader),
                        e_dis.item(),
                        e_fake_g.item(),
                        d_x,
                        d_z_1,
                        d_z_2,
                    )
                )

            # Save Losses for plotting later

            iters += 1


class InfoGan:
    def __init__(
        self,
        n_inc=50,
        n_filters_discriminator=32,
        n_filters_generator=32,
        n_cat_dim=10,
        n_con_dim=2,
        lr_gen=0.0002,
        lr_dis=0.0002,
        device=None,
        optimizer="adam",
    ):

        self.n_inc = n_inc
        self.latent_dim = n_inc + n_cat_dim + n_con_dim
        self.n_filters_discriminator = n_filters_discriminator
        self.n_filters_generator = n_filters_generator
        self.device = device
        self.gan_type = "standard"
        self.optimizer = optimizer
        self.n_cat_dim = n_cat_dim
        self.n_con_dim = n_con_dim

        #
        # The device
        #

        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == "gpu" or device == torch.device("cuda:0"):
            self.device = torch.device("cuda:0")
        elif device == "cpu" or device == torch.device("cpu"):
            self.device = torch.device("cpu")
        else:
            raise Exception("Unknown device")

        self.generator = InfoGenerator(
            n_inc,
            n_cat_dim=n_cat_dim,
            n_con_dim=n_con_dim,
            n_filters=n_filters_generator,
        )
        self.discriminator = InfoDiscriminator(
            n_filters=n_filters_discriminator, n_cat_dim=n_cat_dim, n_con_dim=n_con_dim
        )
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.generator.device = self.device

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        for l in self.generator.children():
            weights_init(l)

            for l in self.discriminator.children():
                weights_init(l)

        self.generator_losses = []
        self.discriminator_losses = []
        self.categorical_losses = []
        self.image_list = []
        self.input_list = []

        #
        # Create optimizers
        #

        beta1 = 0.5
        if optimizer == "adam":
            params = [
                {"params": self.discriminator.body.parameters()},
                {"params": self.discriminator.head_dis.parameters()},
            ]
            self.optimizer_dis = torch.optim.Adam(
                params, lr=lr_dis, betas=(beta1, 0.999)
            )
            params = [{"params": self.generator.parameters()}]
            self.optimizer_gen = torch.optim.Adam(
                params, lr=lr_gen, betas=(beta1, 0.999)
            )
            params = itertools.chain(
                self.discriminator.head_q.parameters(),
                self.discriminator.head_c.parameters(),
                self.discriminator.head_q_mean.parameters(),
                self.discriminator.head_q_var.parameters(),
                self.discriminator.body.parameters(),
                self.generator.parameters(),
            )
            self.optimizer_cat = torch.optim.Adam(
                params,
                lr=lr_gen,
            )
        elif optimizer == "sgd":
            self.optimizer_dis = torch.optim.SGD(
                self.discriminator.parameters(), lr=lr_dis
            )
            self.optimizer_gen = torch.optim.SGD(self.generator.parameters(), lr=lr_gen)
        else:
            raise Exception("Unknown optimizer type.")

        # Random input to track progress
        bs = self.n_cat_dim * 8
        self.fixed_noise = torch.randn(bs, self.latent_dim, 1, 1, device=self.device)
        self.fixed_noise[:, self.n_inc :] = 0.0
        inds = np.arange(self.n_cat_dim * 8) // 8 + self.n_inc
        self.fixed_noise[range(bs), inds] = 1.0
        qs = (
            torch.linspace(0, 1, 8)
            .view(-1, 1, 1, 1)
            .repeat(self.n_cat_dim, self.n_con_dim, 1, 1)
        )
        self.fixed_noise[:, self.n_inc + self.n_cat_dim :] = qs

        self.criterion = nn.BCELoss()
        self.criterion_cat = nn.NLLLoss()
        self.criterion_q = NormalNLLLoss()

    def train(self, dataloader, noise=0.1):

        self.discriminator.to(self.device)
        self.generator.to(self.device)

        real_label = 0.9
        fake_label = 0
        iters = 0

        counts = torch.zeros(self.n_cat_dim, requires_grad=False)
        totals = torch.zeros(self.n_cat_dim, requires_grad=False)

        for i, data in enumerate(dataloader, 0):

            # Handle supervised dataset
            if type(data) == list:
                data = data[0]
            # Skip incomplete batch
            if data.size()[0] < dataloader.batch_size:
                continue

            self.optimizer_dis.zero_grad()

            real = data.to(self.device)
            real = real + noise * torch.randn(real.size(), device=self.device)
            real = torch.clamp(real, -1.0, 1.0)

            bs = real.size(0)
            label = torch.full((bs,), real_label, device=self.device)

            # Forward pass real batch through D
            output, class_logits, _, _ = self.discriminator(real)
            n = self.n_cat_dim

            classes = torch.argmax(class_logits, dim=1).float()
            counts = counts + torch.histc(classes, n, -0.5, self.n_cat_dim - 0.5)
            totals = totals + dataloader.batch_size

            err_dis_real = self.criterion(output.view(-1), label)
            err_dis_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            fake, c_target, q_target = self.generator.generate(bs)
            fake = fake + noise * torch.randn(real.size(), device=self.device)
            fake = torch.clamp(fake, -1.0, 1.0)

            output, c, q_mean, q_log_var = self.discriminator(fake.detach())
            label.fill_(fake_label)
            err_dis_fake = self.criterion(output.view(-1), label)
            err_dis_fake.backward()

            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            err_dis = err_dis_real + err_dis_fake
            self.optimizer_dis.step()

            #
            # Train generator
            #

            self.optimizer_gen.zero_grad()
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output, c, q_mean, q_log_var = self.discriminator(fake)
            err_dis_gen = self.criterion(output.view(-1), label)

            err_gen = err_dis_gen
            err_gen.backward(retain_graph=True)

            D_G_z2 = output.mean().item()
            self.optimizer_gen.step()

            #
            # Train Q
            #

            self.optimizer_cat.zero_grad()
            c_target = c_target.long()
            c = c.view(-1, self.n_cat_dim)
            c_target = c_target.view(-1)
            err_cat = self.criterion_cat(c, c_target)

            q_target = q_target.view(-1, self.n_con_dim)
            q_mean = q_mean.view(-1, self.n_con_dim)
            q_log_var = q_log_var.view(-1, self.n_con_dim)

            print(q_target[0, :])
            print(q_mean[0, :])
            print(q_log_var[0, :])

            err_q = self.criterion_q(q_target, q_mean, q_log_var)

            err = err_cat + 0.1 * err_q

            err.backward()

            self.optimizer_cat.step()

            # Check how the generator is doing by saving G's output on fixed_noise
            self.fixed_noise.to(self.device)
            if iters % 500 == 0:
                with torch.no_grad():
                    fake = self.generator(self.fixed_noise).detach().cpu()
                    self.image_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                    )
                    self.input_list.append((real, fake))

            # if (iters % 2000 == 0):
            #    if (iters > 0) and totals.sum().item() > 0:
            #        self.generator.update_frequencies(counts/ totals)
            #        counts.fill_ = 0.0
            #        totals.fill_ = 0.0

            #    ws = 1.0 / self.generator.frequencies[-1]
            #    self.criterion_cat = nn.NLLLoss(weight = ws)

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d]\tL_D: %.4f L_G: %.3f | D(x): %.4f, D(G(z)): %.3f / %.3f | L_C: %.3f, L_Q: %.3f"
                    % (
                        i,
                        len(dataloader),
                        err_dis.item(),
                        err_dis_gen.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                        err_cat.item(),
                        err_q.item(),
                    )
                )

            # Save Losses for plotting later
            self.generator_losses.append(err_dis_gen.item())
            self.discriminator_losses.append(err_dis.item())
            self.categorical_losses.append(err_cat.item())

            iters += 1

    def get_features(self, x):
        for l in next(self.discriminator.children()).children():
            if x.size()[-1] > 1:
                x = l(x)
            else:
                break
        return x

    def save(self, path):
        torch.save(
            {
                "n_inc": self.n_inc,
                "n_cat_dim": self.n_cat_dim,
                "n_con_dim": self.n_con_dim,
                "n_filters_discriminator": self.n_filters_discriminator,
                "n_filters_generator": self.n_filters_generator,
                "device": self.device,
                "optimizer": self.optimizer,
                "discriminator_state": self.discriminator.state_dict(),
                "generator_state": self.generator.state_dict(),
                "discriminator_opt_state": self.optimizer_dis.state_dict(),
                "generator_opt_state": self.optimizer_gen.state_dict(),
                "discriminator_losses": self.discriminator_losses,
                "generator_losses": self.generator_losses,
                "generator_frequencies": self.generator.frequencies,
                "gan_type": self.gan_type,
                "image_list": self.image_list,
                "input_list": self.input_list,
            },
            path,
        )

    @staticmethod
    def load(path):

        state = torch.load(path)
        print(state.keys())

        keys = [
            "n_inc",
            "n_cat_dim",
            "n_con_dim",
            "n_filters_discriminator",
            "n_filters_generator",
            "device",
            "optimizer",
        ]
        kwargs = dict([(k, state[k]) for k in keys])

        gan = InfoGan(**kwargs)

        try:
            gan.discriminator.load_state_dict(state["discriminator_state"])
            gan.optimizer_dis.load_state_dict(state["discriminator_opt_state"])
            gan.generator.load_state_dict(state["generator_state"])
            gan.optimizer_gen.load_state_dict(state["generator_opt_state"])
        except Exception as e:
            print(e)
            pass

        gan.discriminator_losses = state["discriminator_losses"]
        gan.generator_losses = state["generator_losses"]
        gan.gan_type = state["gan_type"]
        gan.image_list = state["image_list"]
        gan.input_list = state["input_list"]
        gan.generator.frequencies = state["generator_frequencies"]

        return gan


class WGan(Gan):
    def __init__(
        self,
        latent_dim=100,
        n_filters_discriminator=32,
        n_filters_generator=32,
        features=32,
        device=None,
        optimizer="rmsprop",
        c=0.02,
        n_critic=5,
    ):

        super(WGan, self).__init__(
            latent_dim,
            n_filters_discriminator=n_filters_discriminator,
            n_filters_generator=n_filters_generator,
            features=features,
            device=device,
        )

        modules_dis = list(next(iter(self.discriminator.children())).children())
        self.discriminator.main = nn.Sequential(*modules_dis[:-1])
        self.discriminator.to(self.device)

        self.gan_type = "wasserstein"
        alpha = 0.00005

        self.optimizer = optimizer

        if not optimizer == "rmsprop":
            # raise Exception("Only rmsprop supported for WGAN.")
            pass

        self.optimizer_gen = torch.optim.RMSprop(self.generator.parameters(), lr=alpha)
        self.optimizer_dis = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=alpha
        )
        self.c = c
        self.n_critic = n_critic

    def train(self, dataloader, lr_gen=0.0002, lr_dis=0.0002, noise=0.1):

        self.optimizer_gen.lr = lr_gen
        self.optimizer_dis.lr = lr_dis

        self.discriminator.to(self.device)
        self.generator.to(self.device)

        iters = 0
        one = torch.FloatTensor([1])
        mone = one * -1
        one, mone = one.to(self.device), mone.to(self.device)
        bs = dataloader.batch_size

        for i, data in enumerate(dataloader, 0):

            # Turn on gradients
            for p in self.discriminator.parameters():
                p.requires_grad = True
            # Clip weights
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.c, self.c)
            # Remove old gradients
            self.discriminator.zero_grad()

            #
            # Real image
            #

            # Add noise to data
            real = data.to(self.device)
            real = real + noise * torch.randn(real.size(), device=self.device)
            real = torch.clamp(real, -1.0, 1.0)

            # Forward pass real batch through D
            output = self.discriminator(real).view(-1)
            e_real = output.mean()
            d_x = output.mean().item()

            #
            # Fake image
            #

            fake = self.generator.generate(real.size()[0])
            fake = fake + noise * torch.randn(real.size(), device=self.device)
            fake = torch.clamp(fake, -1.0, 1.0)

            output = self.discriminator(fake.detach()).view(-1)
            e_fake = output.mean()
            d_z_1 = e_fake.item()

            # Add the gradients from the all-real and all-fake batches
            e_dis = -e_real + e_fake
            e_dis.backward()
            self.optimizer_dis.step()

            #
            # Train generator
            #

            if (i % self.n_critic) == 0:
                self.generator.zero_grad()

                for p in self.discriminator.parameters():
                    p.requires_grad = False

                output = self.discriminator(fake).view(-1)
                e_fake_g = -output.mean()
                e_fake_g.backward()

                d_z_2 = -e_fake_g.item()
                self.optimizer_gen.step()

                self.generator_losses.append(e_fake_g.item())
                self.discriminator_losses.append(e_dis.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            self.fixed_noise.to(self.device)
            if iters % 500 == 0:
                with torch.no_grad():
                    fake = (
                        self.generator(self.fixed_noise.view(-1, 100, 1, 1))
                        .detach()
                        .cpu()
                    )
                    self.image_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                    )
                    self.input_list.append((real, fake))

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        i,
                        len(dataloader),
                        e_dis.item(),
                        e_fake_g.item(),
                        d_x,
                        d_z_1,
                        d_z_2,
                    )
                )

            # Save Losses for plotting later

            iters += 1
