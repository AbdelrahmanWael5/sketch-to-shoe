import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler

from DiT.models_edge_latents_conditioning import DiT_XL_2


class ShoeGenerator:
    def __init__(
        self,
        checkpoint_path,
        device=None,
        num_inference_steps=100,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.num_inference_steps = num_inference_steps

        # -----------------------
        # Load VAE
        # -----------------------
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
        ).to(self.device)

        self.vae.eval()

        for p in self.vae.parameters():
            p.requires_grad = False

        # -----------------------
        # Load DiT
        # -----------------------
        self.dit = DiT_XL_2(input_size=32).to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.dit.load_state_dict(checkpoint["DiT_state_dict"])
        self.dit.eval()

        # -----------------------
        # Scheduler
        # -----------------------
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000
        )

        # -----------------------
        # Transform
        # -----------------------
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ])

    @torch.no_grad()
    def encode_img(self, img):
        latents = self.vae.encode(img)
        latents = latents.latent_dist.sample()
        latents *= self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        return self.vae.decode(
            latents / self.vae.config.scaling_factor
        ).sample

    def preprocess(self, image):
        """
        image:
            - PIL.Image
            - path to image
        """

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        image = self.transform(image)
        image = image.unsqueeze(0)

        return image.to(self.device)

    @torch.no_grad()
    def generate(self, image):
        """
        Parameters
        ----------
        image : PIL.Image or str

        Returns
        -------
        PIL.Image
        """

        edge = self.preprocess(image)

        batch_size = edge.shape[0]

        edge_latents = self.encode_img(edge)

        latents = torch.randn(
            batch_size,
            4,
            32,
            32,
            device=self.device,
        )

        latents *= self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(
            self.num_inference_steps,
            device=self.device,
        )

        for t in self.scheduler.timesteps:

            timestep = torch.full(
                (batch_size,),
                t,
                device=self.device,
                dtype=torch.long,
            )

            model_input = torch.cat(
                [latents, edge_latents],
                dim=1,
            )

            pred = self.dit(
                model_input,
                timestep,
            )

            pred_noise = pred[:, :4]

            latents = self.scheduler.step(
                pred_noise,
                t,
                latents,
            ).prev_sample

        image = self.decode_latents(latents)

        image = (image + 1) / 2
        image = image.clamp(0, 1)

        image = transforms.ToPILImage()(image.squeeze(0).cpu())

        return image