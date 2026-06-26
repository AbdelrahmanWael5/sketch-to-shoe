👟 Sketch-to-Shoe: Investigating Conditioning Strategies for Diffusion Transformers

A research-oriented implementation that explores conditioning mechanisms for Diffusion Transformers (DiT) to perform conditional image generation from edge sketches.

Overview

This project investigates how the original Diffusion Transformer (DiT) architecture can be adapted for Sketch-to-Image generation.

Instead of reproducing the original DiT, this work focuses on designing and comparing different conditioning strategies that enable the model to generate realistic shoe images from edge sketches while operating in the latent space of Stable Diffusion v1.5.

The final model was trained for approximately 25,000 optimization steps on 10,000 training image pairs. Although additional training is expected to further improve image quality, the current model already demonstrates promising structural understanding and sketch-guided generation.

Methodology

Two conditioning strategies were investigated:

• CLIP Conditioning

The input sketch is encoded using a pretrained CLIP image encoder, and the resulting embedding is injected into the Diffusion Transformer through the AdaLN conditioning mechanism.

• Edge Latent Conditioning (Final Model)

The input sketch is encoded using the Stable Diffusion VAE encoder to obtain an edge latent representation.

The edge latent is concatenated with the noisy target latent before being processed by the Diffusion Transformer, providing explicit spatial guidance throughout the denoising process.

This approach consistently produced more structurally faithful generations than CLIP conditioning, demonstrating that spatial latent conditioning preserves shoe geometry more effectively than global semantic conditioning.

Dataset

Training was performed using the Edges2Shoes Dataset.

Dataset: https://www.kaggle.com/datasets/balraj98/edges2shoes-dataset
Training Samples: 10,000
Testing Samples: 1,000
Image Resolution: 256 × 256
Training Configuration
Parameter	Value
Backbone	Diffusion Transformer (DiT)
VAE	Stable Diffusion v1.5
Scheduler	DDPM
Optimizer	AdamW
Learning Rate	5e-5
Loss	MSE
Training Steps	~25,000

Running the Projectj
Launch the Streamlit application:

streamlit run app.py
Repository Structure
Sketch-to-Shoe
│
├── app.py
├── inference.py
├── dataset.py
├── train_DiT.ipynb
│
├── DiT/
│   ├── diffusion/
│   ├── models.py
│   ├── models_clip_conditioning.py
│   └── models_edge_latents_conditioning.py
│
├── requirements.txt
└── README.md
Future Work
Continue training beyond 25k optimization steps.
Train on the complete Edges2Shoes dataset.
Investigate additional conditioning mechanisms.
Evaluate CFG and alternative diffusion schedulers.
Report quantitative metrics such as FID and CLIP Score.
References
Scalable Diffusion Models with Transformers (DiT) — Peebles & Xie, ICCV 2023
Stable Diffusion v1.5
OpenAI CLIP
Edges2Shoes Dataset
Author

Abdelrahman Wael

If you find this project interesting, consider giving the repository a ⭐
