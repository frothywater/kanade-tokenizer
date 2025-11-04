from dataclasses import dataclass
from typing import Literal

import jsonargparse
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .data.datamodule import AudioBatch
from .model import KanadeModel, KanadeModelConfig
from .module.audio_feature import MelSpectrogramFeature
from .module.discriminator import SpectrogramDiscriminator
from .module.fsq import FiniteScalarQuantizer
from .module.global_encoder import GlobalEncoder
from .module.postnet import PostNet
from .module.ssl_extractor import SSLFeatureExtractor
from .module.transformer import Transformer
from .util import freeze_modules, get_logger

logger = get_logger()


@dataclass
class KanadePipelineConfig:
    # Training control
    train_feature: bool = True  # Whether to train the feature reconstruction branch
    train_mel: bool = True  # Whether to train the mel spectrogram generation branch

    # Audio settings
    audio_length: int = 138240  # Length of audio input in samples

    # Optimization settings
    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    gradient_clip_val: float | None = 1.0

    # LR scheduling parameters
    warmup_percent: float = 0.1
    lr_div_factor: float = 10.0
    lr_final_div_factor: float = 1.0
    anneal_mode: str = "cos"

    # Loss weights
    feature_l1_weight: float = 30.0
    feature_l2_weight: float = 0.0
    mel_l1_weight: float = 30.0
    mel_l2_weight: float = 0.0
    adv_weight: float = 1.0
    feature_matching_weight: float = 10.0

    # GAN settings
    use_discriminator: bool = False
    adv_loss_type: Literal["hinge", "least_square"] = "hinge"  # Type of adversarial loss
    discriminator_lr: float | None = None  # Learning rate for discriminator
    discriminator_start_step: int = 0  # Step to start training discriminator
    discriminator_update_prob: float = 1.0  # Probability of updating discriminator at each step

    # Checkpoint loading
    ckpt_path: str | None = None  # Path to checkpoint to load from
    skip_loading_modules: tuple[str, ...] = ()  # Modules to skip when loading checkpoint

    # Other settings
    log_mel_samples: int = 10
    use_torch_compile: bool = True


class KanadePipeline(L.LightningModule):
    """LightningModule wrapper for KanadeModel, handling training (including GAN)."""

    def __init__(
        self,
        model_config: KanadeModelConfig,
        pipeline_config: KanadePipelineConfig,
        ssl_feature_extractor: SSLFeatureExtractor,
        local_encoder: Transformer,
        local_quantizer: FiniteScalarQuantizer,
        feature_decoder: Transformer | None,
        global_encoder: GlobalEncoder,
        mel_prenet: Transformer,
        mel_decoder: Transformer,
        mel_postnet: PostNet,
        discriminator: SpectrogramDiscriminator | None = None,
    ):
        super().__init__()
        self.config = pipeline_config
        self.save_hyperparameters("model_config", "pipeline_config")
        self.strict_loading = False
        self.automatic_optimization = False
        self.torch_compiled = False

        # Validate components required for training
        assert not pipeline_config.train_feature or feature_decoder is not None, (
            "Feature decoder must be provided if training feature reconstruction"
        )
        logger.info(
            f"Training configuration: train_feature={pipeline_config.train_feature}, train_mel={pipeline_config.train_mel}"
        )

        # 1. Kanade model
        self.model = KanadeModel(
            config=model_config,
            ssl_feature_extractor=ssl_feature_extractor,
            local_encoder=local_encoder,
            local_quantizer=local_quantizer,
            feature_decoder=feature_decoder,
            global_encoder=global_encoder,
            mel_decoder=mel_decoder,
            mel_prenet=mel_prenet,
            mel_postnet=mel_postnet,
        )
        self._freeze_unused_modules(pipeline_config.train_feature, pipeline_config.train_mel)

        # Calculate padding for expected SSL output length
        self.padding = self.model._calculate_waveform_padding(pipeline_config.audio_length)
        logger.info(f"Input waveform padding for SSL feature extractor: {self.padding} samples")

        # Calculate target mel spectrogram length
        self.target_mel_length = self.model._calculate_target_mel_length(pipeline_config.audio_length)
        logger.info(f"Target mel spectrogram length: {self.target_mel_length} frames")

        # 2. Discriminator
        self._init_discriminator(pipeline_config, discriminator)

        # 3. Mel spectrogram feature extractor for loss computation
        if pipeline_config.train_mel:
            self.mel_spec = MelSpectrogramFeature(
                sample_rate=model_config.sample_rate,
                n_fft=model_config.n_fft,
                hop_length=model_config.hop_length,
                n_mels=model_config.n_mels,
                padding=model_config.padding,
            )

        # Mel sample storage for logging
        self.vocoder = None
        self.validation_examples = []
        self.log_mel_samples = pipeline_config.log_mel_samples

    def _freeze_unused_modules(self, train_feature: bool, train_mel: bool):
        model = self.model
        if not train_feature:
            # Freeze local branch components if not training feature reconstruction
            freeze_modules([model.local_encoder, model.local_quantizer, model.feature_decoder])
            if model.conv_downsample is not None:
                freeze_modules([model.conv_downsample, model.conv_upsample])
            logger.info("Feature reconstruction branch frozen: local_encoder, local_quantizer, feature_decoder")

        if not train_mel:
            # Freeze global branch and mel generation components if not training mel generation
            freeze_modules(
                [model.global_encoder, model.mel_prenet, model.mel_conv_upsample, model.mel_decoder, model.mel_postnet]
            )
            logger.info(
                "Mel generation branch frozen: global_encoder, mel_prenet, mel_conv_upsample, mel_decoder, mel_postnet"
            )

    def _init_discriminator(self, config: KanadePipelineConfig, discriminator: SpectrogramDiscriminator | None):
        # Setup discriminator if provided
        self.discriminator = discriminator
        self.use_discriminator = config.use_discriminator and discriminator is not None and config.train_mel

        if config.use_discriminator and discriminator is None:
            logger.error(
                "Discriminator is enabled in config but no discriminator model provided. Disabling GAN training."
            )
        if config.use_discriminator and discriminator is not None and not config.train_mel:
            logger.warning(
                "Discriminator is enabled but train_mel=False. Discriminator will not be effective without mel training."
            )

        self.discriminator_start_step = config.discriminator_start_step
        self.discriminator_update_prob = config.discriminator_update_prob
        if self.use_discriminator:
            logger.info("Discriminator initialized for GAN training")
            logger.info(f"Discriminator start step: {self.discriminator_start_step}")
            logger.info(f"Discriminator update probability: {self.discriminator_update_prob}")

    def setup(self, stage: str):
        # Torch compile model if enabled
        if torch.__version__ >= "2.0" and self.config.use_torch_compile:
            self.model = torch.compile(self.model)
            if self.discriminator is not None:
                self.discriminator = torch.compile(self.discriminator)
            self.torch_compiled = True

        # Load checkpoint if provided
        if self.config.ckpt_path:
            ckpt_path = self.config.ckpt_path

            # Download weights from HuggingFace Hub if needed
            if ckpt_path.startswith("hf:"):
                from huggingface_hub import hf_hub_download

                repo_id = ckpt_path[len("hf:") :]
                # Separate out revision if specified
                revision = None
                if "@" in repo_id:
                    repo_id, revision = repo_id.split("@", 1)

                ckpt_path = hf_hub_download(repo_id, filename="model.safetensors", revision=revision)

            self._load_weights(ckpt_path)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            ssl_real: Extracted SSL features for local branch (B, T, C)
            ssl_recon: Reconstructed SSL features (B, T, C) - only if train_feature=True
            mel_recon: Generated mel spectrogram (B, n_mels, T) - only if train_mel=True
            loss_dict: Dictionary with auxiliary information (codes, losses, etc.)
        """
        loss_dict = {}

        # 1. Extract SSL features
        local_ssl_features, global_ssl_features = self.model.forward_ssl_features(waveform, padding=self.padding)

        # 2. Content branch processing
        content_embeddings, _, ssl_recon, perplexity = self.model.forward_content(local_ssl_features)
        loss_dict["local/perplexity"] = perplexity

        # 3. Global branch processing and mel reconstruction
        mel_recon = None
        if self.config.train_mel:
            global_embeddings = self.model.forward_global(global_ssl_features)
            mel_recon = self.model.forward_mel(content_embeddings, global_embeddings, mel_length=self.target_mel_length)

        return local_ssl_features, ssl_recon, mel_recon, loss_dict

    def _get_reconstruction_loss(
        self, audio_real: torch.Tensor, ssl_real: torch.Tensor, ssl_recon: torch.Tensor, mel_recon: torch.Tensor
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """Compute L1 + L2 loss for SSL feature and mel spectrogram reconstruction.
        Returns:
            total_loss: Combined reconstruction loss
            loss_dict: Dictionary with individual loss components
            mel_real: Real mel spectrogram for reference
        """
        if audio_real.dim() == 3:
            audio_real = audio_real.squeeze(1)

        loss_dict = {}
        feature_loss, mel_loss = 0, 0

        # Compute SSL feature reconstruction losses if training features
        if self.config.train_feature and self.model.feature_decoder is not None:
            assert ssl_real is not None and ssl_recon is not None, (
                "SSL features must be provided for training feature reconstruction"
            )
            ssl_l1 = F.l1_loss(ssl_recon, ssl_real)
            ssl_l2 = F.mse_loss(ssl_recon, ssl_real)

            feature_loss = self.config.feature_l1_weight * ssl_l1 + self.config.feature_l2_weight * ssl_l2
            loss_dict.update({"ssl_l1": ssl_l1, "ssl_l2": ssl_l2, "feature_loss": feature_loss})

        # Compute mel spectrogram reconstruction losses if training mel
        mel_real = None
        if self.config.train_mel:
            assert mel_recon is not None, "Mel reconstruction must be provided for training mel generation"
            # Extract reference mel spectrogram from audio
            mel_real = self.mel_spec(audio_real)

            mel_l1 = F.l1_loss(mel_recon, mel_real)
            mel_l2 = F.mse_loss(mel_recon, mel_real)
            mel_loss = self.config.mel_l1_weight * mel_l1 + self.config.mel_l2_weight * mel_l2
            loss_dict.update({"mel_l1": mel_l1, "mel_l2": mel_l2, "mel_loss": mel_loss})

        total_loss = feature_loss + mel_loss
        return total_loss, loss_dict, mel_real

    def _get_discriminator_loss(
        self, real_outputs: torch.Tensor, fake_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the adversarial loss for discriminator.
        Returns:
            disc_loss: Total discriminator loss
            real_loss: Loss component from real samples
            fake_loss: Loss component from fake samples
        """
        if self.config.adv_loss_type == "hinge":
            real_loss = torch.mean(torch.clamp(1 - real_outputs, min=0))
            fake_loss = torch.mean(torch.clamp(1 + fake_outputs, min=0))
        elif self.config.adv_loss_type == "least_square":
            real_loss = torch.mean((real_outputs - 1) ** 2)
            fake_loss = torch.mean(fake_outputs**2)
        else:
            raise ValueError(f"Unknown adversarial loss type: {self.config.adv_loss_type}")

        disc_loss = real_loss + fake_loss
        return disc_loss, real_loss, fake_loss

    def _get_generator_loss(self, fake_outputs: torch.Tensor) -> torch.Tensor:
        """Compute the adversarial loss for generator."""
        if self.config.adv_loss_type == "hinge":
            return torch.mean(torch.clamp(1 - fake_outputs, min=0))
        elif self.config.adv_loss_type == "least_square":
            return torch.mean((fake_outputs - 1) ** 2)
        else:
            raise ValueError(f"Unknown adversarial loss type: {self.config.adv_loss_type}")

    def _get_feature_matching_loss(
        self, real_intermediates: list[torch.Tensor], fake_intermediates: list[torch.Tensor]
    ) -> torch.Tensor:
        losses = []
        for real_feat, fake_feat in zip(real_intermediates, fake_intermediates):
            losses.append(torch.mean(torch.abs(real_feat.detach() - fake_feat)))
        fm_loss = torch.mean(torch.stack(losses))
        return fm_loss

    def _discriminator_step(
        self, batch: AudioBatch, optimizer_disc: torch.optim.Optimizer
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]:
        """
        Returns:
            ssl_real: Real SSL features
            ssl_recon: Reconstructed SSL features from generator
            mel_recon: Generated mel spectrogram
            loss_dict: Dictionary with auxiliary information
            real_intermediates: Intermediate feature maps from discriminator for real mel
        """
        assert self.use_discriminator, "Discriminator step called but discriminator is not enabled"

        ssl_real, ssl_recon, mel_recon, loss_dict = self(batch.waveform)
        assert mel_recon is not None, "Mel reconstruction must be available for discriminator step"

        # Get true mel spectrogram (always use original waveform)
        mel_real = self.mel_spec(batch.waveform)

        # Get discriminator outputs and intermediates for real mel
        real_outputs, real_intermediates = self.discriminator(mel_real)
        fake_outputs, _ = self.discriminator(mel_recon.detach())

        # Compute discriminator loss
        disc_loss, real_loss, fake_loss = self._get_discriminator_loss(real_outputs, fake_outputs)

        # Log discriminator losses
        batch_size = batch.waveform.size(0)
        self.log("train/disc/real", real_loss, batch_size=batch_size)
        self.log("train/disc/fake", fake_loss, batch_size=batch_size)
        self.log("train/disc/loss", disc_loss, batch_size=batch_size, prog_bar=True)
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, batch_size=batch_size)

        # Optimize discriminator
        optimizer_disc.zero_grad()
        self.manual_backward(disc_loss)

        # Log gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(), max_norm=self.config.gradient_clip_val or torch.inf
        )
        self.log("train/disc/grad_norm", grad_norm, batch_size=batch_size)

        optimizer_disc.step()

        return ssl_real, ssl_recon, mel_recon, loss_dict, real_intermediates

    def _generator_step(
        self,
        batch: AudioBatch,
        optimizer_gen: torch.optim.Optimizer,
        ssl_real: torch.Tensor | None = None,
        ssl_recon: torch.Tensor | None = None,
        mel_recon: torch.Tensor | None = None,
        loss_dict: dict | None = None,
        real_intermediates: list[torch.Tensor] | None = None,
        training_disc: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            batch: Audio batch with waveform and augmented_waveform
            optimizer_gen: Generator optimizer
            ssl_real: Real SSL features (optional)
            ssl_recon: Reconstructed SSL features (optional)
            mel_recon: Generated mel spectrogram (optional)
            loss_dict: Dictionary with auxiliary information (optional)
            real_intermediates: Intermediate feature maps from discriminator for real mel (optional)
            training_disc: Whether discriminator is being trained in this step

        Returns:
            gen_loss: Total generator loss
        """
        # Forward pass through the model if not already done in discriminator step
        if loss_dict is None:
            ssl_real, ssl_recon, mel_recon, loss_dict = self(batch.waveform)

        # Compute reconstruction loss (always use original waveform for mel target)
        recon_loss, recon_dict, mel_real = self._get_reconstruction_loss(batch.waveform, ssl_real, ssl_recon, mel_recon)
        gen_loss = recon_loss

        # Compute adversarial and feature matching losses if using discriminator
        batch_size = batch.waveform.size(0)
        if training_disc:
            assert mel_real is not None and mel_recon is not None, "Mel spectrograms must be provided for GAN training"

            if real_intermediates is None:
                _, real_intermediates = self.discriminator(mel_real)

            fake_outputs, fake_intermediates = self.discriminator(mel_recon)

            # Compute adversarial loss
            adv_loss = self._get_generator_loss(fake_outputs)
            gen_loss += self.config.adv_weight * adv_loss
            self.log("train/gen/adv_loss", adv_loss, batch_size=batch_size)

            # Compute feature matching loss
            feature_matching_loss = self._get_feature_matching_loss(real_intermediates, fake_intermediates)
            gen_loss += self.config.feature_matching_weight * feature_matching_loss
            self.log("train/gen/feature_matching_loss", feature_matching_loss, batch_size=batch_size)

        # Log reconstruction losses
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, batch_size=batch_size)
        for name, value in recon_dict.items():
            self.log(f"train/gen/{name}", value, batch_size=batch_size)

        self.log("train/loss", gen_loss, batch_size=batch_size, prog_bar=True)

        # Optimize generator
        optimizer_gen.zero_grad()
        self.manual_backward(gen_loss)

        # Log gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.gradient_clip_val or torch.inf
        )
        self.log("train/gen/grad_norm", grad_norm, batch_size=batch_size)

        optimizer_gen.step()

        return gen_loss

    def training_step(self, batch: AudioBatch, batch_idx: int):
        if self.use_discriminator:
            optimizer_disc, optimizer_gen = self.optimizers()
            scheduler_disc, scheduler_gen = self.lr_schedulers()
        else:
            optimizer_gen = self.optimizers()
            scheduler_gen = self.lr_schedulers()

        # Determine if discriminator should be trained in this step
        training_disc = (
            self.use_discriminator
            and self.global_step >= self.discriminator_start_step
            and torch.rand(1).item() < self.discriminator_update_prob
        )
        if self.global_step == self.discriminator_start_step and self.use_discriminator:
            logger.info(f"Discriminator training starts at step {self.global_step}")

        ssl_real, ssl_recon, mel_recon, loss_dict, real_intermediates = None, None, None, None, None

        # Train discriminator if conditions are met
        if training_disc:
            ssl_real, ssl_recon, mel_recon, loss_dict, real_intermediates = self._discriminator_step(
                batch, optimizer_disc
            )
            scheduler_disc.step()
        elif self.use_discriminator:
            # Step the discriminator scheduler even when not training discriminator
            scheduler_disc.step()

        # Train generator
        self._generator_step(
            batch, optimizer_gen, ssl_real, ssl_recon, mel_recon, loss_dict, real_intermediates, training_disc
        )
        scheduler_gen.step()

    def validation_step(self, batch: AudioBatch, batch_idx: int):
        audio_real = batch.waveform
        ssl_real, ssl_recon, mel_recon, loss_dict = self(audio_real)

        # Convert to waveform using vocoder for logging
        batch_size = audio_real.size(0)

        # Compute reconstruction loss
        recon_loss, recon_dict, mel_real = self._get_reconstruction_loss(audio_real, ssl_real, ssl_recon, mel_recon)
        gen_loss = recon_loss

        # Log reconstruction losses
        for name, value in loss_dict.items():
            self.log(f"val/{name}", value, batch_size=batch_size)
        for name, value in recon_dict.items():
            self.log(f"val/gen/{name}", value, batch_size=batch_size)
        self.log("val/loss", gen_loss, batch_size=batch_size)

        # Save first few samples for visualization at end of epoch if training mel generation
        if self.config.train_mel and len(self.validation_examples) < self.log_mel_samples:
            assert mel_real is not None and mel_recon is not None, (
                "Mel spectrograms must be provided for validation logging"
            )
            audio_real = audio_real[0].cpu()
            audio_gen = None
            if self.vocoder is not None:
                audio_gen = self.vocode(mel_recon[0:1])[0].cpu()

            self.validation_examples.append((mel_real[0].cpu(), mel_recon[0].detach().cpu(), audio_real, audio_gen))

    def predict_step(self, batch: AudioBatch, batch_idx: int):
        audio_real = batch.waveform
        _, _, mel_gen, _ = self(audio_real)

        audio_gen = self.vocode(mel_gen)

        if audio_gen.dim() == 2:
            audio_gen = audio_gen.unsqueeze(1)
        return {"audio_ids": batch.audio_ids, "audio_real": audio_real, "audio_gen": audio_gen}

    def configure_optimizers(self):
        # Generator optimizer
        optimizer_gen = AdamW(
            self.model.parameters(), lr=self.config.lr, betas=self.config.betas, weight_decay=self.config.weight_decay
        )

        # Generator LR scheduler
        scheduler_gen = OneCycleLR(
            optimizer_gen,
            max_lr=self.config.lr,
            div_factor=self.config.lr_div_factor,
            final_div_factor=self.config.lr_final_div_factor,
            pct_start=self.config.warmup_percent,
            anneal_strategy=self.config.anneal_mode,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        if not self.use_discriminator:
            return ([optimizer_gen], [{"scheduler": scheduler_gen, "interval": "step"}])

        # If using discriminator, also configure discriminator optimizer and scheduler
        optimizer_disc = AdamW(
            self.discriminator.parameters(),
            lr=self.config.discriminator_lr or self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

        # Discriminator LR scheduler
        scheduler_disc = OneCycleLR(
            optimizer_disc,
            max_lr=self.config.discriminator_lr or self.config.lr,
            div_factor=self.config.lr_div_factor,
            final_div_factor=self.config.lr_final_div_factor,
            pct_start=self.config.warmup_percent,
            anneal_strategy=self.config.anneal_mode,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        # Load optimizer state
        if self.config.ckpt_path:
            if self.config.ckpt_path.endswith(".ckpt"):
                checkpoint = torch.load(self.config.ckpt_path)
                optimizer_states = checkpoint["optimizer_states"]
                if len(optimizer_states) > 1 and self.use_discriminator:
                    optimizer_disc.load_state_dict(optimizer_states[0])
                    optimizer_gen.load_state_dict(optimizer_states[1])
                    logger.info("Loaded discriminator and generator's optimizer states from checkpoint")
                elif len(optimizer_states) == 1 and not self.use_discriminator:
                    # Load generator optimizer state only
                    optimizer_gen.load_state_dict(optimizer_states[0])
                    logger.info("Loaded generator's optimizer state from checkpoint")
            else:
                logger.info("No optimizer state loaded since checkpoint is not a .ckpt file")

        return (
            [optimizer_disc, optimizer_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    def _setup_vocoder(self):
        try:
            from vocos import Vocos

            model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            return model.eval()
        except ImportError:
            logger.error("Vocos not found. Please install vocos to enable vocoding during validation/prediction.")
            return None

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        self.vocoder = self.vocoder.to(mel.device)
        mel = mel.float()
        waveform = self.vocoder.decode(mel)  # (B, T)

        return waveform.cpu().float()

    def on_validation_start(self):
        self.vocoder = self._setup_vocoder()

    def on_predict_start(self):
        self.vocoder = self._setup_vocoder()

    def on_validation_end(self):
        if len(self.validation_examples) > 0:
            for i, (mel_real, mel_recon, audio_real, audio_gen) in enumerate(self.validation_examples):
                # Log spectrograms
                fig_real = self._get_spectrogram_plot(mel_real)
                fig_gen = self._get_spectrogram_plot(mel_recon)
                self._log_figure(f"val/{i}_mel_real", fig_real)
                self._log_figure(f"val/{i}_mel_gen", fig_gen)

                # Log audio samples
                if audio_gen is not None:
                    audio_real = audio_real.cpu().numpy()
                    audio_gen = audio_gen.cpu().numpy()
                    self._log_audio(f"val/{i}_audio_real", audio_real)
                    self._log_audio(f"val/{i}_audio_gen", audio_gen)

            self.validation_examples = []

        # Clear vocoder to free memory
        self.vocoder = None

    def _log_figure(self, tag: str, fig):
        """Log a matplotlib figure to the logger."""
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_figure(tag, fig, self.global_step)
        elif isinstance(self.logger, WandbLogger):
            import PIL.Image as Image

            fig.canvas.draw()
            image = Image.frombytes("RGBa", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
            image = image.convert("RGB")
            self.logger.log_image(tag, [image], step=self.global_step)

    def _log_audio(self, tag: str, audio: np.ndarray):
        """Log an audio sample to the logger."""
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_audio(tag, audio, self.global_step, sample_rate=self.model.config.sample_rate)
        elif isinstance(self.logger, WandbLogger):
            self.logger.log_audio(
                tag, [audio.flatten()], sample_rate=[self.model.config.sample_rate], step=self.global_step
            )

    def _get_spectrogram_plot(self, mel: torch.Tensor):
        from matplotlib import pyplot as plt

        mel = mel.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma", vmin=-8.0, vmax=5.0)
        fig.colorbar(im, ax=ax)
        ax.set_ylabel("Mel bins")
        ax.set_xlabel("Time steps")
        fig.tight_layout()
        return fig

    def _load_weights(self, ckpt_path: str | None, model_state_dict: dict[str, torch.Tensor] | None = None):
        """Load model and discriminator weights from checkpoint. Supports .ckpt (Lightning), .safetensors, .pt/.pth formats.
        If model_state_dict is provided, load weights from it instead of ckpt_path."""

        def select_keys(state_dict: dict, prefix: str) -> dict:
            """Select keys from state_dict that start with the given prefix. Remove the prefix from keys."""
            return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

        def remove_prefix(state_dict: dict, prefix: str) -> dict:
            """Remove a prefix from keys that start with that prefix."""
            return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

        def add_prefix(state_dict: dict, prefix: str) -> dict:
            """Add a prefix to keys that do not start with that prefix."""
            return {f"{prefix}{k}" if not k.startswith(prefix) else k: v for k, v in state_dict.items()}

        # Load state dict
        if model_state_dict is not None:
            # Load from provided state dict
            disc_state_dict = {}
        elif ckpt_path.endswith(".ckpt"):
            # Lightning checkpoint
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model_state_dict = select_keys(checkpoint["state_dict"], "model.")
            disc_state_dict = select_keys(checkpoint["state_dict"], "discriminator.")
        elif ckpt_path.endswith(".safetensors"):
            # Safetensors checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(ckpt_path, device="cpu")
            model_state_dict = checkpoint
            disc_state_dict = {}
        elif ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
            # Standard PyTorch checkpoint
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model_state_dict = checkpoint
            disc_state_dict = {}
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

        # Load model weights
        model_state_dict = remove_prefix(model_state_dict, "_orig_mod.")
        model_state_dict = {
            k: v
            for k, v in model_state_dict.items()
            if not any(k.startswith(module) for module in self.config.skip_loading_modules)
        }
        if self.torch_compiled:
            model_state_dict = add_prefix(model_state_dict, "_orig_mod.")

        if len(model_state_dict) > 0:
            result = self.model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"Loaded model weights from {ckpt_path or 'provided state_dict'}.")
            if result.missing_keys:
                logger.debug(f"Missing keys in model state_dict: {result.missing_keys}")
            if result.unexpected_keys:
                logger.debug(f"Unexpected keys in model state_dict: {result.unexpected_keys}")

        # Load discriminator weights if available
        if self.use_discriminator:
            disc_state_dict = remove_prefix(disc_state_dict, "_orig_mod.")
            if self.torch_compiled:
                disc_state_dict = add_prefix(disc_state_dict, "_orig_mod.")

            if len(disc_state_dict) > 0:
                result = self.discriminator.load_state_dict(disc_state_dict, strict=False)
                logger.info(f"Loaded discriminator weights from {ckpt_path}.")
                if result.missing_keys:
                    logger.debug(f"Missing keys in discriminator state_dict: {result.missing_keys}")
                if result.unexpected_keys:
                    logger.debug(f"Unexpected keys in discriminator state_dict: {result.unexpected_keys}")

    @classmethod
    def from_hparams(cls, config_path: str) -> "KanadePipeline":
        """Instantiate KanadePipeline from config file.
        Args:
            config_path (str): Path to model configuration file (.yaml).
        Returns:
            KanadePipeline: Instantiated KanadePipeline.
        """
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Remove related fields to prevent loading actual weights here
        new_config = {"model": config["model"]}
        pipeline_config = new_config["model"]["init_args"]["pipeline_config"]
        if "ckpt_path" in pipeline_config:
            del pipeline_config["ckpt_path"]
        if "skip_loading_modules" in pipeline_config:
            del pipeline_config["skip_loading_modules"]

        # Instantiate model using jsonargparse
        parser = jsonargparse.ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=KanadePipeline)
        cfg = parser.parse_object(new_config)
        cfg = parser.instantiate_classes(cfg)
        return cfg.model

    @staticmethod
    def from_pretrained(config_path: str, ckpt_path: str) -> "KanadePipeline":
        """Load KanadePipeline from training configuration and checkpoint files.
        Args:
            config_path: Path to pipeline configuration file (YAML).
            ckpt_path: Path to checkpoint file (.ckpt) or model weights (.safetensors).
        Returns:
            KanadePipeline: Instantied KanadePipeline with loaded weights.
        """
        # Load pipeline from config
        model = KanadePipeline.from_hparams(config_path)
        # Load the weights
        model._load_weights(ckpt_path)
        return model
