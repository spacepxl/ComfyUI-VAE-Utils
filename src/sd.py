import math
import json
import torch
import logging

from comfy import model_management
from comfy.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
# from comfy.ldm.cascade.stage_a import StageA
# from comfy.ldm.cascade.stage_c_coder import StageC_coder
# from comfy.ldm.audio.autoencoder import AudioOobleckVAE
# import comfy.ldm.genmo.vae.model
# import comfy.ldm.lightricks.vae.causal_video_autoencoder
# import comfy.ldm.cosmos.vae
from .wan.vae import WanVAE
import comfy.ldm.wan.vae2_2
# import comfy.ldm.hunyuan3d.vae
# import comfy.ldm.ace.vae.music_dcae_pipeline
# import comfy.ldm.hunyuan_video.vae
# import comfy.ldm.mmaudio.vae.autoencoder
# import comfy.pixel_space_convert

import comfy.utils

import comfy.model_patcher
import comfy.taesd.taesd

from comfy.sd import VAE

class CustomVAE(VAE):
    def __init__(self, sd=None, device=None, config=None, dtype=None, metadata=None):
        if model_management.is_amd():
            VAE_KL_MEM_RATIO = 2.73
        else:
            VAE_KL_MEM_RATIO = 1.0

        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype) * VAE_KL_MEM_RATIO #These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype) * VAE_KL_MEM_RATIO
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.latent_dim = 2
        self.input_channels = 3
        self.output_channels = 3
        self.real_output_channels = 3
        self.pad_channel_value = None
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        self.working_dtypes = [torch.bfloat16, torch.float32]
        self.disable_offload = False
        self.not_video = False
        self.size = None

        self.downscale_index_formula = None
        self.upscale_index_formula = None
        self.extra_1d_channel = None
        self.crop_input = True

        if config is None:
            if "decoder.mid.block_1.mix_factor" in sd:
                encoder_config = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
                decoder_config = encoder_config.copy()
                decoder_config["video_kernel_size"] = [3, 1, 1]
                decoder_config["alpha"] = 0.0
                self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                            encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': encoder_config},
                                                            decoder_config={'target': "comfy.ldm.modules.temporal_ae.VideoDecoder", 'params': decoder_config})
            elif "taesd_decoder.1.weight" in sd:
                self.latent_channels = sd["taesd_decoder.1.weight"].shape[1]
                self.first_stage_model = comfy.taesd.taesd.TAESD(latent_channels=self.latent_channels)
            # elif "vquantizer.codebook.weight" in sd: #VQGan: stage a of stable cascade
                # self.first_stage_model = StageA()
                # self.downscale_ratio = 4
                # self.upscale_ratio = 4
                # self.process_input = lambda image: image
                # self.process_output = lambda image: image
            # elif "backbone.1.0.block.0.1.num_batches_tracked" in sd: #effnet: encoder for stage c latent of stable cascade
                # self.first_stage_model = StageC_coder()
                # self.downscale_ratio = 32
                # self.latent_channels = 16
                # new_sd = {}
                # for k in sd:
                    # new_sd["encoder.{}".format(k)] = sd[k]
                # sd = new_sd
            # elif "blocks.11.num_batches_tracked" in sd: #previewer: decoder for stage c latent of stable cascade
                # self.first_stage_model = StageC_coder()
                # self.latent_channels = 16
                # new_sd = {}
                # for k in sd:
                    # new_sd["previewer.{}".format(k)] = sd[k]
                # sd = new_sd
            # elif "encoder.backbone.1.0.block.0.1.num_batches_tracked" in sd: #combined effnet and previewer for stable cascade
                # self.first_stage_model = StageC_coder()
                # self.downscale_ratio = 32
                # self.latent_channels = 16
            elif "decoder.conv_in.weight" in sd:
                # if sd['decoder.conv_in.weight'].shape[1] == 64:
                    # ddconfig = {"block_out_channels": [128, 256, 512, 512, 1024, 1024], "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 32, "downsample_match_channel": True, "upsample_match_channel": True}
                    # self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.weight"].shape[1]
                    # self.downscale_ratio = 32
                    # self.upscale_ratio = 32
                    # self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                    # self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                                # encoder_config={'target': "comfy.ldm.hunyuan_video.vae.Encoder", 'params': ddconfig},
                                                                # decoder_config={'target': "comfy.ldm.hunyuan_video.vae.Decoder", 'params': ddconfig})

                    # self.memory_used_encode = lambda shape, dtype: (700 * shape[2] * shape[3]) * model_management.dtype_size(dtype)
                    # self.memory_used_decode = lambda shape, dtype: (700 * shape[2] * shape[3] * 32 * 32) * model_management.dtype_size(dtype)
                # elif sd['decoder.conv_in.weight'].shape[1] == 32:
                    # ddconfig = {"block_out_channels": [128, 256, 512, 1024, 1024], "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 16, "ffactor_temporal": 4, "downsample_match_channel": True, "upsample_match_channel": True, "refiner_vae": False}
                    # self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.weight"].shape[1]
                    # self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                    # self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
                    # self.upscale_index_formula = (4, 16, 16)
                    # self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
                    # self.downscale_index_formula = (4, 16, 16)
                    # self.latent_dim = 3
                    # self.not_video = True
                    # self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                                # encoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Encoder", 'params': ddconfig},
                                                                # decoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Decoder", 'params': ddconfig})

                    # self.memory_used_encode = lambda shape, dtype: (2800 * shape[-2] * shape[-1]) * model_management.dtype_size(dtype)
                    # self.memory_used_decode = lambda shape, dtype: (2800 * shape[-3] * shape[-2] * shape[-1] * 16 * 16) * model_management.dtype_size(dtype)
                # else:
                #default SD1.x/SD2.x VAE parameters
                ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}

                if 'encoder.down.2.downsample.conv.weight' not in sd and 'decoder.up.3.upsample.conv.weight' not in sd: #Stable diffusion x4 upscaler VAE
                    ddconfig['ch_mult'] = [1, 2, 4]
                    self.downscale_ratio = 4
                    self.upscale_ratio = 4

                self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.weight"].shape[1]
                if 'post_quant_conv.weight' in sd:
                    self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd['post_quant_conv.weight'].shape[1])
                else:
                    self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                                encoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Encoder", 'params': ddconfig},
                                                                decoder_config={'target': "comfy.ldm.modules.diffusionmodules.model.Decoder", 'params': ddconfig})
            # elif "decoder.layers.1.layers.0.beta" in sd:
                # self.first_stage_model = AudioOobleckVAE()
                # self.memory_used_encode = lambda shape, dtype: (1000 * shape[2]) * model_management.dtype_size(dtype)
                # self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * 2048) * model_management.dtype_size(dtype)
                # self.latent_channels = 64
                # self.output_channels = 2
                # self.upscale_ratio = 2048
                # self.downscale_ratio =  2048
                # self.latent_dim = 1
                # self.process_output = lambda audio: audio
                # self.process_input = lambda audio: audio
                # self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                # self.disable_offload = True
            # elif "blocks.2.blocks.3.stack.5.weight" in sd or "decoder.blocks.2.blocks.3.stack.5.weight" in sd or "layers.4.layers.1.attn_block.attn.qkv.weight" in sd or "encoder.layers.4.layers.1.attn_block.attn.qkv.weight" in sd: #genmo mochi vae
                # if "blocks.2.blocks.3.stack.5.weight" in sd:
                    # sd = comfy.utils.state_dict_prefix_replace(sd, {"": "decoder."})
                # if "layers.4.layers.1.attn_block.attn.qkv.weight" in sd:
                    # sd = comfy.utils.state_dict_prefix_replace(sd, {"": "encoder."})
                # self.first_stage_model = comfy.ldm.genmo.vae.model.VideoVAE()
                # self.latent_channels = 12
                # self.latent_dim = 3
                # self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * shape[3] * shape[4] * (6 * 8 * 8)) * model_management.dtype_size(dtype)
                # self.memory_used_encode = lambda shape, dtype: (1.5 * max(shape[2], 7) * shape[3] * shape[4] * (6 * 8 * 8)) * model_management.dtype_size(dtype)
                # self.upscale_ratio = (lambda a: max(0, a * 6 - 5), 8, 8)
                # self.upscale_index_formula = (6, 8, 8)
                # self.downscale_ratio = (lambda a: max(0, math.floor((a + 5) / 6)), 8, 8)
                # self.downscale_index_formula = (6, 8, 8)
                # self.working_dtypes = [torch.float16, torch.float32]
            # elif "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd: #lightricks ltxv
                # tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
                # version = 0
                # if tensor_conv1.shape[0] == 512:
                    # version = 0
                # elif tensor_conv1.shape[0] == 1024:
                    # version = 1
                    # if "encoder.down_blocks.1.conv.conv.bias" in sd:
                        # version = 2
                # vae_config = None
                # if metadata is not None and "config" in metadata:
                    # vae_config = json.loads(metadata["config"]).get("vae", None)
                # self.first_stage_model = comfy.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE(version=version, config=vae_config)
                # self.latent_channels = 128
                # self.latent_dim = 3
                # self.memory_used_decode = lambda shape, dtype: (900 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)) * model_management.dtype_size(dtype)
                # self.memory_used_encode = lambda shape, dtype: (70 * max(shape[2], 7) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                # self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)
                # self.upscale_index_formula = (8, 32, 32)
                # self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 32, 32)
                # self.downscale_index_formula = (8, 32, 32)
                # self.working_dtypes = [torch.bfloat16, torch.float32]
            # elif "decoder.conv_in.conv.weight" in sd and sd['decoder.conv_in.conv.weight'].shape[1] == 32:
                # ddconfig = {"block_out_channels": [128, 256, 512, 1024, 1024], "in_channels": 3, "out_channels": 3, "num_res_blocks": 2, "ffactor_spatial": 16, "ffactor_temporal": 4, "downsample_match_channel": True, "upsample_match_channel": True}
                # ddconfig['z_channels'] = sd["decoder.conv_in.conv.weight"].shape[1]
                # self.latent_channels = 64
                # self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
                # self.upscale_index_formula = (4, 16, 16)
                # self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
                # self.downscale_index_formula = (4, 16, 16)
                # self.latent_dim = 3
                # self.not_video = True
                # self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                # self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "comfy.ldm.models.autoencoder.EmptyRegularizer"},
                                                            # encoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Encoder", 'params': ddconfig},
                                                            # decoder_config={'target': "comfy.ldm.hunyuan_video.vae_refiner.Decoder", 'params': ddconfig})

                # self.memory_used_encode = lambda shape, dtype: (1400 * shape[-2] * shape[-1]) * model_management.dtype_size(dtype)
                # self.memory_used_decode = lambda shape, dtype: (1400 * shape[-3] * shape[-2] * shape[-1] * 16 * 16) * model_management.dtype_size(dtype)
            elif "decoder.conv_in.conv.weight" in sd:
                ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
                ddconfig["conv3d"] = True
                ddconfig["time_compress"] = 4
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3
                self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.conv.weight"].shape[1]
                self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd['post_quant_conv.weight'].shape[1])
                self.memory_used_decode = lambda shape, dtype: (1500 * shape[2] * shape[3] * shape[4] * (4 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (900 * max(shape[2], 2) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
            # elif "decoder.unpatcher3d.wavelets" in sd:
                # self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 8, 8)
                # self.upscale_index_formula = (8, 8, 8)
                # self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 8, 8)
                # self.downscale_index_formula = (8, 8, 8)
                # self.latent_dim = 3
                # self.latent_channels = 16
                # ddconfig = {'z_channels': 16, 'latent_channels': self.latent_channels, 'z_factor': 1, 'resolution': 1024, 'in_channels': 3, 'out_channels': 3, 'channels': 128, 'channels_mult': [2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [32], 'dropout': 0.0, 'patch_size': 4, 'num_groups': 1, 'temporal_compression': 8, 'spacial_compression': 8}
                # self.first_stage_model = comfy.ldm.cosmos.vae.CausalContinuousVideoTokenizer(**ddconfig)
                # TODO: these values are a bit off because this is not a standard VAE
                # self.memory_used_decode = lambda shape, dtype: (50 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)) * model_management.dtype_size(dtype)
                # self.memory_used_encode = lambda shape, dtype: (50 * (round((shape[2] + 7) / 8) * 8) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                # self.working_dtypes = [torch.bfloat16, torch.float32]
            elif "decoder.middle.0.residual.0.gamma" in sd:
                if "decoder.upsamples.0.upsamples.0.residual.2.weight" in sd:  # Wan 2.2 VAE
                    self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
                    self.upscale_index_formula = (4, 16, 16)
                    self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
                    self.downscale_index_formula = (4, 16, 16)
                    self.latent_dim = 3
                    self.latent_channels = 48
                    ddconfig = {"dim": 160, "z_dim": self.latent_channels, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_scales": [], "temperal_downsample": [False, True, True], "dropout": 0.0}
                    self.first_stage_model = comfy.ldm.wan.vae2_2.WanVAE(**ddconfig)
                    self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                    self.memory_used_encode = lambda shape, dtype: 3300 * shape[3] * shape[4] * model_management.dtype_size(dtype)
                    self.memory_used_decode = lambda shape, dtype: 8000 * shape[3] * shape[4] * (16 * 16) * model_management.dtype_size(dtype)
                else:  # Wan 2.1 VAE
                    self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                    self.upscale_index_formula = (4, 8, 8)
                    self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                    self.downscale_index_formula = (4, 8, 8)
                    self.input_channels = sd["encoder.conv1.weight"].shape[1]
                    # self.output_channels = sd["decoder.head.2.weight"].shape[0]
                    self.output_channels = self.input_channels
                    self.real_output_channels = sd["decoder.head.2.weight"].shape[0]
                    self.latent_dim = 3
                    self.latent_channels = 16
                    self.pad_channel_value = 1.0
                    
                    ddconfig = {
                        "in_channels": self.input_channels,
                        # "out_channels": self.output_channels,
                        "out_channels": self.real_output_channels,
                        "dim": 96,
                        "z_dim": self.latent_channels,
                        "dim_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                        "attn_scales": [],
                        "temperal_downsample": [False, True, True],
                        "dropout": 0.0,
                    }
                    self.first_stage_model = WanVAE(**ddconfig)
                    self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                    self.memory_used_encode = lambda shape, dtype: 6000 * shape[3] * shape[4] * model_management.dtype_size(dtype)
                    self.memory_used_decode = lambda shape, dtype: 7000 * shape[3] * shape[4] * (8 * 8) * model_management.dtype_size(dtype)
            # Hunyuan 3d v2 2.0 & 2.1
            # elif "geo_decoder.cross_attn_decoder.ln_1.bias" in sd:
                # self.latent_dim = 1

                # def estimate_memory(shape, dtype, num_layers = 16, kv_cache_multiplier = 2):
                    # batch, num_tokens, hidden_dim = shape
                    # dtype_size = model_management.dtype_size(dtype)

                    # total_mem = batch * num_tokens * hidden_dim * dtype_size * (1 + kv_cache_multiplier * num_layers)
                    # return total_mem

                # better memory estimations
                # self.memory_used_encode = lambda shape, dtype, num_layers = 8, kv_cache_multiplier = 0:\
                    # estimate_memory(shape, dtype, num_layers, kv_cache_multiplier)

                # self.memory_used_decode = lambda shape, dtype, num_layers = 16, kv_cache_multiplier = 2: \
                    # estimate_memory(shape, dtype, num_layers, kv_cache_multiplier)

                # self.first_stage_model = comfy.ldm.hunyuan3d.vae.ShapeVAE()
                # self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
            # elif "vocoder.backbone.channel_layers.0.0.bias" in sd: #Ace Step Audio
                # self.first_stage_model = comfy.ldm.ace.vae.music_dcae_pipeline.MusicDCAE(source_sample_rate=44100)
                # self.memory_used_encode = lambda shape, dtype: (shape[2] * 330) * model_management.dtype_size(dtype)
                # self.memory_used_decode = lambda shape, dtype: (shape[2] * shape[3] * 87000) * model_management.dtype_size(dtype)
                # self.latent_channels = 8
                # self.output_channels = 2
                # self.upscale_ratio = 4096
                # self.downscale_ratio = 4096
                # self.latent_dim = 2
                # self.process_output = lambda audio: audio
                # self.process_input = lambda audio: audio
                # self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                # self.disable_offload = True
                # self.extra_1d_channel = 16
            # elif "pixel_space_vae" in sd:
                # self.first_stage_model = comfy.pixel_space_convert.PixelspaceConversionVAE()
                # self.memory_used_encode = lambda shape, dtype: (1 * shape[2] * shape[3]) * model_management.dtype_size(dtype)
                # self.memory_used_decode = lambda shape, dtype: (1 * shape[2] * shape[3]) * model_management.dtype_size(dtype)
                # self.downscale_ratio = 1
                # self.upscale_ratio = 1
                # self.latent_channels = 3
                # self.latent_dim = 2
                # self.output_channels = 3
            # elif "vocoder.activation_post.downsample.lowpass.filter" in sd: #MMAudio VAE
                # sample_rate = 16000
                # if sample_rate == 16000:
                    # mode = '16k'
                # else:
                    # mode = '44k'

                # self.first_stage_model = comfy.ldm.mmaudio.vae.autoencoder.AudioAutoencoder(mode=mode)
                # self.memory_used_encode = lambda shape, dtype: (30 * shape[2]) * model_management.dtype_size(dtype)
                # self.memory_used_decode = lambda shape, dtype: (90 * shape[2] * 1411.2) * model_management.dtype_size(dtype)
                # self.latent_channels = 20
                # self.output_channels = 2
                # self.upscale_ratio = 512 * (44100 / sample_rate)
                # self.downscale_ratio = 512 * (44100 / sample_rate)
                # self.latent_dim = 1
                # self.process_output = lambda audio: audio
                # self.process_input = lambda audio: audio
                # self.working_dtypes = [torch.float32]
                # self.crop_input = False
            else:
                logging.warning("WARNING: No VAE weights detected, VAE not initalized.")
                self.first_stage_model = None
                return
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']))
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logging.warning("Missing VAE keys {}".format(m))

        if len(u) > 0:
            logging.debug("Leftover VAE keys {}".format(u))

        if device is None:
            device = model_management.vae_device()
        self.device = device
        offload_device = model_management.vae_offload_device()
        if dtype is None:
            dtype = model_management.vae_dtype(self.device, self.working_dtypes)
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = model_management.intermediate_device()

        self.patcher = comfy.model_patcher.ModelPatcher(self.first_stage_model, load_device=self.device, offload_device=offload_device)
        logging.info("VAE load device: {}, offload device: {}, dtype: {}".format(self.device, offload_device, self.vae_dtype))

    def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
        return self.process_output(
            comfy.utils.tiled_scale_multidim(
                samples,
                decode_fn,
                tile=(tile_t, tile_x, tile_y),
                overlap=overlap,
                upscale_amount=self.upscale_ratio,
                out_channels=self.real_output_channels,
                index_formulas=self.upscale_index_formula,
                output_device=self.output_device,
            )
        )
