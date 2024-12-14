import torch
import os
import logging
import folder_paths
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import numpy as np
from .attention_processor import IPAFluxAttnProcessor2_0
from .utils import is_model_patched, FluxUpdateModules

MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter-flux")
if "ipadapter-flux" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
folder_paths.folder_names_and_paths["ipadapter-flux"] = (current_paths, folder_paths.supported_pt_extensions)

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class InstantXFluxIPAdapterModel:
    def __init__(self, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        # load image encoder
        self.image_encoder = SiglipVisionModel.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
        self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
        # state_dict
        self.state_dict = torch.load(os.path.join(MODELS_DIR,self.ip_ckpt), map_location="cpu")
        self.joint_attention_dim = 4096
        self.hidden_size = 3072

    def init_proj(self):
        self.image_proj_model = MLPProjModel(
            cross_attention_dim=self.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)

    def set_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        ip_attn_procs = {} # 19+38=57
        dsb_count = len(flux_model.diffusion_model.double_blocks)
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale = weight,
                    timestep_range = timestep_range
                ).to(self.device, dtype=torch.float16)
        ssb_count = len(flux_model.diffusion_model.single_blocks)
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale = weight,
                    timestep_range = timestep_range
                ).to(self.device, dtype=torch.float16)
        return ip_attn_procs
    
    def load_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        self.image_proj_model.load_state_dict(self.state_dict["image_proj"], strict=True)
        ip_attn_procs = self.set_ip_adapter(flux_model, weight, timestep_percent_range)
        ip_layers = torch.nn.ModuleList(ip_attn_procs.values())
        ip_layers.load_state_dict(self.state_dict["ip_adapter"], strict=True)
        return ip_attn_procs

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=torch.float16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

class IPAdapterFluxLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                "provider": (["cuda", "cpu", "mps"],),
                }
            }
    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, ipadapter, provider):
        logging.info("Loading InstantX IPAdapter Flux model.")
        clip_path = os.path.join(folder_paths.models_dir, "clip", "siglip-so400m-patch14-384")
        model = InstantXFluxIPAdapterModel(image_encoder_path=clip_path, ip_ckpt=ipadapter, device=provider, num_tokens=128)
        return (model,)

class ApplyIPAdapterFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX", ),
                "image": ("IMAGE", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent):
        # convert image to pillow
        pil_image = image.numpy()[0] * 255.0
        pil_image = Image.fromarray(pil_image.astype(np.uint8))
        # initialize ipadapter
        ipadapter_flux.init_proj()
        ip_attn_procs = ipadapter_flux.load_ip_adapter(model.model, weight, (start_percent, end_percent))
        # process control image 
        image_prompt_embeds = ipadapter_flux.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=None
        )
        # set model
        is_patched = is_model_patched(model.model)
        bi = model.clone()
        FluxUpdateModules(bi, ip_attn_procs, image_prompt_embeds, is_patched)
        return (bi,)

NODE_CLASS_MAPPINGS = {
    "IPAdapterFluxLoader": IPAdapterFluxLoader,
    "ApplyIPAdapterFlux": ApplyIPAdapterFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterFluxLoader": "Load IPAdapter Flux Model",
    "ApplyIPAdapterFlux": "Apply IPAdapter Flux Model",
}
