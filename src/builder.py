
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from src.utils_model import safe_load_ctclip
import torch


                      
def build_ct_clip(args, device: torch.device) -> tuple[CTCLIP, BertTokenizer]:
    """
    Build CT-CLIP + text encoder in the same spirit as generate_embeddings.py.
    We won't use the vision backbone here (we pass img_embed directly).
    """
    # text side
    tokenizer = BertTokenizer.from_pretrained(args.biomed_bert_path, do_lower_case=True)
    text_encoder = BertModel.from_pretrained(args.biomed_bert_path)
    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim=512, codebook_size=8192,
        image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4,
        temporal_depth=4, dim_head=32, heads=8
    )


    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912, dim_text=768, dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False
    )
    safe_load_ctclip(model=clip, ckpt_path=args.ct_clip_path)
    clip.to(device).eval()
    
    return clip, tokenizer

