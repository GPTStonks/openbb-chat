import guidance
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig


class GuidanceWrapper:
    """Wrapper around `guidance` to be used with HF models."""

    def __init__(self, model_id: str = "openlm-research/open_llama_3b_v2"):
        """Init method.

        Args:
            model_id (`str`):
                Name of the HF model to use.
        """

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # set the default language model used to execute guidance programs
        guidance.llm = guidance.llms.Transformers(
            model=model_id,
            tokenizer=tokenizer,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )

    def __call__(self, *args, **kwargs):
        """Calls `guidance.__call__` passing all the arguments to it."""
        return guidance(*args, **kwargs)
