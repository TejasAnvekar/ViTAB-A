"""
Unified model handler for all supported vision-language models.

Supports: Gemma, InternVL 3.5, Qwen3-VL, and Molmo2.
Routes to the correct handler class based on the model name.
"""
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import torch.nn.functional as F
from dataclasses import dataclass
from PIL import Image
import torch

logger = logging.getLogger(__name__)


def _detect_model_family(model_name: str) -> str:
    """Detect which model family a HuggingFace model ID belongs to.

    Returns:
        "internvl" for InternVL and Gemma models
        "qwen" for Qwen3-VL and Molmo2 models
    """
    name_lower = model_name.lower()
    if "internvl" in name_lower or "gemma" in name_lower:
        return "internvl"
    if "qwen" in name_lower or "molmo" in name_lower:
        return "qwen"
    raise ValueError(
        f"Cannot detect model family for '{model_name}'. "
        "Expected model name to contain one of: internvl, gemma, qwen, molmo."
    )


def _summarize_hf_device_map(device_map: Any) -> Dict[str, int]:
    """Summarize an Accelerate/HF device map.

    Returns counts of modules assigned per device string (e.g., "cuda:0", "cpu", "disk").
    """
    summary: Dict[str, int] = {}
    if not isinstance(device_map, dict):
        return summary

    cuda_available = torch.cuda.is_available()

    def _normalize_device(dev: Any) -> str:
        if isinstance(dev, torch.device):
            return str(dev)
        if isinstance(dev, int):
            return f"cuda:{dev}" if cuda_available else str(dev)
        dev_str = str(dev)
        if cuda_available and dev_str.isdigit():
            return f"cuda:{dev_str}"
        # Common accelerate labels: 'cpu', 'disk', 'cuda:0'
        return dev_str

    for _, dev in device_map.items():
        dev_str = _normalize_device(dev)
        summary[dev_str] = summary.get(dev_str, 0) + 1
    return summary


def _log_runtime_device_info(model: Any) -> None:
    """Log information that helps confirm GPU usage and detect offloading."""
    try:
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        if cuda_available:
            try:
                device_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                total_gb = float(props.total_memory) / (1024 ** 3)
                logger.info(f"CUDA device 0: {device_name} ({total_gb:.1f} GiB)")
                logger.info(
                    "CUDA mem (GiB): "
                    f"allocated={torch.cuda.memory_allocated(0)/(1024**3):.2f}, "
                    f"reserved={torch.cuda.memory_reserved(0)/(1024**3):.2f}"
                )
            except Exception as e:
                logger.info(f"CUDA device query failed (non-fatal): {e}")

        # Hugging Face / Accelerate device placement summary
        device_map = getattr(model, "hf_device_map", None) or getattr(model, "_hf_device_map", None)
        if isinstance(device_map, dict):
            summary = _summarize_hf_device_map(device_map)
            logger.info(f"hf_device_map summary (module shards per device): {summary}")

            offload_targets = {k for k in summary.keys() if k.startswith("cpu") or k.startswith("disk")}
            if offload_targets:
                logger.warning(
                    "Model appears partially offloaded to non-GPU devices: "
                    f"{sorted(offload_targets)}. This is expected if GPU VRAM is insufficient and can slow inference."
                )
        else:
            # Fallback: best-effort device from first parameter
            try:
                first_param_device = next(model.parameters()).device
                logger.info(f"Model parameter device (first param): {first_param_device}")
            except Exception:
                pass
    except Exception as e:
        logger.info(f"Device info logging failed (non-fatal): {e}")


@dataclass
class InferenceResult:
    """Result from a single model inference"""
    output_text: str
    inference_time_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw_input: Optional[Dict[str, Any]] = None
    # Added for confidence extraction (Confidence Under the Hood paper)
    logits: Optional[torch.Tensor] = None  # All generated token logits stacked [seq_len, vocab_size]
    token_probabilities: Optional[Dict[int, float]] = None  # Token ID -> probability (first token only, for backwards compat)
    generated_token_ids: Optional[List[int]] = None  # All generated token IDs


# ---------------------------------------------------------------------------
# InternVL / Gemma handler
# ---------------------------------------------------------------------------

class InternVLModel:
    """Handler for InternVL and Gemma models, following official HF documentation"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = False,
        use_torch_compile: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.use_flash_attention = use_flash_attention
        self.use_torch_compile = use_torch_compile
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load InternVL model and processor following official documentation"""
        if self._loaded:
            return

        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info(f"Loading InternVL model: {self.model_name}")

        # Load processor (trust_remote_code required for InternVL)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model using InternVL's image-text generation class
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).eval()

        # Log device placement and potential offloading (cpu/disk) for verification.
        _log_runtime_device_info(self.model)

        # Apply torch.compile if enabled
        if self.use_torch_compile:
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compilation complete")

        self._loaded = True
        logger.info(f"InternVL model loaded successfully: {self.model_name}")

    def unload(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info(f"Model unloaded: {self.model_name}")

    def generate_text_only(
        self,
        prompt: str,
        return_logits: bool = False
    ) -> InferenceResult:
        """
        Text-only generation for InternVL (JSON/Markdown representations).
        """
        if not self._loaded:
            self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.model.device)

        raw_input = {
            "type": "text_only",
            "prompt": prompt,
        }

        start_time = time.perf_counter()

        all_token_logits = None
        token_probs = None

        with torch.no_grad():
            if return_logits:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences

                if outputs.scores:
                    all_token_logits = torch.stack(
                        [s[0].float().cpu() for s in outputs.scores],
                        dim=0
                    )

                    probs = F.softmax(outputs.scores[0][0], dim=-1)
                    top_k = 100
                    top_probs, top_indices = torch.topk(probs, top_k)
                    token_probs = {
                        int(idx): float(prob)
                        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                    }
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        gen_tokens_list = generated_ids_trimmed[0].tolist() if generated_ids_trimmed else []

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return InferenceResult(
            output_text=output_text.strip(),
            inference_time_ms=inference_time_ms,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(gen_tokens_list),
            raw_input=raw_input,
            logits=all_token_logits,
            token_probabilities=token_probs,
            generated_token_ids=gen_tokens_list
        )

    def generate_with_image(
        self,
        prompt: str,
        image: Image.Image,
        example_image: Optional[Image.Image] = None,
        return_logits: bool = False
    ) -> InferenceResult:
        """
        Generate response for image + text input following official InternVL documentation.
        """
        if not self._loaded:
            self.load()

        content = []
        if example_image is not None:
            if "<IMAGE>" not in prompt:
                raise ValueError(
                    "Few-shot image prompting requires exactly two <IMAGE> markers in the prompt "
                    "to place example and main images."
                )

            parts = prompt.split("<IMAGE>")
            if len(parts) != 3:
                raise ValueError(
                    f"Expected exactly two <IMAGE> markers, found {len(parts) - 1}."
                )

            content.append({"type": "text", "text": parts[0]})
            content.append({"type": "image", "image": example_image})
            content.append({"type": "text", "text": parts[1]})
            content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": parts[2]})
            logger.info(
                "Multi-image message structure: "
                f"[text, example_image({example_image.size}), text, main_image({image.size}), text]"
            )
        else:
            if "<IMAGE>" in prompt:
                raise ValueError(
                    "Prompt contains <IMAGE> markers but no example image was provided."
                )
            content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            logger.debug(f"Single image: size={image.size}")

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Move to device (do NOT pass dtype to .to() - BatchEncoding doesn't support it)
        inputs = inputs.to(self.model.device)

        # Manually cast pixel_values to the correct dtype if present
        if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "to"):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.torch_dtype)

        raw_input = {
            "type": "image_text",
            "prompt": prompt,
            "image_size": image.size,
            "image_mode": image.mode
        }

        start_time = time.perf_counter()

        all_token_logits = None
        token_probs = None

        with torch.no_grad():
            if return_logits:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences

                if outputs.scores:
                    all_token_logits = torch.stack(
                        [s[0].float().cpu() for s in outputs.scores],
                        dim=0
                    )

                    probs = F.softmax(outputs.scores[0][0], dim=-1)
                    top_k = 100
                    top_probs, top_indices = torch.topk(probs, top_k)
                    token_probs = {
                        int(idx): float(prob)
                        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                    }
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        gen_tokens_list = generated_ids_trimmed[0].tolist() if generated_ids_trimmed else []

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return InferenceResult(
            output_text=output_text.strip(),
            inference_time_ms=inference_time_ms,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(gen_tokens_list),
            raw_input=raw_input,
            logits=all_token_logits,
            token_probabilities=token_probs,
            generated_token_ids=gen_tokens_list
        )

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        return_logits: bool = False
    ) -> InferenceResult:
        """Unified generate method for InternVL/Gemma inputs."""
        if image is None:
            return self.generate_text_only(prompt, return_logits=return_logits)
        return self.generate_with_image(prompt, image, return_logits=return_logits)


# ---------------------------------------------------------------------------
# Qwen3-VL / Molmo2 handler
# ---------------------------------------------------------------------------

class Qwen3VLModel:
    """Handler for Qwen3-VL and Molmo2 models"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        use_torch_compile: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.use_flash_attention = use_flash_attention
        self.use_torch_compile = use_torch_compile
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load model and processor"""
        if self._loaded:
            return

        from transformers import AutoProcessor

        logger.info(f"Loading model: {self.model_name}")

        trust_remote_code = False
        # Detect if this is a Molmo2 model
        is_molmo2 = "molmo2" in self.model_name.lower()

        if "30b" in self.model_name.lower() or "a3b" in self.model_name.lower():
            logger.info(f"Enabling trust_remote_code=True for {self.model_name}")
            trust_remote_code = True

        # Molmo2 models require trust_remote_code
        if is_molmo2:
            logger.info(f"Detected Molmo2 model, enabling trust_remote_code=True")
            trust_remote_code = True

        # Use different model class depending on model type
        if is_molmo2:
            logger.info("Using AutoModelForImageTextToText for Molmo2 model with dtype='auto'")
            from transformers import AutoModelForImageTextToText

            # Molmo2 models require dtype="auto" as string (not torch dtype object)
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    dtype="auto",
                    device_map="auto",
                    trust_remote_code=trust_remote_code
                )
            except Exception as e:
                logger.warning(f"Failed to load Molmo2 model components: {e}")
                logger.warning("Attempting to reload with force_download=True to fix potential cache corruption...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    dtype="auto",
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    force_download=True
                )
        else:
            logger.info("Using Qwen3VLForConditionalGeneration for Qwen model")
            from transformers import Qwen3VLForConditionalGeneration

            try:
                if self.use_flash_attention:
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        dtype=self.torch_dtype,
                        attn_implementation="flash_attention_2",
                        device_map="auto",
                        trust_remote_code=trust_remote_code
                    )
                else:
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        dtype=self.torch_dtype,
                        device_map="auto",
                        trust_remote_code=trust_remote_code
                    )
            except Exception as e:
                logger.warning(f"Failed to load with specified settings, trying with dtype='auto': {e}")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    dtype="auto",
                    device_map="auto",
                    trust_remote_code=trust_remote_code
                )

        # Load processor with appropriate parameters
        if is_molmo2:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=trust_remote_code,
                    dtype="auto",
                    device_map="auto"
                )
            except Exception as e:
                logger.warning(f"Failed to load Molmo2 processor: {e}")
                logger.warning("Attempting to reload processor with force_download=True...")
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=trust_remote_code,
                    dtype="auto",
                    device_map="auto",
                    force_download=True
                )
        else:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code
            )

        # Log device placement and potential offloading (cpu/disk) for verification.
        _log_runtime_device_info(self.model)

        # Apply torch.compile if enabled
        if self.use_torch_compile:
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compilation complete")

        self._loaded = True
        logger.info(f"Model loaded successfully: {self.model_name}")

    def unload(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info(f"Model unloaded: {self.model_name}")

    def generate_text_only(
        self,
        prompt: str,
        return_logits: bool = False
    ) -> InferenceResult:
        """
        Generate response for text-only input (JSON or Markdown tables)
        """
        if not self._loaded:
            self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
            max_length=None
        )
        inputs = inputs.to(self.model.device)

        raw_input = {
            "type": "text_only",
            "prompt": prompt,
            "messages": messages
        }

        start_time = time.perf_counter()

        all_token_logits = None
        token_probs = None

        with torch.no_grad():
            if return_logits:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences

                if outputs.scores:
                    all_token_logits = torch.stack(
                        [s[0].float().cpu() for s in outputs.scores],
                        dim=0
                    )

                    probs = F.softmax(outputs.scores[0][0], dim=-1)
                    top_k = 100
                    top_probs, top_indices = torch.topk(probs, top_k)
                    token_probs = {
                        int(idx): float(prob)
                        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                    }
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        gen_tokens_list = generated_ids_trimmed[0].tolist() if generated_ids_trimmed else []

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return InferenceResult(
            output_text=output_text.strip(),
            inference_time_ms=inference_time_ms,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(gen_tokens_list),
            raw_input=raw_input,
            logits=all_token_logits,
            token_probabilities=token_probs,
            generated_token_ids=gen_tokens_list
        )

    def generate_with_image(
        self,
        prompt: str,
        image: Image.Image,
        example_image: Optional[Image.Image] = None,
        return_logits: bool = False
    ) -> InferenceResult:
        """
        Generate response for image + text input.
        """
        if not self._loaded:
            self.load()

        content = []
        if example_image is not None:
            if "<IMAGE>" in prompt:
                parts = prompt.split("<IMAGE>")
                if len(parts) == 3:
                    content.append({"type": "text", "text": parts[0]})
                    content.append({"type": "image", "image": example_image})
                    content.append({"type": "text", "text": parts[1]})
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": parts[2]})
                    logger.info(f"Multi-image message structure: [text, example_image({example_image.size}), text, main_image({image.size}), text]")
                else:
                    logger.warning(f"Unexpected <IMAGE> marker count: {len(parts)-1}")
                    content.append({"type": "image", "image": example_image})
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": prompt.replace("<IMAGE>", "")})
                    logger.info(f"Multi-image fallback: [example_image({example_image.size}), main_image({image.size}), text]")
            else:
                if "Now analyze this table:" in prompt:
                    parts = prompt.split("Now analyze this table:")
                    example_part = parts[0]
                    main_part = "Now analyze this table:" + parts[1]

                    content.append({"type": "image", "image": example_image})
                    content.append({"type": "text", "text": example_part})
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": main_part})
                    logger.info(f"Text-based few-shot with images: [example_image({example_image.size}), text, main_image({image.size}), text]")
                else:
                    content.append({"type": "image", "image": example_image})
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": prompt})
                    logger.info(f"Few-shot fallback: [example_image({example_image.size}), main_image({image.size}), text]")
        else:
            content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            logger.debug(f"Single image: size={image.size}")

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
            max_length=None
        )
        inputs = inputs.to(self.model.device)

        raw_input = {
            "type": "image_text",
            "prompt": prompt,
            "image_size": image.size,
            "image_mode": image.mode
        }

        start_time = time.perf_counter()

        all_token_logits = None
        token_probs = None

        with torch.no_grad():
            if return_logits:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences

                if outputs.scores:
                    all_token_logits = torch.stack(
                        [s[0].float().cpu() for s in outputs.scores],
                        dim=0
                    )

                    probs = F.softmax(outputs.scores[0][0], dim=-1)
                    top_k = 100
                    top_probs, top_indices = torch.topk(probs, top_k)
                    token_probs = {
                        int(idx): float(prob)
                        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                    }
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        gen_tokens_list = generated_ids_trimmed[0].tolist() if generated_ids_trimmed else []

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return InferenceResult(
            output_text=output_text.strip(),
            inference_time_ms=inference_time_ms,
            input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(gen_tokens_list),
            raw_input=raw_input,
            logits=all_token_logits,
            token_probabilities=token_probs,
            generated_token_ids=gen_tokens_list
        )

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        return_logits: bool = False
    ) -> InferenceResult:
        """Unified generate method for Qwen3-VL/Molmo2 inputs."""
        if image is not None:
            return self.generate_with_image(
                prompt=prompt,
                image=image,
                example_image=None,
                return_logits=return_logits
            )
        else:
            return self.generate_text_only(prompt, return_logits)


# ---------------------------------------------------------------------------
# Unified ModelManager with automatic routing
# ---------------------------------------------------------------------------

class ModelManager:
    """Manager for loading/unloading multiple models efficiently.

    Automatically detects the model family from the HuggingFace model ID
    and instantiates the appropriate handler class.
    """

    def __init__(self, config: Any):
        self.config = config
        self.current_model: Optional[Union[InternVLModel, Qwen3VLModel]] = None
        self.current_model_name: Optional[str] = None

    def get_model(self, model_name: str) -> Union[InternVLModel, Qwen3VLModel]:
        """
        Get a model, loading it if necessary and unloading the previous one.
        Automatically routes to the correct handler class based on model name.
        """
        if self.current_model_name == model_name and self.current_model is not None:
            return self.current_model

        # Unload current model if any
        if self.current_model is not None:
            logger.info(f"Unloading model: {self.current_model_name}")
            self.current_model.unload()

        # Detect model family and instantiate appropriate handler
        family = _detect_model_family(model_name)
        logger.info(f"Loading model: {model_name} (family: {family})")

        if family == "internvl":
            self.current_model = InternVLModel(
                model_name=model_name,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype,
                use_flash_attention=self.config.use_flash_attention,
                use_torch_compile=self.config.use_torch_compile,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample
            )
        else:  # "qwen"
            self.current_model = Qwen3VLModel(
                model_name=model_name,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype,
                use_flash_attention=self.config.use_flash_attention,
                use_torch_compile=self.config.use_torch_compile,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample
            )

        self.current_model.load()
        self.current_model_name = model_name

        return self.current_model

    def cleanup(self):
        """Cleanup all resources"""
        if self.current_model is not None:
            self.current_model.unload()
            self.current_model = None
            self.current_model_name = None
