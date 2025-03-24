import torch
from logging_config import logger
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from fastapi import HTTPException

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

class LLMManager:
    def __init__(self, model_name: str, device: str = DEVICE):
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = TORCH_DTYPE
        self.model = None
        self.is_loaded = False
        self.processor = None
        self.token_cache = {}
        logger.info(f"LLMManager initialized with model {model_name} on {self.device}")

    def unload(self):
        if self.is_loaded:
            del self.model
            del self.processor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.info(f"GPU memory allocated after unload: {torch.cuda.memory_allocated()}")
            self.is_loaded = False
            logger.info(f"LLM {self.model_name} unloaded from {self.device}")

    def load(self):
        if not self.is_loaded:
            try:
                # Enable TF32 for better performance on supported GPUs
                if self.device.type == "cuda":
                    torch.set_float32_matmul_precision('high')
                    logger.info("Enabled TF32 matrix multiplication for improved performance")

                # Configure 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # NF4 (Normal Float 4) is optimized for LLMs
                    bnb_4bit_compute_dtype=self.torch_dtype,  # Use bfloat16 for computations
                    bnb_4bit_use_double_quant=True  # Nested quantization for better accuracy
                )

                # Load model with 4-bit quantization
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=self.torch_dtype,
                    max_memory={0: "10GiB"}  # Adjust based on your GPU capacity
                ).eval()

                # Move model to device (handled by device_map, but explicit for clarity)
                self.model.to(self.device)

                # Load processor with use_fast=True for faster tokenization
                self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
                self.is_loaded = True
                logger.info(f"LLM {self.model_name} loaded on {self.device} with 4-bit quantization and fast processor")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    async def generate(self, prompt: str, max_tokens: int = 50) -> str:
        if not self.is_loaded:
            self.load()

        cache_key = f"system_prompt_{prompt}"
        if cache_key in self.token_cache and "response" in self.token_cache[cache_key]:
            logger.info("Using cached response")
            return self.token_cache[cache_key]["response"]

        if cache_key in self.token_cache and "inputs" in self.token_cache[cache_key]:
            inputs_vlm = self.token_cache[cache_key]["inputs"]
            logger.info("Using cached tokenized input")
        else:
            messages_vlm = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            try:
                inputs_vlm = self.processor.apply_chat_template(
                    messages_vlm,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device, dtype=torch.bfloat16)
                self.token_cache[cache_key] = {"inputs": inputs_vlm}
            except Exception as e:
                logger.error(f"Error in tokenization: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]
        adjusted_max_tokens = min(max_tokens, max(20, input_len * 2))

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=adjusted_max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True)
        self.token_cache[cache_key]["response"] = response  # Cache the full response
        logger.info(f"Generated response: {response}")
        return response

    async def vision_query(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Summarize your answer in one sentence maximum."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}] + ([{"type": "image", "image": image}] if image else [])
            }
        ]

        cache_key = f"vision_{query}_{'image' if image else 'no_image'}"
        if cache_key in self.token_cache and "response" in self.token_cache[cache_key]:
            logger.info("Using cached response")
            return self.token_cache[cache_key]["response"]

        if cache_key in self.token_cache and "inputs" in self.token_cache[cache_key]:
            inputs_vlm = self.token_cache[cache_key]["inputs"]
            logger.info("Using cached tokenized input")
        else:
            try:
                inputs_vlm = self.processor.apply_chat_template(
                    messages_vlm,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device, dtype=torch.bfloat16)
                self.token_cache[cache_key] = {"inputs": inputs_vlm}
            except Exception as e:
                logger.error(f"Error in apply_chat_template: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]
        adjusted_max_tokens = min(50, max(20, input_len * 2))

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=adjusted_max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True)
        self.token_cache[cache_key]["response"] = response  # Cache the full response
        logger.info(f"Vision query response: {response}")
        return response

    async def chat_v2(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}] + ([{"type": "image", "image": image}] if image else [])
            }
        ]

        cache_key = f"chat_v2_{query}_{'image' if image else 'no_image'}"
        if cache_key in self.token_cache and "response" in self.token_cache[cache_key]:
            logger.info("Using cached response")
            return self.token_cache[cache_key]["response"]

        if cache_key in self.token_cache and "inputs" in self.token_cache[cache_key]:
            inputs_vlm = self.token_cache[cache_key]["inputs"]
            logger.info("Using cached tokenized input")
        else:
            try:
                inputs_vlm = self.processor.apply_chat_template(
                    messages_vlm,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device, dtype=torch.bfloat16)
                self.token_cache[cache_key] = {"inputs": inputs_vlm}
            except Exception as e:
                logger.error(f"Error in apply_chat_template: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]
        adjusted_max_tokens = min(50, max(20, input_len * 2))

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=adjusted_max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True)
        self.token_cache[cache_key]["response"] = response  # Cache the full response
        logger.info(f"Chat_v2 response: {response}")
        return response