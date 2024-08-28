from transformers import pipeline
from typing import Union, Dict, Any
import subprocess


allowed_kwargs = [
    # "model",
    "tokenizer",
    "modelcard",
    "framework",
    "task",
    "num_workers",
    "batch_size",
    "args_parser",
    "device",
    "torch_dtype",
    "min_length_for_response",
    "minimum_tokens",
    "mask_token",
    "max_length",
    "max_new_tokens",
]


class TransformersService(BaseService):
    def __init__(
        self,
        task: str = None,
        model: Union[str, Any] = None,
        config: Union[str, Dict, Any] = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.task = task
        self.model = model
        self.config = config
        self.allowed_kwargs = allowed_kwargs
        self.pipe = pipeline(
            task=task, model=model, config=config, device=device, **kwargs
        )

    async def serve_chat(self, messages, **kwargs):
        if self.task:
            if self.task != "conversational":
                raise ValueError(f"Invalid transformers pipeline task: {self.task}.")

        payload = {"messages": messages}
        config = {}
        for k, v in kwargs.items():
            if k == "max_tokens":
                config["max_new_tokens"] = v
            if k in allowed_kwargs:
                config[k] = v

        msg = "".join([i["content"] for i in messages if i["role"] == "user"])
        conversation = ""
        response = self.pipe(msg, **config)
        try:
            conversation = response[0]["generated_text"]
        except:
            conversation = response

        completion = {"choices": [{"message": {"content": conversation}}]}

        return payload, completion
