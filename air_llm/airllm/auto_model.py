import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx

class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )
    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            print(f"using hf_token")
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        arch = config.architectures[0]
        model_type = getattr(config, 'model_type', '')
        
        if "Qwen2ForCausalLM" in arch:
            return "airllm", "AirLLMQWen2"
        elif "QWen" in arch:
            return "airllm", "AirLLMQWen"
        elif "Baichuan" in arch:
            return "airllm", "AirLLMBaichuan"
        # GLM-4 MoE models (like GLM-4.7) - check before ChatGLM
        elif "GLM4" in arch or model_type == "glm4_moe" or model_type == "glm4":
            # Use MoE class for MoE models, base class for dense
            if "moe" in model_type.lower() or "MoE" in arch:
                return "airllm", "AirLLMGLM4MoE"
            return "airllm", "AirLLMGLM4"
        elif "ChatGLM" in arch:
            return "airllm", "AirLLMChatGLM"
        elif "InternLM" in arch:
            return "airllm", "AirLLMInternLM"
        elif "Mistral" in arch:
            return "airllm", "AirLLMMistral"
        elif "Mixtral" in arch:
            return "airllm", "AirLLMMixtral"
        elif "Llama" in arch:
            return "airllm", "AirLLMLlama2"
        else:
            print(f"unknown architecture: {arch}, try to use Llama2...")
            return "airllm", "AirLLMLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, ** kwargs)

        module, cls = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, cls)
        return class_(pretrained_model_name_or_path, *inputs, ** kwargs)