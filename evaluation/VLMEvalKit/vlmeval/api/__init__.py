def _unavailable_model(name, module):
    class UnavailableModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} is unavailable because this VLMEvalKit snapshot is missing {module}"
            )

    UnavailableModel.__name__ = name
    return UnavailableModel


from .gpt import OpenAIWrapper, GPT4V
from .hf_chat_model import HFChatModel
try:
    from .gemini import GeminiWrapper, Gemini
except ModuleNotFoundError:
    GeminiWrapper = _unavailable_model("GeminiWrapper", "vlmeval.api.gemini")
    Gemini = _unavailable_model("Gemini", "vlmeval.api.gemini")
try:
    from .qwen_vl_api import QwenVLWrapper, QwenVLAPI, Qwen2VLAPI
except ModuleNotFoundError:
    QwenVLWrapper = _unavailable_model("QwenVLWrapper", "vlmeval.api.qwen_vl_api")
    QwenVLAPI = _unavailable_model("QwenVLAPI", "vlmeval.api.qwen_vl_api")
    Qwen2VLAPI = _unavailable_model("Qwen2VLAPI", "vlmeval.api.qwen_vl_api")
try:
    from .qwen_api import QwenAPI
except ModuleNotFoundError:
    QwenAPI = _unavailable_model("QwenAPI", "vlmeval.api.qwen_api")
from .claude import Claude_Wrapper, Claude3V
try:
    from .reka import Reka
except ModuleNotFoundError:
    Reka = _unavailable_model("Reka", "vlmeval.api.reka")
from .glm_vision import GLMVisionAPI
from .cloudwalk import CWWrapper
try:
    from .sensechat_vision import SenseChatVisionAPI
except ModuleNotFoundError:
    SenseChatVisionAPI = _unavailable_model(
        "SenseChatVisionAPI", "vlmeval.api.sensechat_vision"
    )
from .siliconflow import SiliconFlowAPI, TeleMMAPI
from .hunyuan import HunyuanVision
try:
    from .bailingmm import bailingMMAPI
except ModuleNotFoundError:
    bailingMMAPI = _unavailable_model("bailingMMAPI", "vlmeval.api.bailingmm")
try:
    from .bluelm_api import BlueLMWrapper, BlueLM_API
except ModuleNotFoundError:
    BlueLMWrapper = _unavailable_model("BlueLMWrapper", "vlmeval.api.bluelm_api")
    BlueLM_API = _unavailable_model("BlueLM_API", "vlmeval.api.bluelm_api")
from .jt_vl_chat import JTVLChatAPI
from .taiyi import TaiyiAPI
from .lmdeploy import LMDeployAPI
from .taichu import TaichuVLAPI, TaichuVLRAPI
from .doubao_vl_api import DoubaoVL
from .mug_u import MUGUAPI
from .kimivl_api import KimiVLAPIWrapper, KimiVLAPI
try:
    from .vllm_api import VLLMAPIWrapper
except ModuleNotFoundError:
    VLLMAPIWrapper = _unavailable_model("VLLMAPIWrapper", "optional Jupyter agent dependencies")
from .visual_agent_api import VisualAgentAPI


__all__ = [
    'OpenAIWrapper', 'HFChatModel', 'GeminiWrapper', 'GPT4V', 'Gemini',
    'QwenVLWrapper', 'QwenVLAPI', 'QwenAPI', 'Claude3V', 'Claude_Wrapper',
    'Reka', 'GLMVisionAPI', 'CWWrapper', 'SenseChatVisionAPI', 'HunyuanVision',
    'Qwen2VLAPI', 'BlueLMWrapper', 'BlueLM_API', 'JTVLChatAPI',
    'bailingMMAPI', 'TaiyiAPI', 'TeleMMAPI', 'SiliconFlowAPI', 'LMDeployAPI',
    'TaichuVLAPI', 'TaichuVLRAPI', 'DoubaoVL', "MUGUAPI", 'KimiVLAPIWrapper', 'KimiVLAPI', 'VLLMAPIWrapper',
    'VisualAgentAPI',
]
