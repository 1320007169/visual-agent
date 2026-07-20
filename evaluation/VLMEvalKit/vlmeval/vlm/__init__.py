import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)


def _unavailable_model(name, module):
    class UnavailableModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} is unavailable because this VLMEvalKit snapshot is missing {module}"
            )

    UnavailableModel.__name__ = name
    return UnavailableModel


from .aria import Aria
from .base import BaseModel
from .hawk_vl import HawkVL
from .thyme import Thyme
from .cogvlm import CogVlm, GLM4v, GLMThinking
from .emu import Emu, Emu3_chat, Emu3_gen
from .eagle_x import Eagle
from .granite_vision import GraniteVision3
from .idefics import IDEFICS, IDEFICS2
from .instructblip import InstructBLIP
from .kosmos import Kosmos2
from .llava import (
    LLaVA,
    LLaVA_Next,
    LLaVA_XTuner,
    LLaVA_Next2,
    LLaVA_OneVision,
    LLaVA_OneVision_HF,
)
from .vita import VITA, VITAQwen2
from .long_vita import LongVITA
from .minicpm_v import MiniCPM_V, MiniCPM_Llama3_V, MiniCPM_V_2_6, MiniCPM_o_2_6, MiniCPM_V_4, MiniCPM_V_4_5
from .minigpt4 import MiniGPT4
from .mmalaya import MMAlaya, MMAlaya2
from .monkey import Monkey, MonkeyChat
from .moondream import Moondream1, Moondream2
from .minimonkey import MiniMonkey
from .mplug_owl2 import mPLUG_Owl2
from .omnilmm import OmniLMM12B
from .open_flamingo import OpenFlamingo
from .pandagpt import PandaGPT
try:
    from .qwen_vl import QwenVL, QwenVLChat
except ModuleNotFoundError:
    QwenVL = _unavailable_model("QwenVL", "vlmeval.vlm.qwen_vl")
    QwenVLChat = _unavailable_model("QwenVLChat", "vlmeval.vlm.qwen_vl")
try:
    from .qwen2_vl import Qwen2VLChat, Qwen2VLChatAguvis
except ModuleNotFoundError:
    Qwen2VLChat = _unavailable_model("Qwen2VLChat", "vlmeval.vlm.qwen2_vl")
    Qwen2VLChatAguvis = _unavailable_model(
        "Qwen2VLChatAguvis", "vlmeval.vlm.qwen2_vl"
    )
from .transcore_m import TransCoreM
from .visualglm import VisualGLM
from .xcomposer import (
    ShareCaptioner,
    XComposer,
    XComposer2,
    XComposer2_4KHD,
    XComposer2d5,
)
from .yi_vl import Yi_VL
try:
    from .internvl import InternVLChat
except ModuleNotFoundError:
    InternVLChat = _unavailable_model("InternVLChat", "vlmeval.vlm.internvl")
from .deepseek_vl import DeepSeekVL
from .deepseek_vl2 import DeepSeekVL2
from .janus import Janus
from .mgm import Mini_Gemini
from .bunnyllama3 import BunnyLLama3
from .vxverse import VXVERSE
from .gemma import PaliGemma, Gemma3
from .qh_360vl import QH_360VL
from .phi3_vision import Phi3Vision, Phi3_5Vision
from .phi4_multimodal import Phi4Multimodal
from .wemm import WeMM
from .cambrian import Cambrian
from .chameleon import Chameleon
from .video_llm import (
    VideoLLaVA,
    VideoLLaVA_HF,
    Chatunivi,
    VideoChatGPT,
    LLaMAVID,
    VideoChat2_HD,
    PLLaVA,
)
from .vila import VILA, NVILA
from .ovis import Ovis, Ovis1_6, Ovis1_6_Plus, Ovis2, OvisU1, Ovis2_5
from .mantis import Mantis
from .mixsense import LLama3Mixsense
from .parrot import Parrot
from .omchat import OmChat
from .rbdash import RBDash
from .xgen_mm import XGenMM
from .slime import SliME
from .mplug_owl3 import mPLUG_Owl3
from .pixtral import Pixtral
from .llama_vision import llama_vision
from .llama4 import llama4
from .molmo import molmo
from .points import POINTS, POINTSV15
from .nvlm import NVLM
from .vintern_chat import VinternChat
from .h2ovl_mississippi import H2OVLChat
from .falcon_vlm import Falcon2VLM
from .smolvlm import SmolVLM, SmolVLM2
from .sail_vl import SailVL
from .valley import Valley2Chat
from .ross import Ross
from .ola import Ola
from .x_vl import X_VL_HF
from .ursa import UrsaChat
try:
    from .vlm_r1 import VLMR1Chat
except ModuleNotFoundError:
    VLMR1Chat = _unavailable_model("VLMR1Chat", "vlmeval.vlm.qwen2_vl")
from .aki import AKI
from .ristretto import Ristretto
try:
    from .vlaa_thinker import VLAAThinkerChat
except ModuleNotFoundError:
    VLAAThinkerChat = _unavailable_model("VLAAThinkerChat", "vlmeval.vlm.qwen2_vl")
from .kimi_vl import KimiVL
try:
    from .wethink_vl import WeThinkVL
except ModuleNotFoundError:
    WeThinkVL = _unavailable_model("WeThinkVL", "vlmeval.vlm.qwen2_vl")
from .flash_vl import FlashVL
from .oryx import Oryx
try:
    from .treevgr import TreeVGR
except ModuleNotFoundError:
    TreeVGR = _unavailable_model("TreeVGR", "vlmeval.vlm.qwen2_vl")
from .varco_vision import VarcoVision
from .qtunevl import (
    QTuneVL,
    QTuneVLChat,
)
from .logics import Logics_Thinking
from .cosmos import Cosmos
