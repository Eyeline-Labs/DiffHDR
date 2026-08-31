import torch
from .wan_video_dit import WanModel


class LoRAFromCivitai:
    def __init__(self):
        self.supported_model_classes = []
        self.lora_prefix = []
        self.renamed_lora_prefix = {}
        self.special_keys = {}

    def convert_state_dict(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        for key in state_dict:
            if ".lora_up" in key:
                return self.convert_state_dict_up_down(state_dict, lora_prefix, alpha)
        return self.convert_state_dict_AB(state_dict, lora_prefix, alpha)

    def convert_state_dict_up_down(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        new_state_dict = {}
        for key in state_dict:
            if ".lora_up" not in key:
                continue
            original_key = key.replace(".lora_up", "")
            up_key = key
            down_key = key.replace(".lora_up", ".lora_down")
            if original_key.startswith(lora_prefix):
                original_key = original_key[len(lora_prefix):]
            original_key = original_key.replace(".", "_", original_key.count(".") - 1).replace("_", ".", 1)
            for special_key in self.special_keys:
                original_key = original_key.replace(special_key, self.special_keys[special_key])
            weight_up = state_dict[up_key].float()
            weight_down = state_dict[down_key].float()
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                lora_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                lora_weight = alpha * torch.mm(weight_up, weight_down)
            new_state_dict[original_key] = lora_weight
        return new_state_dict

    def convert_state_dict_AB(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        new_state_dict = {}
        for key in state_dict:
            if ".lora_B." not in key:
                continue
            original_key = key.replace(".lora_B.", ".")
            up_key = key
            down_key = key.replace(".lora_B.", ".lora_A.")
            alpha_key = key.replace(".lora_B.", ".alpha")
            if lora_prefix:
                original_key = original_key.replace(lora_prefix, "")
            original_key = original_key.replace(".", "_", original_key.count(".") - 1).replace("_", ".", 1)
            for special_key in self.special_keys:
                original_key = original_key.replace(special_key, self.special_keys[special_key])
            weight_up = state_dict[up_key].float()
            weight_down = state_dict[down_key].float()
            if alpha_key in state_dict:
                lora_alpha = state_dict[alpha_key].float()
                scale = alpha * lora_alpha / weight_down.shape[0]
            else:
                scale = alpha
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                lora_weight = scale * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                lora_weight = scale * torch.mm(weight_up, weight_down)
            new_state_dict[original_key] = lora_weight
        return new_state_dict

    def match(self, model, state_dict_lora):
        if not isinstance(model, tuple(self.supported_model_classes)):
            return None
        for prefix, renamed_prefix in zip(self.lora_prefix, self.renamed_lora_prefix.get("", self.lora_prefix)):
            if any(key.startswith(prefix) for key in state_dict_lora):
                return prefix, renamed_prefix
        return None

    def load(self, model, state_dict_lora, lora_prefix="", alpha=1.0, model_resource=""):
        state_dict = self.convert_state_dict(state_dict_lora, lora_prefix, alpha)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"    {len(state_dict)} tensors are updated.")
        if unexpected_keys:
            print(f"    Unexpected keys: {len(unexpected_keys)}")


class GeneralLoRAFromPeft:
    def __init__(self):
        self.supported_model_classes = [WanModel]

    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict

    def match(self, model: torch.nn.Module, state_dict_lora):
        lora_name_dict = self.get_name_dict(state_dict_lora)
        model_name_dict = {name: None for name, _ in model.named_parameters()}
        matched_num = sum([i in model_name_dict for i in lora_name_dict])
        if matched_num == len(lora_name_dict):
            return "", ""
        else:
            return None

    def fetch_device_and_dtype(self, state_dict):
        device, dtype = None, None
        for name, param in state_dict.items():
            device, dtype = param.device, param.dtype
            break
        computation_device = device
        computation_dtype = dtype
        if computation_device == torch.device("cpu"):
            if torch.cuda.is_available():
                computation_device = torch.device("cuda")
        if computation_dtype == torch.float8_e4m3fn:
            computation_dtype = torch.float32
        return device, dtype, computation_device, computation_dtype

    def load(self, model, state_dict_lora, lora_prefix="", alpha=1.0, model_resource=""):
        state_dict_model = model.state_dict()
        device, dtype, computation_device, computation_dtype = self.fetch_device_and_dtype(state_dict_model)
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name in lora_name_dict:
            weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=computation_device, dtype=computation_dtype)
            weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=computation_device, dtype=computation_dtype)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_lora = alpha * torch.mm(weight_up, weight_down)
            weight_model = state_dict_model[name].to(device=computation_device, dtype=computation_dtype)
            weight_patched = weight_model + weight_lora
            state_dict_model[name] = weight_patched.to(device=device, dtype=dtype)
        print(f"    {len(lora_name_dict)} tensors are updated.")
        model.load_state_dict(state_dict_model)


class WanLoRAConverter:
    @staticmethod
    def align_to_opensource_format(state_dict, **kwargs):
        state_dict = {"diffusion_model." + name.replace(".default.", "."): param for name, param in state_dict.items()}
        return state_dict

    @staticmethod
    def align_to_diffsynth_format(state_dict, **kwargs):
        state_dict = {name.replace("diffusion_model.", "").replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): param for name, param in state_dict.items()}
        return state_dict


def get_lora_loaders():
    return [GeneralLoRAFromPeft()]
