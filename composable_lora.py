from typing import List, Dict
import re
import torch
import composable_lora_step
import composable_lycoris
import plot_helper
from modules import extra_networks, devices

def lora_forward(compvis_module, input, res):
    global text_model_encoder_counter
    global diffusion_model_counter
    global step_counter
    global should_print
    global first_log_drawing
    global drawing_lora_first_index
    import lora

    if composable_lycoris.has_webui_lycoris:
        import lycoris
        if len(lycoris.loaded_lycos) > 0 and not first_log_drawing:
            print("Found LyCORIS models, Using Composable LyCORIS.")

    if not first_log_drawing:
        first_log_drawing = True
        if enabled:
            print("Composable LoRA load successful.")
        if opt_plot_lora_weight:
            log_lora()
            drawing_lora_first_index = drawing_data[0]

    if len(lora.loaded_loras) == 0:
        return res

    lora_layer_name_loading : str | None = getattr(compvis_module, 'lora_layer_name', None)
    if lora_layer_name_loading is None:
        return res
    #let it type is actually a string
    lora_layer_name : str = str(lora_layer_name_loading)
    del lora_layer_name_loading

    num_loras = len(lora.loaded_loras)
    if composable_lycoris.has_webui_lycoris:
        num_loras += len(lycoris.loaded_lycos)

    if text_model_encoder_counter == -1:
        text_model_encoder_counter = len(prompt_loras) * num_loras

    tmp_check_loras = [] #store which lora are already apply
    tmp_check_loras.clear()

    for m_lora in lora.loaded_loras:
        module = m_lora.modules.get(lora_layer_name, None)
        if module is None:
            #fix the lyCORIS issue
            composable_lycoris.check_lycoris_end_layer(lora_layer_name, res, num_loras)
            continue

        current_lora = m_lora.name
        lora_already_used = False
        if current_lora in tmp_check_loras:
            lora_already_used = True
        #store the applied lora into list
        tmp_check_loras.append(current_lora)
        if lora_already_used:
            composable_lycoris.check_lycoris_end_layer(lora_layer_name, res, num_loras)
            continue
        
        #support for lyCORIS
        patch = composable_lycoris.get_lora_patch(module, input, res, lora_layer_name)
        alpha = composable_lycoris.get_lora_alpha(module, 1.0)
        num_prompts = len(prompt_loras)

        # print(f"lora.name={m_lora.name} lora.mul={m_lora.multiplier} alpha={alpha} pat.shape={patch.shape}")
        res = apply_composable_lora(lora_layer_name, m_lora, "lora", patch, alpha, res, num_loras, num_prompts)
    return res

re_AND = re.compile(r"\bAND\b")

def load_prompt_loras(prompt: str):
    global is_single_block
    global full_controllers
    global first_log_drawing
    prompt_loras.clear()
    prompt_blocks.clear()
    lora_controllers.clear()
    drawing_data.clear()
    full_controllers.clear()
    drawing_lora_names.clear()
    cache_layer_list.clear()
    #load AND...AND block
    subprompts = re_AND.split(prompt)
    tmp_prompt_loras = []
    tmp_prompt_blocks = []
    for i, subprompt in enumerate(subprompts):
        loras = {}
        _, extra_network_data = extra_networks.parse_prompt(subprompt)
        for m_type in ['lora', 'lyco']:
            if m_type in extra_network_data.keys():
                for params in extra_network_data[m_type]:
                    name = params.items[0]
                    multiplier = float(params.items[1]) if len(params.items) > 1 else 1.0
                    loras[f"{m_type}:{name}"] = multiplier

        tmp_prompt_loras.append(loras)
        tmp_prompt_blocks.append(subprompt)
    is_single_block = (len(tmp_prompt_loras) == 1)

    #load [A:B:N] syntax
    if opt_composable_with_step:
        print("Loading LoRA step controller...")
    tmp_lora_controllers = composable_lora_step.parse_step_rendering_syntax(prompt)

    #for batches > 1
    prompt_loras.extend(tmp_prompt_loras * num_batches)
    lora_controllers.extend(tmp_lora_controllers * num_batches)
    prompt_blocks.extend(tmp_prompt_blocks * num_batches)

    for controller_it in tmp_lora_controllers:
        full_controllers += controller_it
    first_log_drawing = False

def reset_counters():
    global text_model_encoder_counter
    global diffusion_model_counter
    global step_counter
    global should_print

    # reset counter to uc head
    text_model_encoder_counter = -1
    diffusion_model_counter = 0
    step_counter += 1
    should_print = True
    
def reset_step_counters():
    global step_counter
    global should_print

    should_print = True
    step_counter = 0

def add_step_counters(): 
    global step_counter
    global should_print

    should_print = True
    step_counter += 1

    if step_counter > num_steps:
        step_counter = 0
    else:
        if opt_plot_lora_weight:
            log_lora()

def log_lora():
    import lora
    loaded_loras = lora.loaded_loras
    loaded_lycos = []
    if composable_lycoris.has_webui_lycoris:
        import lycoris
        loaded_lycos = lycoris.loaded_lycos

    tmp_data : List[float] = []
    if len(loaded_loras) + len(loaded_lycos) <= 0:
        tmp_data = [0.0]
        if len(drawing_lora_names) <= 0:
            drawing_lora_names.append("LoRA Model Not Found.")
    for m_type in [("lora", loaded_loras), ("lyco", loaded_lycos)]:
        for m_lora in m_type[1]:
            current_lora = f"{m_type[0]}:{m_lora.name}"
            multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, "lora_layer_name")
            if opt_composable_with_step:
                multiplier = composable_lora_step.check_lora_weight(full_controllers, current_lora, step_counter, num_steps)
            index = -1
            if current_lora in drawing_lora_names:
                index = drawing_lora_names.index(current_lora)
            else:
                index = len(drawing_lora_names)
                drawing_lora_names.append(current_lora)
            if index >= len(tmp_data):
                for i in range(len(tmp_data), index):
                    tmp_data.append(0.0)
                tmp_data.append(multiplier)
            else:
                tmp_data[index] = multiplier
    drawing_data.append(tmp_data)

def plot_lora():
    max_size = -1
    if len(drawing_data) < num_steps:
        item = drawing_data[len(drawing_data) - 1] if len(drawing_data) > 0 else [0.0]
        drawing_data.extend([item]*(num_steps - len(drawing_data)))
    drawing_data.insert(0, drawing_lora_first_index)
    for datalist in drawing_data:
        datalist_len = len(datalist)
        if datalist_len > max_size:
            max_size = datalist_len
    for i, datalist in enumerate(drawing_data):
        datalist_len = len(datalist)
        if datalist_len < max_size:
            drawing_data[i].extend([0.0]*(max_size - datalist_len))
    return plot_helper.plot_lora_weight(drawing_data, drawing_lora_names)

def clear_cache_lora(compvis_module):
    lora_layer_name = getattr(compvis_module, 'lora_layer_name', 'unknown layer')
    if lora_layer_name in cache_layer_list:
        return
    cache_layer_list.append(lora_layer_name)
    lyco_weights_backup = getattr(compvis_module, "lyco_weights_backup", None)
    lora_weights_backup = getattr(compvis_module, "lora_weights_backup", None)
    if enabled:
        if lyco_weights_backup is not None:
            if isinstance(compvis_module, torch.nn.MultiheadAttention):
                compvis_module.in_proj_weight.copy_(lyco_weights_backup[0])
                compvis_module.out_proj.weight.copy_(lyco_weights_backup[1])
                lora_weights_backup = (
                    lyco_weights_backup[0].to(devices.cpu, copy=True), 
                    lyco_weights_backup[1].to(devices.cpu, copy=True)
                )
            else:
                compvis_module.weight.copy_(lyco_weights_backup)
                lora_weights_backup = lyco_weights_backup.to(devices.cpu, copy=True)
            setattr(compvis_module, "lora_weights_backup", lora_weights_backup)
        elif lora_weights_backup is not None:
            if isinstance(compvis_module, torch.nn.MultiheadAttention):
                compvis_module.in_proj_weight.copy_(lora_weights_backup[0])
                compvis_module.out_proj.weight.copy_(lora_weights_backup[1])
            else:
                compvis_module.weight.copy_(lora_weights_backup)
        setattr(compvis_module, "lora_current_names", ())
        setattr(compvis_module, "lyco_current_names", ())  

def apply_composable_lora(lora_layer_name, m_lora, m_type: str, patch, alpha, res, num_loras, num_prompts):
    global text_model_encoder_counter
    global diffusion_model_counter
    global step_counter
    m_lora_name = f"{m_type}:{m_lora.name}"
    # print(f"lora.name={m_lora.name} lora.mul={m_lora.multiplier} alpha={alpha} pat.shape={patch.shape}")
    if enabled:
        if lora_layer_name.startswith("transformer_"):  # "transformer_text_model_encoder_"
            #
            if 0 <= text_model_encoder_counter // num_loras < len(prompt_loras):
                # c
                loras = prompt_loras[text_model_encoder_counter // num_loras]
                multiplier = loras.get(m_lora_name, 0.0)
                if multiplier != 0.0:
                    multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
                    # print(f"c #{text_model_encoder_counter // num_loras} lora.name={m_lora_name} mul={multiplier}  lora_layer_name={lora_layer_name}")
                    res += multiplier * alpha * patch
            else:
                # uc
                multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
                if (opt_uc_text_model_encoder or (is_single_block and (not opt_single_no_uc))) and multiplier != 0.0:
                    # print(f"uc #{text_model_encoder_counter // num_loras} lora.name={m_lora_name} lora.mul={multiplier}  lora_layer_name={lora_layer_name}")
                    res += multiplier * alpha * patch

            if lora_layer_name.endswith("_11_mlp_fc2"):  # last lora_layer_name of text_model_encoder
                text_model_encoder_counter += 1
                # c1 c1 c2 c2 .. .. uc uc
                if text_model_encoder_counter == (len(prompt_loras) + num_batches) * num_loras:
                    text_model_encoder_counter = 0

        elif lora_layer_name.startswith("diffusion_model_"):  # "diffusion_model_"

            if res.shape[0] == num_batches * num_prompts + num_batches:
                # tensor.shape[1] == uncond.shape[1]
                tensor_off = 0
                uncond_off = num_batches * num_prompts
                for b in range(num_batches):
                    # c
                    for p, loras in enumerate(prompt_loras):
                        multiplier = loras.get(m_lora_name, 0.0)
                        if opt_composable_with_step:
                            prompt_block_id = p
                            lora_controller = lora_controllers[prompt_block_id]
                            multiplier = composable_lora_step.check_lora_weight(lora_controller, m_lora_name, step_counter, num_steps)
                        if multiplier != 0.0:
                            multiplier *= composable_lycoris.lycoris_get_multiplier_normalized(m_lora, lora_layer_name)
                            # print(f"tensor #{b}.{p} lora.name={m_lora_name} mul={multiplier} lora_layer_name={lora_layer_name}")
                            res[tensor_off] += multiplier * alpha * patch[tensor_off]
                        tensor_off += 1

                    # uc
                    multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
                    if (opt_uc_diffusion_model or (is_single_block and (not opt_single_no_uc))) and multiplier != 0.0:
                        # print(f"uncond lora.name={m_lora_name} lora.mul={m_lora.multiplier} lora_layer_name={lora_layer_name}")
                        if is_single_block and opt_composable_with_step:
                            multiplier = composable_lora_step.check_lora_weight(full_controllers, m_lora_name, step_counter, num_steps)
                            multiplier *= composable_lycoris.lycoris_get_multiplier_normalized(m_lora, lora_layer_name)
                        res[uncond_off] += multiplier * alpha * patch[uncond_off]
                    
                    uncond_off += 1
            else:
                # tensor.shape[1] != uncond.shape[1]
                cur_num_prompts = res.shape[0]
                base = (diffusion_model_counter // cur_num_prompts) // num_loras * cur_num_prompts
                prompt_len = len(prompt_loras)
                if 0 <= base < len(prompt_loras):
                    # c
                    for off in range(cur_num_prompts):
                        if base + off < prompt_len:
                            loras = prompt_loras[base + off]
                            multiplier = loras.get(m_lora_name, 0.0)
                            if opt_composable_with_step:
                                prompt_block_id = base + off
                                lora_controller = lora_controllers[prompt_block_id]
                                multiplier = composable_lora_step.check_lora_weight(lora_controller, m_lora_name, step_counter, num_steps)
                            if multiplier != 0.0:
                                multiplier *= composable_lycoris.lycoris_get_multiplier_normalized(m_lora, lora_layer_name)
                                # print(f"c #{base + off} lora.name={m_lora_name} mul={multiplier} lora_layer_name={lora_layer_name}")
                                res[off] += multiplier * alpha * patch[off]
                else:
                    # uc
                    multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
                    if (opt_uc_diffusion_model or (is_single_block and (not opt_single_no_uc))) and multiplier != 0.0:
                        # print(f"uc {lora_layer_name} lora.name={m_lora_name} lora.mul={m_lora.multiplier}")
                        if is_single_block and opt_composable_with_step:
                            multiplier = composable_lora_step.check_lora_weight(full_controllers, m_lora_name, step_counter, num_steps)
                            multiplier *= composable_lycoris.lycoris_get_multiplier_normalized(m_lora, lora_layer_name)
                        res += multiplier * alpha * patch

                if lora_layer_name.endswith("_11_1_proj_out"):  # last lora_layer_name of diffusion_model
                    diffusion_model_counter += cur_num_prompts
                    # c1 c2 .. uc
                    if diffusion_model_counter >= (len(prompt_loras) + num_batches) * num_loras:
                        diffusion_model_counter = 0
                        add_step_counters()
        else:
            # default
            multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
            if multiplier != 0.0:
                # print(f"default {lora_layer_name} lora.name={m_lora_name} lora.mul={m_lora.multiplier}")
                res += multiplier * alpha * patch
    else:
        # default
        multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
        if multiplier != 0.0:
            # print(f"DEFAULT {lora_layer_name} lora.name={m_lora_name} lora.mul={m_lora.multiplier}")
            res += multiplier * alpha * patch
    return res

def lora_Linear_forward(self, input):
    clear_cache_lora(self)
    if (not self.weight.is_cuda) and input.is_cuda: #if variables not on the same device (between cpu and gpu)
        self_weight_cuda = self.weight.cuda() #pass to GPU
        to_del = self.weight
        self.weight = None                    #delete CPU variable
        del to_del
        del self.weight                       #avoid pytorch 2.0 throwing exception
        self.weight = self_weight_cuda        #load GPU data to self.weight
    res = torch.nn.Linear_forward_before_lora(self, input)
    res = lora_forward(self, input, res)
    if composable_lycoris.has_webui_lycoris:
        res = composable_lycoris.lycoris_forward(self, input, res)
    return res

def lora_Conv2d_forward(self, input):
    clear_cache_lora(self)
    if (not self.weight.is_cuda) and input.is_cuda:
        self_weight_cuda = self.weight.cuda()
        to_del = self.weight
        self.weight = None
        del to_del
        del self.weight #avoid "cannot assign XXX as parameter YYY (torch.nn.Parameter or None expected)"
        self.weight = self_weight_cuda
    res = torch.nn.Conv2d_forward_before_lora(self, input)
    res = lora_forward(self, input, res)
    if composable_lycoris.has_webui_lycoris:
        res = composable_lycoris.lycoris_forward(self, input, res)
    return res

def should_reload():
    #pytorch 2.0 should reload
    match = re.search(r"\d+(\.\d+)?",str(torch.__version__)) 
    if not match:
        return True
    ver = float(match.group(0))
    return ver >= 2.0

enabled = False
opt_composable_with_step = False
opt_uc_text_model_encoder = False
opt_uc_diffusion_model = False
opt_plot_lora_weight = False
opt_single_no_uc = False
verbose = True

drawing_lora_names : List[str] = []
drawing_data : List[List[float]] = []
drawing_lora_first_index : List[float] = []
first_log_drawing : bool = False

is_single_block : bool = False
num_batches: int = 0
num_steps: int = 20
prompt_loras: List[Dict[str, float]] = []
text_model_encoder_counter: int = -1
diffusion_model_counter: int = 0
step_counter: int = 0
cache_layer_list : List[str] = []

should_print : bool = True
prompt_blocks: List[str] = []
lora_controllers: List[List[composable_lora_step.LoRA_Controller_Base]] = []
full_controllers: List[composable_lora_step.LoRA_Controller_Base] = []
