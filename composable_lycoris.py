import torch
from modules import shared

def get_lora_patch(module, input, res):
    if is_loha(module):
        if input.is_cuda: #if is cuda, pass to cuda; otherwise do nothing
            pass_loha_to_gpu(module)
    if getattr(shared.opts, "lora_apply_to_outputs", False) and res.shape == input.shape:
        if hasattr(module, 'inference'): #support for lyCORIS
            return module.inference(res)
        elif hasattr(module, 'up'):     #LoRA
            return module.up(module.down(res))
        else:
            raise NotImplementedError(
                "Your settings, extensions or models are not compatible with each other."
            )
    else:
        if hasattr(module, 'inference'): #support for lyCORIS
            return module.inference(input)
        elif hasattr(module, 'up'):     #LoRA
            return module.up(module.down(input))
        else:
            raise NotImplementedError(
                "Your settings, extensions or models are not compatible with each other."
            )
        
def get_lora_alpha(module, default_val):
    if hasattr(module, 'up'):
        return (module.alpha / module.up.weight.shape[1] if module.alpha else default_val)
    elif hasattr(module, 'dim'): #support for lyCORIS
        return (module.alpha / module.dim if module.alpha else default_val)
    else:
        return default_val
    
def check_lycoris_end_layer(lora_layer_name: str, res, num_loras):
    if lora_layer_name.endswith("_11_mlp_fc2") or lora_layer_name.endswith("_11_1_proj_out"):
        import composable_lora as lora_controller
        if lora_layer_name.endswith("_11_mlp_fc2"):  # lyCORIS maybe doesn't has _11_mlp_fc2 layer
            lora_controller.text_model_encoder_counter += 1
            if lora_controller.text_model_encoder_counter == (len(lora_controller.prompt_loras) + lora_controller.num_batches) * num_loras:
                lora_controller.text_model_encoder_counter = 0
        if lora_layer_name.endswith("_11_1_proj_out"):  # lyCORIS maybe doesn't has _11_1_proj_out layer
            lora_controller.diffusion_model_counter += res.shape[0]
            if lora_controller.diffusion_model_counter >= (len(lora_controller.prompt_loras) + lora_controller.num_batches) * num_loras:
                lora_controller.diffusion_model_counter = 0
                lora_controller.add_step_counters()

def is_loha(m_lora):
    return hasattr(m_lora, 'w1a') or hasattr(m_lora, 'w1b') or hasattr(m_lora, 'w2a') or hasattr(m_lora, 'w2b')

def pass_loha_to_gpu(m_loha):
    if hasattr(m_loha, 'bias'):
        if isinstance(m_loha.bias, torch.Tensor):
            if not m_loha.bias.is_cuda:
                to_cuda = m_loha.bias.cuda()
                to_del = m_loha.bias
                m_loha.bias = None
                del to_del
                del m_loha.bias
                m_loha.bias = to_cuda
    if hasattr(m_loha, 't1'):
        if isinstance(m_loha.t1, torch.Tensor):
            if not m_loha.t1.is_cuda:
                to_cuda = m_loha.t1.cuda()
                to_del = m_loha.t1
                m_loha.t1 = None
                del to_del
                del m_loha.t1
                m_loha.t1 = to_cuda
    if hasattr(m_loha, 't2'):
        if isinstance(m_loha.t2, torch.Tensor):
            if not m_loha.t2.is_cuda:
                to_cuda = m_loha.t2.cuda()
                to_del = m_loha.t2
                m_loha.t2 = None
                del to_del
                del m_loha.t2
                m_loha.t2 = to_cuda
    if hasattr(m_loha, 'w'):
        if isinstance(m_loha.w, torch.Tensor):
            if not m_loha.w.is_cuda:
                to_cuda = m_loha.w.cuda()
                to_del = m_loha.w
                m_loha.w = None
                del to_del
                del m_loha.w
                m_loha.w = to_cuda
    if hasattr(m_loha, 'w1'):
        if isinstance(m_loha.w1, torch.Tensor):
            if not m_loha.w1.is_cuda:
                to_cuda = m_loha.w1.cuda()
                to_del = m_loha.w1
                m_loha.w1 = None
                del to_del
                del m_loha.w1
                m_loha.w1 = to_cuda
    if hasattr(m_loha, 'w1a'):
        if isinstance(m_loha.w1a, torch.Tensor):
            if not m_loha.w1a.is_cuda:
                to_cuda = m_loha.w1a.cuda()
                to_del = m_loha.w1a
                m_loha.w1a = None
                del to_del
                del m_loha.w1a
                m_loha.w1a = to_cuda
    if hasattr(m_loha, 'w1b'):
        if isinstance(m_loha.w1b, torch.Tensor):
            if not m_loha.w1b.is_cuda:
                to_cuda = m_loha.w1b.cuda()
                to_del = m_loha.w1b
                m_loha.w1b = None
                del to_del
                del m_loha.w1b
                m_loha.w1b = to_cuda
    if hasattr(m_loha, 'w2'):
        if isinstance(m_loha.w2, torch.Tensor):
            if not m_loha.w2.is_cuda:
                to_cuda = m_loha.w2.cuda()
                to_del = m_loha.w2
                m_loha.w2 = None
                del to_del
                del m_loha.w2
                m_loha.w2 = to_cuda
    if hasattr(m_loha, 'w2a'):
        if isinstance(m_loha.w2a, torch.Tensor):
            if not m_loha.w2a.is_cuda:
                to_cuda = m_loha.w2a.cuda()
                to_del = m_loha.w2a
                m_loha.w2a = None
                del to_del
                del m_loha.w2a
                m_loha.w2a = to_cuda
    if hasattr(m_loha, 'w2b'):
        if isinstance(m_loha.w2b, torch.Tensor):
            if not m_loha.w2b.is_cuda:
                to_cuda = m_loha.w2b.cuda()
                to_del = m_loha.w2b
                m_loha.w2b = None
                del to_del
                del m_loha.w2b
                m_loha.w2b = to_cuda
