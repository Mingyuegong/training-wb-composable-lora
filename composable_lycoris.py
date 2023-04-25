import torch
from modules import shared

#support for <lyco:MODEL> 
def lycoris_forward(compvis_module, input, res):
    import composable_lora as lora_controller
    import lora
    import lycoris
    if len(lycoris.loaded_lycos) == 0:
        return res
    
    lycoris_layer_name_loading : str | None = getattr(compvis_module, 'lyco_layer_name', None)
    if lycoris_layer_name_loading is None:
        return res
    #let it type is actually a string
    lycoris_layer_name : str = str(lycoris_layer_name_loading)
    del lycoris_layer_name_loading

    sd_module = shared.sd_model.lora_layer_mapping.get(lycoris_layer_name, None)
    num_loras = len(lora.loaded_loras) + len(lycoris.loaded_lycos)

    if lora_controller.text_model_encoder_counter == -1:
        lora_controller.text_model_encoder_counter = len(lora_controller.prompt_loras) * num_loras

    tmp_check_loras = [] #store which lora are already apply
    tmp_check_loras.clear()

    for m_lycoris in lycoris.loaded_lycos:
        module = m_lycoris.modules.get(lycoris_layer_name, None)
        if module is None:
            #fix the lyCORIS issue
            check_lycoris_end_layer(lycoris_layer_name, res, num_loras)
            continue

        current_lora = m_lycoris.name
        lora_already_used = False
        if current_lora in tmp_check_loras:
            lora_already_used = True
        #store the applied lora into list
        tmp_check_loras.append(current_lora)
        if lora_already_used:
            check_lycoris_end_layer(lycoris_layer_name, res, num_loras)
            continue

        converted_module = convert_lycoris(module, sd_module)
        if converted_module is None:
            check_lycoris_end_layer(lycoris_layer_name, res, num_loras)
            continue
        
        patch = get_lora_patch(converted_module, input, res, lycoris_layer_name)
        alpha = get_lora_alpha(converted_module, 1.0)
        num_prompts = len(lora_controller.prompt_loras)

        # print(f"lora.name={m_lora.name} lora.mul={m_lora.multiplier} alpha={alpha} pat.shape={patch.shape}")
        res = lora_controller.apply_composable_lora(lycoris_layer_name, m_lycoris, "lyco", patch, alpha, res, num_loras, num_prompts)

    return res

def get_lora_inference(module, input):
    if hasattr(module, 'inference'): #support for lyCORIS
        return module.inference(input)
    elif hasattr(module, 'up'):     #LoRA
        return module.up(module.down(input))
    else:
        return None

def get_lora_patch(module, input, res, lora_layer_name):
    if is_loha(module):
        if input.is_cuda: #if is cuda, pass to cuda; otherwise do nothing
            pass_loha_to_gpu(module)
    if getattr(shared.opts, "lora_apply_to_outputs", False) and res.shape == input.shape:
        inference = get_lora_inference(module, res)
        if inference is not None: 
            return inference
        else:
            converted_module = convert_lycoris(module, shared.sd_model.lora_layer_mapping.get(lora_layer_name, None))
            if converted_module is not None:
                return get_lora_inference(converted_module, res)
            else:
                raise NotImplementedError(
                    "Your settings, extensions or models are not compatible with each other."
                )
    else:
        inference = get_lora_inference(module, input)
        if inference is not None: 
            return inference
        else:
            converted_module = convert_lycoris(module, shared.sd_model.lora_layer_mapping.get(lora_layer_name, None))
            if converted_module is not None:
                return get_lora_inference(converted_module, input)
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

def lycoris_get_multiplier(lycoris_model, lora_layer_name):
    multiplier = 1.0
    if hasattr(lycoris_model, 'te_multiplier'):
        multiplier = (
            lycoris_model.te_multiplier if 'transformer' in lora_layer_name[:20] 
            else lycoris_model.unet_multiplier
        )
    elif hasattr(lycoris_model, 'multiplier'):
        multiplier = getattr(lycoris_model, 'multiplier', 1.0)
    return multiplier

def lycoris_get_multiplier_normalized(lycoris_model, lora_layer_name):
    multiplier = 1.0
    if hasattr(lycoris_model, 'te_multiplier'):
        te_multiplier = 1.0
        unet_multiplier = lycoris_model.unet_multiplier / lycoris_model.te_multiplier
        multiplier = (
            te_multiplier if 'transformer' in lora_layer_name[:20] 
            else unet_multiplier
        )
    return multiplier

class FakeModule(torch.nn.Module):
    def __init__(self, weight, func):
        super().__init__()
        self.weight = weight
        self.func = func
    
    def forward(self, x):
        return self.func(x)

class FullModule:
    def __init__(self):
        self.weight = None
        self.alpha = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.up = None
    
    def down(self, x):
        return x
    
    def inference(self, x):
        return self.op(x, self.weight, **self.extra_args)

class LoraUpDownModule:
    def __init__(self):
        self.up_model = None
        self.mid_model = None
        self.down_model = None
        self.alpha = None
        self.dim = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.bias = None
        self.up = None
    
    def down(self, x):
        return x
    
    def inference(self, x):
        if hasattr(self, 'bias') and isinstance(self.bias, torch.Tensor):
            out_dim = self.up_model.weight.size(0)
            rank = self.down_model.weight.size(0)
            rebuild_weight = (
                self.up_model.weight.reshape(out_dim, -1) @ self.down_model.weight.reshape(rank, -1)
                + self.bias
            ).reshape(self.shape)
            return self.op(
                x, rebuild_weight,
                bias=None,
                **self.extra_args
            )
        else:
            if self.mid_model is None:
                return self.up_model(self.down_model(x))
            else:
                return self.up_model(self.mid_model(self.down_model(x)))

def make_weight_cp(t, wa, wb):
    temp = torch.einsum('i j k l, j r -> i r k l', t, wb)
    return torch.einsum('i j k l, i r -> r j k l', temp, wa)

class LoraHadaModule:
    def __init__(self):
        self.t1 = None
        self.w1a = None
        self.w1b = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self.alpha = None
        self.dim = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.bias = None
        self.up = None
    
    def down(self, x):
        return x
    
    def inference(self, x):
        if hasattr(self, 'bias') and isinstance(self.bias, torch.Tensor):
            bias = self.bias
        else:
            bias = 0
        
        if self.t1 is None:
            return self.op(
                x,
                ((self.w1a @ self.w1b) * (self.w2a @ self.w2b) + bias).view(self.shape),
                bias=None,
                **self.extra_args
            )
        else:
            return self.op(
                x,
                (make_weight_cp(self.t1, self.w1a, self.w1b) 
                 * make_weight_cp(self.t2, self.w2a, self.w2b) + bias).view(self.shape),
                bias=None,
                **self.extra_args
            )

def make_kron(orig_shape, w1, w2):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(orig_shape)

class LoraKronModule:
    def __init__(self):
        self.w1 = None
        self.w1a = None
        self.w1b = None
        self.w2 = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self._alpha = None
        self.dim = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.bias = None
        self.up = None
    
    @property
    def alpha(self):
        if self.w1a is None and self.w2a is None:
            return None
        else:
            return self._alpha
    
    @alpha.setter
    def alpha(self, x):
        self._alpha = x
    
    def down(self, x):
        return x
    
    def inference(self, x):
        if hasattr(self, 'bias') and isinstance(self.bias, torch.Tensor):
            bias = self.bias
        else:
            bias = 0
        
        if self.t2 is None:
            return self.op(
                x,
                (torch.kron(self.w1, self.w2a@self.w2b) + bias).view(self.shape),
                **self.extra_args
            )
        else:
            # will raise NotImplemented Error
            return self.op(
                x,
                (torch.kron(self.w1, make_weight_cp(self.t2, self.w2a, self.w2b)) + bias).view(self.shape),
                **self.extra_args
            )

def convert_lycoris(lycoris_module, sd_module):
    result_module = getattr(lycoris_module, 'lyco_converted_lora_module', None)
    if result_module is not None:
        return result_module
    if lycoris_module.__class__.__name__ == "LycoUpDownModule" or lycoris_module.__class__.__name__ == "LoraUpDownModule":
        result_module = LoraUpDownModule()
        if (type(sd_module) == torch.nn.Linear
            or type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear
            or type(sd_module) == torch.nn.MultiheadAttention):
            result_module.op = torch.nn.functional.linear
        elif type(sd_module) == torch.nn.Conv2d:
            result_module.op = torch.nn.functional.conv2d
            result_module.extra_args = {
                'stride': sd_module.stride,
                'padding': sd_module.padding
            }
        else:
            return None
        result_module.up_model = lycoris_module.up_model
        result_module.mid_model = lycoris_module.mid_model
        result_module.down_model = lycoris_module.down_model
        result_module.alpha = lycoris_module.alpha
        result_module.dim = lycoris_module.dim 
        result_module.shape = lycoris_module.shape
        result_module.bias = lycoris_module.bias
        result_module.up = FakeModule(
            result_module.up_model.weight,
            result_module.inference
        )
    elif lycoris_module.__class__.__name__ == "FullModule":
        result_module = FullModule()
        result_module.weight = lycoris_module.weight#.to(device=devices.device, dtype=devices.dtype)
        result_module.alpha = lycoris_module.alpha
        result_module.shape = lycoris_module.shape
        result_module.up = FakeModule(
            result_module.weight,
            result_module.inference
        )
        if len(result_module.weight.shape)==2:
            result_module.op = torch.nn.functional.linear
            result_module.extra_args = {
                'bias': None
            }
        else:
            result_module.op = torch.nn.functional.conv2d
            result_module.extra_args = {
                'stride': sd_module.stride,
                'padding': sd_module.padding,
                'bias': None
            }
        setattr(lycoris_module, "lyco_converted_lora_module", result_module)
        return result_module
    elif lycoris_module.__class__.__name__ == "LycoHadaModule" or lycoris_module.__class__.__name__ == "LoraHadaModule":
        result_module = LoraHadaModule()
        result_module.t1 = lycoris_module.t1
        result_module.w1a = lycoris_module.w1a
        result_module.w1b = lycoris_module.w1b
        result_module.t2 = lycoris_module.t2
        result_module.w2a = lycoris_module.w2a
        result_module.w2b = lycoris_module.w2b
        result_module.alpha = lycoris_module.alpha
        result_module.dim = lycoris_module.dim
        result_module.shape = lycoris_module.shape
        result_module.bias = lycoris_module.bias
        result_module.up = FakeModule(
            result_module.t1 if result_module.t1 is not None else result_module.w1a,
            result_module.inference
        )
        if (type(sd_module) == torch.nn.Linear
            or type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear
            or type(sd_module) == torch.nn.MultiheadAttention):
            result_module.op = torch.nn.functional.linear
        elif type(sd_module) == torch.nn.Conv2d:
            result_module.op = torch.nn.functional.conv2d
            result_module.extra_args = {
                'stride': sd_module.stride,
                'padding': sd_module.padding
            }
    elif lycoris_module.__class__.__name__ == "LycoKronModule" or lycoris_module.__class__.__name__ == "LoraKronModule" :
        result_module = LoraKronModule()
        result_module.w1 = lycoris_module.w1
        result_module.w1a = lycoris_module.w1a
        result_module.w1b = lycoris_module.w1b
        result_module.w2 = lycoris_module.w2
        result_module.t2 = lycoris_module.t2
        result_module.w2a = lycoris_module.w2a
        result_module.w2b = lycoris_module.w2b
        result_module._alpha = lycoris_module._alpha
        result_module.dim = lycoris_module.dim
        result_module.shape = lycoris_module.shape
        result_module.bias = lycoris_module.bias
        result_module.up = FakeModule(
            result_module.w1a if result_module.w1a is not None else result_module.w2a,
            result_module.inference
        )
        if (any(isinstance(sd_module, torch_layer) for torch_layer in 
                [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention])):
            result_module.op = torch.nn.functional.linear
        elif isinstance(sd_module, torch.nn.Conv2d):
            result_module.op = torch.nn.functional.conv2d
            result_module.extra_args = {
                'stride': sd_module.stride,
                'padding': sd_module.padding
            }
    if result_module is not None:
        setattr(lycoris_module, "lyco_converted_lora_module", result_module)
        return result_module
    return None

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

has_webui_lycoris : bool = False