import torch
import composable_lora
import composable_lycoris

def on_enable():
    #backup original forward methods
    composable_lora.backup_lora_Linear_forward = torch.nn.Linear.forward
    composable_lora.backup_lora_Conv2d_forward = torch.nn.Conv2d.forward

    if hasattr(torch.nn, 'Linear_forward_before_lyco'):
        #if a1111-sd-webui-lycoris installed, backup it's forward methods
        import lycoris
        composable_lycoris.has_webui_lycoris = True
        if hasattr(torch.nn, 'Linear_forward_before_lyco'):
            composable_lycoris.backup_Linear_forward_before_lyco = torch.nn.Linear_forward_before_lyco
        if hasattr(torch.nn, 'Linear_load_state_dict_before_lyco'):
            composable_lycoris.backup_Linear_load_state_dict_before_lyco = torch.nn.Linear_load_state_dict_before_lyco
        if hasattr(torch.nn, 'Conv2d_forward_before_lyco'):
            composable_lycoris.backup_Conv2d_forward_before_lyco = torch.nn.Conv2d_forward_before_lyco
        if hasattr(torch.nn, 'Conv2d_load_state_dict_before_lyco'):
            composable_lycoris.backup_Conv2d_load_state_dict_before_lyco = torch.nn.Conv2d_load_state_dict_before_lyco
        if hasattr(torch.nn, 'MultiheadAttention_forward_before_lyco'):
            composable_lycoris.backup_MultiheadAttention_forward_before_lyco = torch.nn.MultiheadAttention_forward_before_lyco
        if hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lyco'):
            composable_lycoris.backup_MultiheadAttention_load_state_dict_before_lyco = torch.nn.MultiheadAttention_load_state_dict_before_lyco

        torch.nn.Linear.forward = composable_lora.lora_Linear_forward
        torch.nn.Conv2d.forward = composable_lora.lora_Conv2d_forward
        torch.nn.MultiheadAttention.forward = lycoris.lyco_MultiheadAttention_forward
        torch.nn.MultiheadAttention._load_from_state_dict = lycoris.lyco_MultiheadAttention_load_state_dict
    else:
        composable_lycoris.has_webui_lycoris = False

    if (composable_lora.should_reload() or (torch.nn.Linear.forward != composable_lora.lora_Linear_forward)):
        if composable_lora.enabled:
            torch.nn.Linear.forward = composable_lora.lora_Linear_forward
            torch.nn.Conv2d.forward = composable_lora.lora_Conv2d_forward

def on_disable():
    torch.nn.Linear.forward = composable_lora.backup_lora_Linear_forward
    torch.nn.Conv2d.forward = composable_lora.backup_lora_Conv2d_forward
    if hasattr(torch.nn, 'Linear_forward_before_lyco'):
        composable_lycoris.has_webui_lycoris = True
        if hasattr(composable_lycoris, 'backup_Linear_forward_before_lyco'):
            torch.nn.Linear_forward_before_lyco = composable_lycoris.backup_Linear_forward_before_lyco
        if hasattr(composable_lycoris, 'backup_Linear_load_state_dict_before_lyco'):
            torch.nn.Linear_load_state_dict_before_lyco = composable_lycoris.backup_Linear_load_state_dict_before_lyco
        if hasattr(composable_lycoris, 'backup_Conv2d_forward_before_lyco'):
            torch.nn.Conv2d_forward_before_lyco = composable_lycoris.backup_Conv2d_forward_before_lyco
        if hasattr(composable_lycoris, 'backup_Conv2d_load_state_dict_before_lyco'):
            torch.nn.Conv2d_load_state_dict_before_lyco = composable_lycoris.backup_Conv2d_load_state_dict_before_lyco
        if hasattr(composable_lycoris, 'backup_MultiheadAttention_forward_before_lyco'):
            torch.nn.MultiheadAttention_forward_before_lyco = composable_lycoris.backup_MultiheadAttention_forward_before_lyco
        if hasattr(composable_lycoris, 'backup_MultiheadAttention_load_state_dict_before_lyco'):
            torch.nn.MultiheadAttention_load_state_dict_before_lyco = composable_lycoris.backup_MultiheadAttention_load_state_dict_before_lyco
    else:
        composable_lycoris.has_webui_lycoris = False