import torch

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
