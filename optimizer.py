from torch.optim import Adam, SGD, AdamW
from config import BETA, AMSGRAD # Chỉ import hằng số

def get_optimizer(model, lr, weight_decay, opt_name="AdamW"):
    params = model.parameters()
    
    if opt_name == "Adam":
        return Adam(params, lr=lr, betas=BETA, weight_decay=weight_decay, amsgrad=AMSGRAD)
    
    elif opt_name == "AdamW":
        return AdamW(params, lr=lr, betas=BETA, weight_decay=weight_decay, amsgrad=AMSGRAD)
    
    elif opt_name == "SGD":
        return SGD(params, lr=lr, weight_decay=weight_decay)
    
    else:
        print(f"[WARN] Optimizer {opt_name} not found. Defaulting to AdamW.")
        return AdamW(params, lr=lr, weight_decay=weight_decay)