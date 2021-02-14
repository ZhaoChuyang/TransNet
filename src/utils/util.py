import torch
from .logger import log, logger
import scipy.io as sio


def save_model(model, optim, name, detail, amp=None):
    path = "checkpoints/%s_ep%d.pt" % (name, detail['epoch'])
    if amp:
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "detail": detail,
            "amp": amp.state_dict(),
        }, path)
    else:
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "detail": detail,
        }, path)
    log("save model to %s" % path)


def load_model(path, model, use_gpu=False, optim=None, amp=None):
    # remap everthing onto CPU
    state = torch.load(str(path), map_location=lambda storage, location: storage)
    model.load_state_dict(state["model"])
    if optim:
        log("loading optim")
        optim.load_state_dict(state["optim"])
    else:
        log("not loading optim")
    if amp:
        log("loading amp")
        amp.load_state_dict(state["amp"])
    else:
        log("not loading amp")
    if use_gpu:
        model.cuda()
    detail = state["detail"]
    log("loaded model from %s" % path)
    return detail