import torch
from .logger import log, logger


def save_model(model, optim, detail):
    path = "checkpoints/attn_siamese_ep%d.pt" % detail['epoch']
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "detail": detail,
    }, path)
    log("save model to %s" % path)


def load_model(path, model, use_gpu=False, optim=None):
    # remap everthing onto CPU
    state = torch.load(str(path), map_location=lambda storage, location: storage)
    model.load_state_dict(state["model"])
    if optim:
        log("loading optim")
        optim.load_state_dict(state["model"])
    else:
        log("not loading optim")
    if use_gpu:
        model.cuda()
    detail = state["detail"]
    log("loaded model from %s" % path)
    return detail