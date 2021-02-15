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


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def get_features(cfg, model, loader):
    features = torch.FloatTensor()
    count = 0
    for inputs, ids, is_query, indices in loader:
        n, c, h, w = inputs.size()
        count += n
        # print(count)
        ff = torch.FloatTensor(n, 512).zero_()
        if cfg.use_gpu:
            ff = ff.cuda()
        # if opt.PCB:
        #     ff = torch.FloatTensor(n, 2048, 6).zero_().cuda()  # we have six parts

        for i in range(2):
            if (i == 1):
                inputs = fliplr(inputs)
            if cfg.use_gpu:
                input_img = inputs.cuda()
            else:
                input_img = inputs
            # for scale in ms:
            #     if scale != 1:
            #         # bicubic is only  available in pytorch>= 1.1
            #         input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic',
            #                                               align_corners=False)
            # using scale 1
            outputs = model(input_img)
            ff += outputs
        # norm feature
        # if opt.PCB:
        #     # feature size (n,2048,6)
        #     # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        #     # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        #     fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        #     ff = ff.div(fnorm.expand_as(ff))
        #     ff = ff.view(ff.size(0), -1)
        # else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features