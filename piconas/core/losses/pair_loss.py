import torch


def pair_loss(outputs, labels):
    output = outputs.unsqueeze(1)
    output1 = output.repeat(1, outputs.shape[0])
    label = labels.unsqueeze(1)
    label1 = label.repeat(1, labels.shape[0])
    tmp = (output1 - output1.t()) * torch.sign(label1 - label1.t())
    tmp = torch.log(1 + torch.exp(-tmp))
    eye_tmp = tmp * torch.eye(len(tmp)).cuda()
    new_tmp = tmp - eye_tmp
    loss = torch.sum(new_tmp) / (outputs.shape[0] * (outputs.shape[0] - 1))
    return loss
