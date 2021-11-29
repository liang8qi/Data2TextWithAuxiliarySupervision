import torch
import torch.nn as nn


class PtrCrossEntropyLoss(object):
    def __init__(self):
        self.nll_loss = nn.NLLLoss(reduction='none')

    def compute(self, sorted_list, indices_list):
        loss = []
        total_n_records = 0
        for i, sorted_col, in enumerate(sorted_list):
            bsz, num, _ = sorted_col.size()
            total_n_records += bsz*num
            sorted_col = sorted_col.view(-1, sorted_col.size(-1))
            indices = indices_list[i].view(-1)
            table_loss = self.nll_loss(sorted_col, indices)
            table_loss = table_loss.view(bsz, -1).sum(1)

            loss.append(table_loss)

        loss = torch.cat(loss, dim=-1)

        loss = loss.mean()
        assert not torch.isnan(loss)

        return loss, total_n_records
