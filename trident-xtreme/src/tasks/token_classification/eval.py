import torch
from trident import TridentModule

from src.modules.functional.metrics import cka


def layer_cka(self: TridentModule, outputs: dict, *args, **kwargs) -> dict:
    # (N, L, C)
    # (N, L) L incl padding
    clf = self.model.classifier.weight.data
    N = outputs["logits"].shape[0]
    preds = outputs["logits"].argmax(-1)
    src_preds = preds[: N // 2]
    trg_preds = preds[N // 2 :]
    src_embeds = outputs["hidden_states"][: N // 2]
    trg_embeds = outputs["hidden_states"][N // 2 :]

    attention_mask = outputs["attention_mask"].clamp(min=0, max=1).bool()
    src_attention_mask = attention_mask[: N // 2]
    trg_attention_mask = attention_mask[N // 2 :]

    cka_ = []
    for (i, sp, tp, se, te) in zip(
        range(N), src_preds, trg_preds, src_embeds, trg_embeds
    ):
        sp_ = sp[src_attention_mask[i]][1:-1]  # excl. special tokens
        tp_ = tp[trg_attention_mask[i]][1:-1]
        se_ = se[src_attention_mask[i]][1:-1][sp_ != 0]  # excl. seecial tokens
        te_ = te[trg_attention_mask[i]][1:-1][tp_ != 0]
        if len(se_) == 0 or len(te_) == 0:
            pass
        else:
            se_cka = cka(clf, se_)
            te_cka = cka(clf, te_)
            val = (se_cka - te_cka).abs()
            if torch.isnan(val):
                import pudb
                pu.db
            cka_.append(val)
    outputs["cka"] = torch.Tensor(cka_).mean()
    return outputs


def identity(x):
    return x


def sent_cka(logits, hidden_states, attention_mask) -> torch.Tensor:
    # (N, L, C)
    # (N, L) L incl padding
    preds = logits.argmax(-1)
    probs = logits.softmax(-1)
    N = logits.shape[0]
    src_preds = preds[: N // 2]
    trg_preds = preds[N // 2 :]
    src_embeds = hidden_states[: N // 2]
    trg_embeds = hidden_states[N // 2 :]
    src_probs = probs[: N // 2]
    trg_probs = probs[N // 2 :]

    attention_mask = attention_mask.clamp(min=0, max=1).bool()
    src_attention_mask = attention_mask[: N // 2]
    trg_attention_mask = attention_mask[N // 2 :]

    cka_ = []
    for (i, sp, tp, se, te) in zip(
        range(N), src_preds, trg_preds, src_embeds, trg_embeds
    ):
        # get class for each valid
        # (N, L-) L excl padding
        # se_ = se - se.mean(0)
        # te_ = te - te.mean(0)
        # se_ = se / se.norm(2, -1, keepdim=True)
        # te_ = te / te.norm(2, -1, keepdim=True)
        sp_ = sp[src_attention_mask[i]][1:-1]  # excl. special tokens
        tp_ = tp[trg_attention_mask[i]][1:-1]
        se_ = se[src_attention_mask[i]][1:-1][sp_ != 0]  # excl. seecial tokens
        te_ = te[trg_attention_mask[i]][1:-1][tp_ != 0]
        if len(se_) > 0 and len(te_) > 0:
            cka_.append(cka(se_, te_))
        else:
            cka_.append(0)
        # union = torch.cat([sp_, tp_])
        # join = union.unique()
        #
        #
        # # total_cka = cka(se_, te_)
        # # total_cka = (se_ @ te_.T).mean()
        # scka = []
        # for cls_ in join:
        #     sp_mask = sp_ == cls_
        #     tp_mask = tp_ == cls_
        #     # import pudb
        #     # pu.db
        #     if any([x == 0 for x in (sp_mask.sum(), tp_mask.sum())]):
        #         scka.append(0)
        #     else:
        #         se_c = se_[sp_mask]
        #         te_c = te_[tp_mask]
        #         # cls_cka = (se_c @ te_c.T).mean()
        #         cls_cka = cka(se_c, te_c)
        #         # import pudb
        #         # pu.db
        #         scka.append(cls_cka)
        # cka_.append(torch.Tensor(scka).mean())
    return torch.Tensor(cka_).mean()
