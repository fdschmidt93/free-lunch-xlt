import random
from itertools import accumulate
from pathlib import Path
from typing import Optional, Tuple, Union

# import deepspeed
import torch
from pytorch_lightning import LightningModule
from trident import TridentModule
from trident.utils.logging import get_logger

log = get_logger(__name__)


def average_checkpoints(
    module: LightningModule,
    checkpoints: Optional[str] = None,
    avg_basepath: Optional[str] = None,
    avg_dataseed: Optional[int] = None,
    avg_seed: list[int] = [42, 43, 44, 45, 46, 47],
    ckpt_name: Optional[str] = None,
):
    if isinstance(checkpoints, str):
        ckpt_paths = list(Path(checkpoints).glob("*.ckpt"))
    else:
        ckpt_paths = []
        assert avg_basepath is not None
        avg_basepath = Path(avg_basepath)
        for seed in avg_seed:
            p = avg_basepath.joinpath(
                f"data_seed-{avg_dataseed}", f"seed-{seed}", "lr-2e-05", "checkpoints"
            )
            ckpts = list(p.glob("*.ckpt"))
            if ckpt_name is not None:
                ckpts = [c for c in ckpts if str(c).endswith(ckpt_name)]
                assert len(ckpts) == 1
                ckpt_paths.append(ckpts[0])
                print(str(ckpts[0]))
            else:
                ckpt_paths.extend(ckpts)
    try:
        log.info([x.name for x in ckpt_paths])
    except:
        pass
    ckpts = [torch.load(ckpt, map_location="cpu")["state_dict"] for ckpt in ckpt_paths]
    keys = ckpts[0].keys()
    N = len(ckpts)
    for key in keys:
        if not "int" in str(ckpts[0][key].dtype):
            ckpts[0][key] *= 1 / N
            for ckpt in ckpts[1:]:
                ckpts[0][key] += (1 / N) * ckpt[key]
        else:
            print(key)
    module.load_state_dict(ckpts[0], strict=True)
    try:
        log.info(f"Successfully averaged weights from {checkpoints}")
    except:
        print(f"Successfully averaged weights from {checkpoints}")


def one_hot_encoding(y: torch.Tensor, num_labels: int = 3) -> torch.Tensor:
    N = y.shape[0]
    out = torch.zeros((N, num_labels), dtype=torch.long).to(y.device)
    out[torch.arange(N, device=y.device), y] = 1
    return out


class MultiTaskSequenceClassification(TridentModule):
    def __init__(
        self,
        avg_ckpts: Optional[str] = None,
        avg_basepath: Optional[str] = None,
        avg_dataseed: Optional[int] = None,
        avg_seed: list[int] = [42, 43, 44, 45, 46],
        ckpt_name: Optional[str] = None,
        freeze_clf: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.avg_ckpts = avg_ckpts
        self.avg_basepath = avg_basepath
        self.avg_dataseed = avg_dataseed
        self.avg_seed = avg_seed
        self.ckpt_name = ckpt_name
        self.freeze_clf = freeze_clf

    def setup(self, stage: str):
        super().setup(stage)
        if self.freeze_clf:
            for name, weights in self.named_parameters():
                if any(freeze in name for freeze in ["classifier", "qa_outputs"]):
                    weights.requires_grad = False
                    print(f"Freezing {name}")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = {
            k: self.model(
                input_ids=v["input_ids"],
                attention_mask=v["attention_mask"],
                labels=v["labels"],
            ).loss
            for k, v in batch.items()
        }
        for k, v in losses.items():
            self.log(f"{k}/loss", v)
        loss = torch.stack(list(losses.values())).mean()
        self.log("train/loss", loss)
        return loss

    def on_test_epoch_start(self):
        if self.avg_ckpts is not None:
            average_checkpoints(
                self,
                self.avg_ckpts,
                self.avg_basepath,
                self.avg_dataseed,
                self.avg_seed,
                self.ckpt_name,
            )


class MultiTaskTokenClassification(TridentModule):
    def __init__(
        self,
        avg_ckpts: Optional[str] = None,
        avg_basepath: Optional[str] = None,
        avg_dataseed: Optional[int] = None,
        avg_seed: list[int] = [42, 43, 44, 45, 46],
        ckpt_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.avg_ckpts = avg_ckpts
        self.avg_basepath = avg_basepath
        self.avg_dataseed = avg_dataseed
        self.avg_seed = avg_seed
        self.ckpt_name = ckpt_name

    def setup(self, stage: str):
        super().setup(stage)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = {
            k: self.model(
                input_ids=v["input_ids"],
                attention_mask=v["attention_mask"],
                labels=v["labels"],
            ).loss
            for k, v in batch.items()
        }
        for k, v in losses.items():
            self.log(f"{k}/loss", v)
        loss = torch.stack(list(losses.values())).mean()
        self.log("train/loss", loss)
        return loss

    def on_test_epoch_start(self):
        if self.avg_ckpts is not None:
            average_checkpoints(
                self,
                self.avg_ckpts,
                self.avg_basepath,
                self.avg_dataseed,
                self.avg_seed,
                self.ckpt_name,
            )


class GradientVaccinationForTokenClassification(TridentModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def setup(self, stage: str):
        super().setup(stage)

    @staticmethod
    def split_batches(
        batch: dict[str, dict]
    ) -> Tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        langs = tuple(batch.keys())
        oracle_lang = random.choice(langs)
        oracle_batch = batch.pop(oracle_lang)
        assert oracle_batch is not None
        return batch, oracle_batch

    def multi_tasking_loss(
        self, batch: dict[str, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        losses = {
            k: self.model(
                input_ids=v["input_ids"],
                attention_mask=v["attention_mask"],
                labels=v["labels"],
            ).loss
            for k, v in batch.items()
        }
        for k, v in losses.items():
            self.log(f"{k}/loss", v)
        loss = torch.stack(list(losses.values())).mean()
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        """
        Adapted implemention from
        https://github.com/fe1ixxu/Mixed-Gradient-Few-Shot/blob/f3e759365a589201311e8b3c426c91401582cd89/third_party/run_tag.py#L78-L138
        """
        opt = self.optimizers()
        opt.zero_grad()

        train_batch, oracle_batch = self.split_batches(batch)

        loss = self.multi_tasking_loss(train_batch)
        self.manual_backward(loss)

        grad_shapes = [
            p.shape if (p.requires_grad is True and p.grad is not None) else None
            for group in opt.param_groups
            for p in group["params"]
        ]
        grad_numel = [
            p.numel() if (p.requires_grad is True and p.grad is not None) else 0
            for group in opt.param_groups
            for p in group["params"]
        ]
        grad = torch.cat(
            [
                p.grad.detach().clone().flatten()
                if (p.requires_grad is True and p.grad is not None)
                else torch.zeros(p.numel(), device=self.device)
                for group in opt.param_groups
                for p in group["params"]
            ],
            dim=0,
        )
        self.model.zero_grad()

        oracle_loss = self.model(**oracle_batch).loss

        self.manual_backward(oracle_loss)

        oracle_grad = torch.cat(
            [
                p.grad.detach().clone().flatten()
                if (p.requires_grad is True and p.grad is not None)
                else torch.zeros(p.numel(), device=self.device)
                for group in opt.param_groups
                for p in group["params"]
            ],
            dim=0,
        )
        self.model.zero_grad()

        inner_product = torch.sum(grad * oracle_grad)
        project_direction = inner_product / torch.sum(oracle_grad * oracle_grad)
        grad = (
            grad
            - torch.min(
                project_direction,
                torch.zeros_like(project_direction, device=self.device),
            )
            * oracle_grad
        )

        indices = [
            0,
        ] + [v for v in accumulate(grad_numel)]
        params = [p for group in opt.param_groups for p in group["params"]]
        assert len(params) == len(grad_shapes) == len(indices[:-1])
        for param, grad_shape, start_idx, end_idx in zip(
            params, grad_shapes, indices[:-1], indices[1:]
        ):
            if grad_shape is not None:
                param.grad[...] = grad[start_idx:end_idx].view(
                    grad_shape
                )  # copy proj grad
        opt.step()
        sch = self.lr_schedulers()
        sch.step()


class GradientVaccinationForSequenceClassification(TridentModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def setup(self, stage: str):
        super().setup(stage)

    @staticmethod
    def split_batches(
        batch: dict[str, dict]
    ) -> Tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        langs = tuple(batch.keys())
        oracle_lang = random.choice(langs)
        oracle_batch = batch.pop(oracle_lang)
        assert oracle_batch is not None
        return batch, oracle_batch

    def multi_tasking_loss(
        self, batch: dict[str, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        losses = {
            k: self.model(
                input_ids=v["input_ids"],
                attention_mask=v["attention_mask"],
                labels=v["labels"],
            ).loss
            for k, v in batch.items()
        }
        for k, v in losses.items():
            self.log(f"{k}/loss", v)
        loss = torch.stack(list(losses.values())).mean()
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        """
        Adapted implemention from
        https://github.com/fe1ixxu/Mixed-Gradient-Few-Shot/blob/f3e759365a589201311e8b3c426c91401582cd89/third_party/run_tag.py#L78-L138
        """
        opt = self.optimizers()
        opt.zero_grad()

        train_batch, oracle_batch = self.split_batches(batch)

        loss = self.multi_tasking_loss(train_batch)
        self.manual_backward(loss)

        grad_shapes = [
            p.shape if (p.requires_grad is True and p.grad is not None) else None
            for group in opt.param_groups
            for p in group["params"]
        ]
        grad_numel = [
            p.numel() if (p.requires_grad is True and p.grad is not None) else 0
            for group in opt.param_groups
            for p in group["params"]
        ]
        grad = torch.cat(
            [
                p.grad.detach().clone().flatten()
                if (p.requires_grad is True and p.grad is not None)
                else torch.zeros(p.numel(), device=self.device)
                for group in opt.param_groups
                for p in group["params"]
            ],
            dim=0,
        )
        self.model.zero_grad()

        oracle_loss = self.model(**oracle_batch).loss

        self.manual_backward(oracle_loss)

        oracle_grad = torch.cat(
            [
                p.grad.detach().clone().flatten()
                if (p.requires_grad is True and p.grad is not None)
                else torch.zeros(p.numel(), device=self.device)
                for group in opt.param_groups
                for p in group["params"]
            ],
            dim=0,
        )
        self.model.zero_grad()

        inner_product = torch.sum(grad * oracle_grad)
        project_direction = inner_product / torch.sum(oracle_grad * oracle_grad)
        grad = (
            grad
            - torch.min(
                project_direction,
                torch.zeros_like(project_direction, device=self.device),
            )
            * oracle_grad
        )

        indices = [
            0,
        ] + [v for v in accumulate(grad_numel)]
        params = [p for group in opt.param_groups for p in group["params"]]
        assert len(params) == len(grad_shapes) == len(indices[:-1])
        for param, grad_shape, start_idx, end_idx in zip(
            params, grad_shapes, indices[:-1], indices[1:]
        ):
            if grad_shape is not None:
                param.grad[...] = grad[start_idx:end_idx].view(
                    grad_shape
                )  # copy proj grad
        opt.step()
        sch = self.lr_schedulers()
        sch.step()


class MultiTaskQuestionAnswering(TridentModule):
    def __init__(
        self,
        avg_ckpts: Optional[str] = None,
        avg_basepath: Optional[str] = None,
        avg_dataseed: Optional[int] = None,
        avg_seed: list[int] = [42, 43, 44, 45, 46],
        ckpt_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.avg_ckpts = avg_ckpts
        self.avg_basepath = avg_basepath
        self.avg_dataseed = avg_dataseed
        self.avg_seed = avg_seed
        self.ckpt_name = ckpt_name

    def setup(self, stage: str):
        super().setup(stage)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = {k: self.model(**v).loss for k, v in batch.items()}
        for k, v in losses.items():
            self.log(f"{k}/loss", v)
        loss = torch.stack(list(losses.values())).mean()
        self.log("train/loss", loss)
        return loss

    def on_test_epoch_start(self):
        if self.avg_ckpts is not None:
            average_checkpoints(
                self,
                self.avg_ckpts,
                self.avg_basepath,
                self.avg_dataseed,
                self.avg_seed,
                self.ckpt_name,
            )


class GradientVaccinationForQuestionAnswering(TridentModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def setup(self, stage: str):
        super().setup(stage)

    @staticmethod
    def split_batches(
        batch: dict[str, dict]
    ) -> Tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        langs = tuple(batch.keys())
        oracle_lang = random.choice(langs)
        oracle_batch = batch.pop(oracle_lang)
        assert oracle_batch is not None
        return batch, oracle_batch

    def multi_tasking_loss(
        self, batch: dict[str, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        losses = {k: self.model(**v).loss for k, v in batch.items()}
        for k, v in losses.items():
            self.log(f"{k}/loss", v)
        loss = torch.stack(list(losses.values())).mean()
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        """
        Adapted implemention from
        https://github.com/fe1ixxu/Mixed-Gradient-Few-Shot/blob/f3e759365a589201311e8b3c426c91401582cd89/third_party/run_tag.py#L78-L138
        """
        opt = self.optimizers()
        opt.zero_grad()

        train_batch, oracle_batch = self.split_batches(batch)

        loss = self.multi_tasking_loss(train_batch)
        self.manual_backward(loss)

        grad_shapes = [
            p.shape if (p.requires_grad is True and p.grad is not None) else None
            for group in opt.param_groups
            for p in group["params"]
        ]
        grad_numel = [
            p.numel() if (p.requires_grad is True and p.grad is not None) else 0
            for group in opt.param_groups
            for p in group["params"]
        ]
        grad = torch.cat(
            [
                p.grad.detach().clone().flatten()
                if (p.requires_grad is True and p.grad is not None)
                else torch.zeros(p.numel(), device=self.device)
                for group in opt.param_groups
                for p in group["params"]
            ],
            dim=0,
        )
        self.model.zero_grad()

        oracle_loss = self.model(**oracle_batch).loss

        self.manual_backward(oracle_loss)

        oracle_grad = torch.cat(
            [
                p.grad.detach().clone().flatten()
                if (p.requires_grad is True and p.grad is not None)
                else torch.zeros(p.numel(), device=self.device)
                for group in opt.param_groups
                for p in group["params"]
            ],
            dim=0,
        )
        self.model.zero_grad()

        inner_product = torch.sum(grad * oracle_grad)
        project_direction = inner_product / torch.sum(oracle_grad * oracle_grad)
        grad = (
            grad
            - torch.min(
                project_direction,
                torch.zeros_like(project_direction, device=self.device),
            )
            * oracle_grad
        )

        indices = [
            0,
        ] + [v for v in accumulate(grad_numel)]
        params = [p for group in opt.param_groups for p in group["params"]]
        assert len(params) == len(grad_shapes) == len(indices[:-1])
        for param, grad_shape, start_idx, end_idx in zip(
            params, grad_shapes, indices[:-1], indices[1:]
        ):
            if grad_shape is not None:
                param.grad[...] = grad[start_idx:end_idx].view(
                    grad_shape
                )  # copy proj grad
        opt.step()
        sch = self.lr_schedulers()
        sch.step()


class TrainBody(TridentModule):
    def __init__(
        self,
        clf_path: Optional[str] = None,
        avg_ckpts: Optional[str] = None,
        avg_basepath: Optional[str] = None,
        avg_dataseed: Optional[int] = None,
        avg_seed: list[int] = [42, 43, 44, 45, 46],
        ckpt_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.avg_ckpts = avg_ckpts
        self.avg_basepath = avg_basepath
        self.avg_dataseed = avg_dataseed
        self.avg_seed = avg_seed
        self.ckpt_name = ckpt_name
        self.clf_path = clf_path

    def on_test_epoch_start(self):
        if self.avg_ckpts is not None:
            average_checkpoints(
                self,
                self.avg_ckpts,
                self.avg_basepath,
                self.avg_dataseed,
                self.avg_seed,
                self.ckpt_name,
            )

    def setup(self, stage: str):
        super().setup(stage)
        # reload classifier
        if isinstance(self.clf_path, str):
            state_dict = self.state_dict()
            loaded_state_dict = torch.load(self.clf_path, map_location="cpu")["state_dict"]
            loaded_state_dict = {
                k: v for k, v in loaded_state_dict.items() if any(unfrozen in k for unfrozen in ["classifier", "qa_outputs"])
            }
            state_dict.update(loaded_state_dict)
            print(loaded_state_dict.keys())
            self.load_state_dict(state_dict, strict=True)
            print("Loaded classifier successfully!")
            for name, weights in self.named_parameters():
                if name in loaded_state_dict:
                    weights.requires_grad = False
                    print(f"Freezing {name}")
