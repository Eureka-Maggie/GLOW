# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union
import torch.distributed as dist
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override
import math
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

import torch.distributed as dist
logger = logging.get_logger(__name__)
from typing import Dict
loss_history: Dict[int, float] = {}
# 2. 定义动态loss的超参数
#DYN_LOSS_ALPHA = 3.0 #最大能偏离1导多远，难样本的权重更高
#DYN_LOSS_BETA = 10.0 #灵敏度
DYN_LOSS_MIN_WEIGHT = 0.5
DYN_LOSS_MAX_WEIGHT = 5

print("DYN_LOSS_MIN_WEIGHT:",DYN_LOSS_MIN_WEIGHT)
print("DYN_LOSS_MAX_WEIGHT:",DYN_LOSS_MAX_WEIGHT)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        self.alpha = kwargs['args'].alpha
        self.beta = kwargs['args'].beta
        
        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        # 0. 从输入中弹出我们添加的 sample_id
        # .get() in case it's not present (e.g., during evaluation)
        sample_ids = inputs.get("sample_id", None)
        
        # 1. 执行标准的模型前向传播
        outputs = model(**inputs)
        
        # 2. 获取每个样本的原始Loss
        logits = outputs.get("logits").float()
        labels = inputs.get("labels")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # 3. 计算每个样本的平均Loss
        per_token_loss = per_token_loss.view(shift_labels.size(0), shift_labels.size(1))
        label_mask = (shift_labels != IGNORE_INDEX)
        sample_loss_sum = torch.sum(per_token_loss * label_mask, dim=1)
        sample_effective_tokens = torch.clamp(torch.sum(label_mask, dim=1), min=1)
        current_losses_per_sample = sample_loss_sum / sample_effective_tokens

        # 如果没有 sample_ids 或者不在训练阶段，则返回未加权的平均loss
        if sample_ids is None or not self.is_in_train:
            loss = current_losses_per_sample.mean()
            return (loss, outputs) if return_outputs else loss
        
        # --- 以下是我们的动态加权逻辑 ---
        avg_delta_l = 0.0
        std_delta_l = 0.0

        # 3. 获取权重
        current_epoch = self.state.epoch
        if current_epoch < 1.0: # 第一个epoch不加权
            weights = torch.ones_like(current_losses_per_sample)
        else:
            previous_losses = [
                loss_history.get(sid.item(), current_losses_per_sample.mean().item()) 
                for sid in sample_ids
            ]
            previous_losses = torch.tensor(previous_losses, device=current_losses_per_sample.device)
            
            delta_l = previous_losses - current_losses_per_sample.detach()
            avg_delta_l = delta_l.mean().item()
            std_delta_l = delta_l.std().item()

            # weights = 1 + self.alpha * (1 - torch.sigmoid(self.beta * delta_l))
            weights = self.alpha * (1 - torch.sigmoid(self.beta * delta_l))
            weights = torch.clamp(weights, min=DYN_LOSS_MIN_WEIGHT, max=DYN_LOSS_MAX_WEIGHT)

        # 4. 计算加权后的最终loss
        final_loss = (current_losses_per_sample * weights.detach()).mean()



        epoch_val = float(self.state.epoch or 0.0)
        step_val  = int(getattr(self.state, "global_step", 0))
        rank = 0; world_size = 1
        try:
            if hasattr(self, "accelerator") and self.accelerator is not None:
                rank, world_size = int(self.accelerator.process_index), int(self.accelerator.num_processes)
            elif dist.is_available() and dist.is_initialized():
                rank, world_size = dist.get_rank(), dist.get_world_size()
        except Exception:
            pass

        # 第一轮 epoch 没历史，用全零；也可以选择直接跳过写
        if 'delta_l' not in locals():
            delta_l = torch.zeros_like(current_losses_per_sample)
            previous_losses = current_losses_per_sample.detach()

        # ---- 每卡写 JSONL（样本级）----
        if sample_ids is not None:
            sids_cpu = [int(x) for x in sample_ids.detach().cpu().tolist()]
            prev_cpu = [float(x) for x in previous_losses.detach().cpu().tolist()]
            curr_cpu = [float(x) for x in current_losses_per_sample.detach().cpu().tolist()]
            dlt_cpu  = [float(x) for x in delta_l.detach().cpu().tolist()]
            wts_cpu  = [float(x) for x in weights.detach().cpu().tolist()]

            out_dir = os.path.join(self.args.output_dir, "delta_l_logs")
            if rank == 0:
                os.makedirs(out_dir, exist_ok=True)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            # rank_file = os.path.join(out_dir, f"delta_rank{rank}.jsonl")
            # record = {
            #     "global_step": step_val,
            #     "epoch": epoch_val,
            #     "rank": rank,
            #     "entries": [
            #         {"sample_id": sid, "prev": p, "curr": c, "delta_l": d, "rel": (d / max(p, 1e-8)), "weight": w}
            #         for sid, p, c, d, w in zip(sids_cpu, prev_cpu, curr_cpu, dlt_cpu, wts_cpu)
            #     ],
            # }
            # with open(rank_file, "a", encoding="utf-8") as f:
            #     f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # ---- 本卡上“退化难”和“停滞难”的候选 ----
            prev_t = previous_losses.detach()
            curr_t = current_losses_per_sample.detach()
            dlt_t  = delta_l.detach()  # previous - current
            rel_t  = dlt_t / torch.clamp(prev_t, min=1e-8)

            # 退化难：delta_l 最小（最负）
            if dlt_t.numel() > 0:
                neg_val, neg_idx = torch.min(dlt_t, dim=0)
                neg_sid = sids_cpu[int(neg_idx.item())]
            else:
                neg_val = torch.tensor([math.inf], device=model.device, dtype=torch.float32)[0]
                neg_sid = -1

            # 停滞难：在“当前loss高”的样本里，选相对改善 r 最小（或 |r| 最小但 curr 高）
            # 阈值：批内 P75，你也可以换成固定阈值
            high_thr = torch.quantile(curr_t, 0.75) if curr_t.numel() > 0 else torch.tensor(0.0, device=model.device)
            mask = curr_t >= high_thr
            if mask.any():
                masked_rel = torch.where(mask, rel_t, torch.full_like(rel_t, float('inf')))
                stagn_val, stagn_idx = torch.min(masked_rel, dim=0)  # 最小 r （可能为负，负表示退化）
                stagn_sid = sids_cpu[int(stagn_idx.item())]
                stagn_curr = float(curr_t[int(stagn_idx.item())].item())
            else:
                stagn_val  = torch.tensor([float('inf')], device=model.device, dtype=torch.float32)[0]
                stagn_sid  = -1
                stagn_curr = float('nan')

            # ---- 跨卡聚合（accelerate 优先；否则 dist；否则单卡）----
            def write_global_jsonl(payload: dict):
                gfile = os.path.join(out_dir, "delta_global_hard.jsonl")
                with open(gfile, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            used_accel = False
            try:
                if hasattr(self, "accelerator") and self.accelerator is not None:
                    used_accel = True
                    # gather 退化难
                    t_neg_val = neg_val.view(1)
                    t_neg_sid = torch.tensor([neg_sid], device=model.device, dtype=torch.long)
                    g_neg_vals = self.accelerator.gather_for_metrics(t_neg_val)
                    g_neg_sids = self.accelerator.gather_for_metrics(t_neg_sid)

                    # gather 停滞难
                    t_stagn_val = stagn_val.view(1)
                    t_stagn_sid = torch.tensor([stagn_sid], device=model.device, dtype=torch.long)
                    t_stagn_cur = torch.tensor([stagn_curr], device=model.device, dtype=torch.float32)
                    g_stagn_vals = self.accelerator.gather_for_metrics(t_stagn_val)
                    g_stagn_sids = self.accelerator.gather_for_metrics(t_stagn_sid)
                    g_stagn_curr = self.accelerator.gather_for_metrics(t_stagn_cur)

                    if self.accelerator.is_main_process:
                        neg_vals = g_neg_vals.detach().cpu().tolist()
                        neg_sids = [int(x) for x in g_neg_sids.detach().cpu().tolist()]
                        # 最小 delta_l（最负）对应的 rank
                        neg_rank = min(range(len(neg_vals)), key=lambda i: neg_vals[i])

                        stagn_vals = g_stagn_vals.detach().cpu().tolist()
                        stagn_sids = [int(x) for x in g_stagn_sids.detach().cpu().tolist()]
                        stagn_currs= g_stagn_curr.detach().cpu().tolist()
                        stagn_rank = min(range(len(stagn_vals)), key=lambda i: stagn_vals[i])

                        write_global_jsonl({
                            "global_step": step_val,
                            "epoch": epoch_val,
                            "hard_regress": {
                                "sample_id": neg_sids[neg_rank],
                                "delta_l": float(neg_vals[neg_rank]),
                                "from_rank": int(neg_rank)
                            },
                            "hard_stagnant": {
                                "sample_id": stagn_sids[stagn_rank],
                                "rel_improve": float(stagn_vals[stagn_rank]),
                                "curr_loss": float(stagn_currs[stagn_rank]),
                                "from_rank": int(stagn_rank)
                            }
                        })
            except Exception:
                used_accel = False

            if not used_accel:
                if dist.is_available() and dist.is_initialized():
                    # 退化难
                    t_neg_val = neg_val.view(1)
                    t_neg_sid = torch.tensor([neg_sid], device=model.device, dtype=torch.long)
                    gv_vals = [torch.zeros_like(t_neg_val) for _ in range(world_size)]
                    gv_sids = [torch.zeros_like(t_neg_sid) for _ in range(world_size)]
                    dist.all_gather(gv_vals, t_neg_val)
                    dist.all_gather(gv_sids, t_neg_sid)

                    # 停滞难
                    t_stagn_val = stagn_val.view(1)
                    t_stagn_sid = torch.tensor([stagn_sid], device=model.device, dtype=torch.long)
                    t_stagn_cur = torch.tensor([stagn_curr], device=model.device, dtype=torch.float32)
                    gs_vals = [torch.zeros_like(t_stagn_val) for _ in range(world_size)]
                    gs_sids = [torch.zeros_like(t_stagn_sid) for _ in range(world_size)]
                    gs_curr= [torch.zeros_like(t_stagn_cur) for _ in range(world_size)]
                    dist.all_gather(gs_vals, t_stagn_val)
                    dist.all_gather(gs_sids, t_stagn_sid)
                    dist.all_gather(gs_curr, t_stagn_cur)

                    if rank == 0:
                        neg_vals = [float(x.item()) for x in gv_vals]
                        neg_sids = [int(x.item()) for x in gv_sids]
                        neg_rank = min(range(len(neg_vals)), key=lambda i: neg_vals[i])

                        stagn_vals = [float(x.item()) for x in gs_vals]
                        stagn_sids = [int(x.item()) for x in gs_sids]
                        stagn_currs= [float(x.item()) for x in gs_curr]
                        stagn_rank = min(range(len(stagn_vals)), key=lambda i: stagn_vals[i])

                        write_global_jsonl({
                            "global_step": step_val,
                            "epoch": epoch_val,
                            "hard_regress": {
                                "sample_id": neg_sids[neg_rank],
                                "delta_l": float(neg_vals[neg_rank]),
                                "from_rank": int(neg_rank)
                            },
                            "hard_stagnant": {
                                "sample_id": stagn_sids[stagn_rank],
                                "rel_improve": float(stagn_vals[stagn_rank]),
                                "curr_loss": float(stagn_currs[stagn_rank]),
                                "from_rank": int(stagn_rank)
                            }
                        })
                else:
                    # 单卡直接写
                    write_global_jsonl({
                        "global_step": step_val,
                        "epoch": epoch_val,
                        "hard_regress": {
                            "sample_id": int(neg_sid),
                            "delta_l": float(neg_val.item()),
                            "from_rank": 0
                        },
                        "hard_stagnant": {
                            "sample_id": int(stagn_sid),
                            "rel_improve": float(stagn_val.item()),
                            "curr_loss": float(stagn_curr),
                            "from_rank": 0
                        }
                    })        
        if self.is_in_train:
            # 补充 delta_l 向量，第一轮 epoch 用全 0（此时 weights=1）
            if current_epoch is None:
                current_epoch = 0.0
            delta_l_vec = torch.zeros_like(current_losses_per_sample)
            if current_epoch >= 1.0:
                # 上面分支里已计算 delta_l
                delta_l_vec = delta_l

            # 本地统计
            local_metrics_tensor = torch.tensor([
                current_losses_per_sample.mean().item(),   # [0] avg_loss_unweighted
                weights.mean().item(),                     # [1] avg_weight
                weights.std().item(),                      # [2] std_weight
                weights.min().item(),                      # [3] weight_min
                weights.max().item(),                      # [4] weight_max
                delta_l_vec.mean().item(),                 # [5] avg_delta_l
                delta_l_vec.std().item(),                  # [6] std_delta_l
                (delta_l_vec < 0).float().mean().item(),   # [7] percent_harder（变差样本占比）
                (delta_l_vec > 0).float().mean().item(),   # [8] percent_easier（变好样本占比）
                final_loss.item(),                         # [9] weighted_loss
            ], device=model.device, dtype=torch.float32)

            # 优先用 accelerate 规约；否则回落到 torch.distributed；再不行就保持本地值
            is_main = True
            reduced = local_metrics_tensor
            try:
                if hasattr(self, "accelerator") and self.accelerator is not None:
                    reduced = self.accelerator.reduce(local_metrics_tensor, reduction="mean")
                    is_main = self.accelerator.is_main_process
                else:
                    
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(local_metrics_tensor, op=dist.ReduceOp.SUM)
                        world_size = dist.get_world_size()
                        reduced = local_metrics_tensor / world_size
                        is_main = (dist.get_rank() == 0)
                    else:
                        reduced = local_metrics_tensor
                        is_main = True
            except Exception:
                # 出现任何问题也不要影响训练流程
                reduced = local_metrics_tensor
                is_main = True

            if is_main and getattr(self, "state", None) is not None:
                self.state.log_history.append({
                    "_dyn_metrics": True,
                    "global_step": int(getattr(self.state, "global_step", 0)),
                    "epoch": float(getattr(self.state, "epoch", 0.0) or 0.0),

                    # 动态指标（全局均值）
                    "avg_loss_unweighted": float(reduced[0].item()),
                    "avg_weight":          float(reduced[1].item()),
                    "std_weight":          float(reduced[2].item()),
                    "weight_min":          float(reduced[3].item()),
                    "weight_max":          float(reduced[4].item()),
                    "avg_delta_l":         float(reduced[5].item()),
                    "std_delta_l":         float(reduced[6].item()),
                    "percent_harder":      float(reduced[7].item()),
                    "percent_easier":      float(reduced[8].item()),
                    "weighted_loss":       float(reduced[9].item()),
                })
        # 5. 更新loss历史记录并进行分布式同步
        local_updates = []
        for i, sid in enumerate(sample_ids):
            local_updates.append((sid.item(), current_losses_per_sample[i].detach().item()))
        if dist.is_initialized():
            global_updates_list = [None] * dist.get_world_size()
            dist.all_gather_object(global_updates_list, local_updates)
            for process_updates in global_updates_list:
                for sid, loss_val in process_updates:
                    loss_history[sid] = loss_val
        else: # 如果不是分布式环境，则直接更新
            for sid, loss_val in local_updates:
                loss_history[sid] = loss_val
        #print("loss_history:", loss_history)
        return (final_loss, outputs) if return_outputs else final_loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
