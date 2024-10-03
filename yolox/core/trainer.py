import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend


import torch_xla.core.xla_model as xm

import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_op_registry as xor
import inspect
from torch.utils.data import DataLoader
# from xla_grad_trainer import GradScaler

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    MlflowLogger,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


# torch.distributed.init_process_group(backend='xla')

class Trainer:
    def __init__(self, exp: Exp, args):
        self.exp = exp
        self.args = args

        # training related attributes
        self.max_epoch = 1 # exp.max_epoch
        self.amp_training = args.fp16
        # self.scaler = GradScaler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        # self.rank = get_rank()
        self.rank=xm.get_ordinal()
        # self.local_rank = get_local_rank()
        # self.device = xm.xla_device()  # Set device to TPU
        self.device = 'xla'
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attributes
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            logger.error(f"Exception in training: {e}")
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
        # xm.mark_step()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            print("------------------------ FINISHED before_iter(), PRINTING METRICS\n")
            print(met.metrics_report())

            self.train_one_iter()
            print("------------------------ FINISHED train_one_iter(), PRINTING METRICS\n")
            print(met.metrics_report())

            self.after_iter()
            print("------------------------ FINISHED after_iter(), PRINTING METRICS\n")
            print(met.metrics_report())
            # xm.mark_step()  # Ensure TPU operations are synchronized

    def train_one_iter(self):
        print("TTTTTTTTT STARTING TRAIN_ONE_ITER\n")
        iter_start_time = time.time()
        batch = next(iter(self.prefetcher))
        inps, targets, _, _ = batch
        # print(f"TTTTTTTTTTTT inps = {inps} , inps.size() = {inps.size()}")
        # print(f"TTTTTTTTTTTT targets = {targets} , targets.size() = {targets.size()}\n")
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        inps = inps.to(self.device)
        targets = targets.to(self.device)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        self.optimizer.zero_grad()
        outputs = self.model(inps, targets)
        loss = outputs["total_loss"]
        print("BOUTT DO SCALER.SCALE\n")
        self.scaler.scale(loss).backward()
        print("FINISHED DA SCALER")
        xm.optimizer_step(optimizer) # XLA MP: performs grad allreduce and optimizer step

        # print(f"************************ PRINTING OUTPUTS from trainer {outputs}\n")
        



        # with torch.cuda.amp.autocast(enabled=self.amp_training):
        #     outputs = self.model(inps, targets)

        # with autocast(xm.xla_device()):
        #     outputs = self.model(inps, targets)

        # outputs = self.model(inps, targets)
        # print(f"************************ PRINTING OUTPUTS {outputs}\n")


        # loss = outputs["total_loss"]
        # print(f"************************ PRINTING LOSS {loss}\n")
        

        # print("************************ BOUTTA DO xm.mark_step() after loss = outputs...\n")
        # xm.mark_step()
        # print("************************ JUST DID xm.mark_step()\n")

        # self.optimizer.zero_grad()
        # print("************************ BOUTTA DO xm.mark_step() after self.optimizer.zero_grad()\n")
        # xm.mark_step()
        # print("************************ JUST DID xm.mark_step()\n")

        # self.scaler.scale(loss).backward()
        # print("************************ BOUTTA DO xm.mark_step() after self.scaler.scale(loss).backward()\n")
        # xm.mark_step()
        # print("************************ JUST DID xm.mark_step()\n")






        print("BOUTT DO SCALER.STEP\n")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        print("BOUTTA DO MARK STEP AFTER SCALER.UPDATE")
        # xm.mark_step()

        print("FINISHED THEEEEEE MARKSTEP FROM TRAINER.PY")

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info(f"args: {self.args}")
        logger.info(f"exp value:\n{self.exp}")

        # model related initialization
        model = self.exp.get_model()
        logger.info(f"Model Summary: {get_model_info(model, self.exp.test_size)}")
        model.to(self.device)
        world_size = xm.xrt_world_size()

        # solver related initialization
        # self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=.01 * world_size)

        # resume training if applicable
        model = self.resume_train(model)

        # data related initialization
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs


        print(f"WORLD_SIZE: {world_size}")
        if world_size > 1:
            train_sampler = DistributedSampler(exp.dataset, 
                                                num_replicas=world_size,
                                                rank=xm.get_ordinal(),
                                                shuffle=True)
            self.is_distributed=True

        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
    # XLA MP: use MpDeviceLoader from torch_xla.distributed


        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = pl.MpDeviceLoader(self.train_loader, self.device)  # DataLoader for XLA

        # self.max_iter = len(self.train_loader)
        self.max_iter = 1
        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            elif self.args.logger == "mlflow":
                self.mlflow_logger = MlflowLogger()
                self.mlflow_logger.setup(args=self.args, exp=self.exp)
            else:
                raise ValueError("logger must be either 'tensorboard', 'mlflow' or 'wandb'")

        logger.info("Training start...")
        logger.info(f"\n{model}")

    def after_train(self):
        logger.info(
            f"Training of experiment is done and the best AP is {self.best_ap * 100:.2f}"
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
            elif self.args.logger == "mlflow":
                metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
                self.mlflow_logger.on_train_end(self.args, file_name=self.file_name,
                                                metadata=metadata)

    def before_epoch(self):
        logger.info(f"---> start train epoch {self.epoch + 1}")

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        if (self.iter + 1) % self.exp.print_interval == 0:
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"

            progress_str = f"epoch: {self.epoch + 1}/{self.max_epoch}, iter: {self.iter + 1}/{self.max_iter}"
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                [f"{k}: {v.latest:.1f}" for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                [f"{k}: {v.avg:.3f}s" for k, v in time_meter.items()]
            )

            mem_str = f"gpu mem: {gpu_mem_usage():.0f}Mb, mem: {mem_usage():.1f}Gb"

            logger.info(
                f"{progress_str}, {mem_str}, {time_str}, {loss_str}, lr: {self.meter['lr'].latest:.3e}"
                + (f", size: {self.input_size[0]}, {eta_str}")
            )

            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
                if self.args.logger == 'mlflow':
                    logs = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    logs.update({"train/lr": self.meter["lr"].latest})
                    self.mlflow_logger.on_log(self.args, self.exp, self.epoch + 1, logs)

            self.meter.clear_meters()

        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                f"loaded checkpoint '{self.args.resume}' (epoch {self.start_epoch})"
            )
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            if self.args.logger == "mlflow":
                logs = {
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "val/best_ap": round(self.best_ap, 3),
                    "train/epoch": self.epoch + 1,
                }
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch + 1, logs)
            logger.info(f"\n{summary}")
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

        if self.args.logger == "mlflow":
            metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
            self.mlflow_logger.save_checkpoints(self.args, self.exp, self.file_name, self.epoch,
                                                metadata, update_best_ckpt)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info(f"Save weights to {self.file_name}")
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap
                    }
                )



# class GradScaler(torch.cuda.amp.GradScaler):
#   """
#   An torch_xla variant of torch.cuda.amp.GradScaler that helps perform the steps of gradient scaling
#   conveniently.
#   Args:
#       init_scale (float, optional, default=2.**16):  Initial scale factor.
#       growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
#           :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
#       backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
#           :meth:`update` if inf/NaN gradients occur in an iteration.
#       growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
#           that must occur for the scale to be multiplied by ``growth_factor``.
#       enabled (bool, optional, default=True):  If ``False``, disables gradient scaling. :meth:`step` simply
#           invokes the underlying ``optimizer.step()``, and other methods become no-ops.
#       use_zero_grad (bool, optional, default=False): If ``True``, enables the torch_xla specific zero gradients
#           optimization that performs ``optimizer.step()`` with gradients set to zero instead of skipping it when
#           inf/NaN gradients occur. This may improve the performance by removing the barrier in GradScaler.
#   """

#   def __init__(
#       self,
#       init_scale=2.0**16,
#       growth_factor=2.0,
#       backoff_factor=0.5,
#       growth_interval=2000,
#       enabled=True,
#       use_zero_grad=False,
#   ):
#     super().__init__(
#         init_scale=init_scale,
#         growth_factor=growth_factor,
#         backoff_factor=backoff_factor,
#         growth_interval=growth_interval,
#         enabled=enabled,
#     )

#     def get_scaling_factor(a):

#       def if_true(a):
#         return xb.Op.zero(a.builder())

#       def if_false(a):
#         return xb.Op.one(a.builder())

#       cond = a != xb.Op.zero(a.builder())
#       return cond.mkconditional((a,), if_true, if_false)

#     self.get_scaling_factor = xor.register("get_scaling_factor",
#                                            get_scaling_factor)
#     self.use_zero_grad = use_zero_grad

#   def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
#     retval = None
#     is_syncfree_optim = "found_inf" in inspect.signature(
#         optimizer.step).parameters
#     if is_syncfree_optim:
#       found_inf = torch.stack(
#           tuple(optimizer_state["found_inf_per_device"].values())).sum()
#       kwargs['found_inf'] = found_inf
#       retval = optimizer.step(*args, **kwargs)
#     elif self.use_zero_grad:
#       found_inf = torch.stack(
#           tuple(optimizer_state["found_inf_per_device"].values())).sum()
#       scaling_factor = self.get_scaling_factor(found_inf)
#       for grad in xm._fetch_gradients(optimizer):
#         grad.nan_to_num_()
#         grad.mul_(scaling_factor)
#       retval = optimizer.step(*args, **kwargs)
#     else:
#       print("+++++++++++++++++++++ DOING MARK_STEP IN GRADSCALER YOOO - didnt actually")
#     #   xm.mark_step()
#       print("+++++++++++++++++++++ FINISHED MARK_STEP IN GRADSCALER YOOO")

#       if not sum(
#           v.item() for v in optimizer_state["found_inf_per_device"].values()):
#         retval = optimizer.step(*args, **kwargs)
#     return retval
