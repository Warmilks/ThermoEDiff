"""Training loop and joint denoising/reconstruction objective."""
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import EsmTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from metrics import Metrictor

def dict_to_device(data, device):
    for k in data:
        if data[k] is not None:
            try:
                data[k] = data[k].to(device)
            except:
                continue
    return data


class BaseClassifier:
    def __init__(self, model):
        pass

    def calculate_y_logit(self, X):
        pass

    def calculate_y_prob(self, X):
        pass

    def calculate_y(self, X):
        pass

    def calculate_y_prob_by_iterator(self, dataStream):
        pass

    def calculate_y_by_iterator(self, dataStream):
        pass

    def calculate_loss(self, X, Y):
        pass
    def train(self, trainDataSet, validDataSet=None, otherDataSet=None,
              batchSize=256, maxSteps=1000000, evalSteps=100, earlyStop=10, saveSteps=-1,
              isHigherBetter=False, metrics="LOSS", report=["LOSS"],
              savePath='model', dataLoadNumWorkers=4, prefetchFactor=16, pinMemory=False,
              warmupSteps=0, SEED=0, doEvalTrain=True, doEvalValid=True, dropLast=True, evalFirst=False):

        metrictor = self.metrictor if hasattr(self, "metrictor") else Metrictor()
        device = next(self.model.parameters()).device

        rank = torch.distributed.get_rank() if self.DDP else 0
        is_rank0 = (rank == 0)

        lr = self.optimizer.param_groups[0]['lr']
        stop = False

        decaySteps = maxSteps - warmupSteps
        schedulerRLR = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_training_steps=maxSteps,
            num_warmup_steps=warmupSteps
        )

        # =========================
        # Sampler / DataLoader
        # =========================
        is_iterable = isinstance(trainDataSet, torch.utils.data.IterableDataset)
        trainSampler = None
        if self.DDP and (not is_iterable):
            trainSampler = torch.utils.data.distributed.DistributedSampler(
                trainDataSet, shuffle=True, seed=SEED
            )

        dl_kwargs = dict(
            batch_size=batchSize,
            num_workers=dataLoadNumWorkers,
            pin_memory=pinMemory,
            collate_fn=self.collateFunc,
            drop_last=dropLast
        )
        if dataLoadNumWorkers > 0:
            dl_kwargs["prefetch_factor"] = prefetchFactor

        # 训练流：IterableDataset 不要 shuffle；map-style 才考虑 sampler/shuffle
        trainStream = DataLoader(
            trainDataSet,
            shuffle=(trainSampler is None and (not is_iterable)),
            sampler=trainSampler,
            **dl_kwargs
        )

        # evalTrainStream：你如果 doEvalTrain=False，它不会被用到（后面也会尊重 doEvalTrain）
        evalTrainStream = DataLoader(
            trainDataSet,
            shuffle=False,
            sampler=trainSampler,
            **dl_kwargs
        )

        evalValidStream = None
        if validDataSet is not None:
            if isinstance(validDataSet, DataLoader):
                evalValidStream = validDataSet
            else:
                is_iterable_valid = isinstance(validDataSet, torch.utils.data.IterableDataset)
                if self.DDP and (not is_iterable_valid):
                    validSampler = torch.utils.data.distributed.DistributedSampler(
                        validDataSet, shuffle=False, seed=SEED
                    )
                else:
                    validSampler = None

                evalValidStream = DataLoader(
                    validDataSet,
                    shuffle=False,
                    sampler=validSampler,
                    **dl_kwargs
                )

        evalOtherStream = None
        if otherDataSet is not None:
            if isinstance(otherDataSet, list):
                if isinstance(otherDataSet[0], DataLoader):
                    evalOtherStream = otherDataSet
                else:
                    otherSampler = [None if not self.DDP else torch.utils.data.distributed.DistributedSampler(ods, shuffle=False, seed=SEED)
                                    for ods in otherDataSet]
                    evalOtherStream = [DataLoader(ods, batch_size=batchSize, shuffle=False,
                                                 num_workers=dataLoadNumWorkers, pin_memory=pinMemory,
                                                 collate_fn=self.collateFunc,
                                                 sampler=os,
                                                 prefetch_factor=(prefetchFactor if dataLoadNumWorkers > 0 else None))
                                     for ods, os in zip(otherDataSet, otherSampler)]
            else:
                if isinstance(otherDataSet, DataLoader):
                    evalOtherStream = otherDataSet
                else:
                    otherSampler = None if not self.DDP else torch.utils.data.distributed.DistributedSampler(otherDataSet, shuffle=False, seed=SEED)
                    # 注意：这里如果你 otherDataSet 也是 IterableDataset，建议你也按 valid 的逻辑禁用 sampler
                    evalOtherStream = DataLoader(
                        otherDataSet,
                        batch_size=batchSize,
                        shuffle=False,
                        num_workers=dataLoadNumWorkers,
                        pin_memory=pinMemory,
                        collate_fn=self.collateFunc,
                        sampler=otherSampler,
                        prefetch_factor=(prefetchFactor if dataLoadNumWorkers > 0 else None)
                    )

        # 在 while True: 之前（创建 trainStream 之后）加上：
        is_iterable = isinstance(trainDataSet, torch.utils.data.IterableDataset)

        # 对无限流：用一个持续的迭代器，别每个 epoch 重建，否则每轮都从同一随机种子开头，等价“重复第一段数据”
        train_iter = iter(trainStream) if is_iterable else None

        # 约定：IterableDataset 的“一个 epoch = epoch_steps 个 step”
        # 你当前脚本里 evalSteps = steps_per_epoch，所以这里直接用 evalSteps 就行
        epoch_steps = int(evalSteps) if is_iterable else None
        if is_iterable and epoch_steps <= 0:
            raise ValueError("For IterableDataset, evalSteps must be > 0 to define epoch length.")

        # =========================
        # Train loop state
        # =========================
        mtc = 0.0
        bestMtc = (-1e18 if isHigherBetter else 1e18)
        stopSteps = 0

        e, locStep = 0, 0
        while True:
            e += 1
            if is_rank0:
                print(f"Training the step {locStep} at epoch {e}'s data...")

            # ------------------------------
            # 关键改动：IterableDataset 用固定步数定义一个 epoch
            # ------------------------------
            if is_iterable:
                pbar = tqdm(range(epoch_steps), disable=(not is_rank0))
                for _ in pbar:
                    # 无限流正常不会 StopIteration，这里只是兜底
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        train_iter = iter(trainStream)
                        data = next(train_iter)

                    self.to_train_mode()

                    if not evalFirst:
                        data = dict_to_device(data, device=device)
                        loss = self._train_step(data)
                        schedulerRLR.step()

                        # 只让 rank0 更新 tqdm 文本，避免多进程 stdout 争用拖慢
                        if is_rank0:
                            cur_lr = self.optimizer.param_groups[0]['lr']
                            if isinstance(loss, dict):
                                tmp = "; ".join([f"{k}: {loss[k]:.3f}" for k in loss])
                                pbar.set_description(
                                    f"Training Loss: {tmp}; Learning rate: {cur_lr:.6f}; Stop rounds: {stopSteps}"
                                )
                            else:
                                pbar.set_description(
                                    f"Training Loss: {loss}; Learning rate: {cur_lr:.6f}; Stop rounds: {stopSteps}"
                                )

                        locStep += 1

                        if locStep >= maxSteps:   # 建议用 >= 更严谨
                            if is_rank0:
                                print(f"Reach the max steps {maxSteps}... break...")
                            stop = True
                            break

                    # ========== DDP-safe eval ==========
                    # 关键原则：只要内部有 all_reduce，就必须所有 rank 进入同一段代码
                    if (validDataSet is not None) and (locStep % evalSteps == 0):
                        if self.DDP:
                            torch.distributed.barrier()

                        if is_rank0:
                            print(f"========== Step:{locStep} at Epoch: {e} ==========")

                        with torch.no_grad():
                            self.to_eval_mode()

                            # --------- (可选) Train eval：默认只让 rank0 跑，避免浪费算力 ----------
                            if doEvalTrain and ((not self.DDP) or is_rank0):
                                if is_rank0:
                                    print("[Total Train]", end="")
                                data_eval = self.calculate_y_prob_by_iterator(
                                    evalTrainStream, showProgress=is_rank0
                                )
                                metrictor.set_data(data_eval)
                                metrictor(report)

                            # --------- Valid eval：DDP 时所有 rank 必须进入（内部 all_reduce 汇总） ----------
                            if doEvalValid and (evalValidStream is not None):
                                if is_rank0:
                                    print("[Total Valid]", end="")

                                mtc = self.stream_token_acc_ddp(
                                    evalValidStream, showProgress=is_rank0
                                )
                                res = {metrics: mtc}

                                if is_rank0:
                                    print(f" ACC={mtc:.4f}", end="")

                                # otherDataSet 评估不含 all_reduce：只让 rank0 做即可
                                if otherDataSet is not None and evalOtherStream is not None and is_rank0:
                                    print("[Total Other]", end="")
                                    if isinstance(evalOtherStream, list):
                                        print()
                                        for ei, eos in enumerate(evalOtherStream):
                                            print(f"{ei:<5d}", end="")
                                            data_other = self.calculate_y_prob_by_iterator(
                                                eos, showProgress=True
                                            )
                                            metrictor.set_data(data_other)
                                            metrictor(report)
                                    else:
                                        data_other = self.calculate_y_prob_by_iterator(
                                            evalOtherStream, showProgress=True
                                        )
                                        metrictor.set_data(data_other)
                                        metrictor(report)

                                if is_rank0:
                                    print("=================================")

                                    improved = (mtc > bestMtc) if isHigherBetter else (mtc < bestMtc)
                                    if improved:
                                        print(f"Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!")
                                        bestMtc = mtc
                                        self.save(f"{savePath}.pkl", e + 1, bestMtc)
                                        stopSteps = 0
                                    else:
                                        stopSteps += 1
                                        if stopSteps >= earlyStop:
                                            print(
                                                f"The val...for more than {earlyStop} evals in epoch {e+1}, stop training."
                                            )
                                            stop = True

                            self.to_train_mode()

                        # --------- 同步 stop：确保所有 rank 一起停/继续 ----------
                        if self.DDP:
                            stop_t = torch.tensor([1 if stop else 0], device=device, dtype=torch.int)
                            torch.distributed.broadcast(stop_t, src=0)
                            stop = bool(stop_t.item())
                            torch.distributed.barrier()

                        if stop:
                            break

                # 结束本 epoch 的固定步数循环

            else:
                # ------------------------------
                # Map-style Dataset：保持你原来的写法
                # ------------------------------
                pbar = tqdm(trainStream, disable=(not is_rank0))

                for data in pbar:
                    self.to_train_mode()

                    if not evalFirst:
                        data = dict_to_device(data, device=device)
                        loss = self._train_step(data)
                        schedulerRLR.step()

                        # 只让 rank0 更新 tqdm 文本，避免多进程 stdout 争用拖慢
                        if is_rank0:
                            cur_lr = self.optimizer.param_groups[0]['lr']
                            if isinstance(loss, dict):
                                tmp = "; ".join([f"{k}: {loss[k]:.3f}" for k in loss])
                                pbar.set_description(
                                    f"Training Loss: {tmp}; Learning rate: {cur_lr:.6f}; Stop rounds: {stopSteps}"
                                )
                            else:
                                pbar.set_description(
                                    f"Training Loss: {loss}; Learning rate: {cur_lr:.6f}; Stop rounds: {stopSteps}"
                                )

                        locStep += 1

                        if locStep >= maxSteps:   # 建议用 >= 更严谨
                            if is_rank0:
                                print(f"Reach the max steps {maxSteps}... break...")
                            stop = True
                            break

                    # ========== DDP-safe eval ==========
                    if (validDataSet is not None) and (locStep % evalSteps == 0):
                        if self.DDP:
                            torch.distributed.barrier()

                        if is_rank0:
                            print(f"========== Step:{locStep} at Epoch: {e} ==========")

                        with torch.no_grad():
                            self.to_eval_mode()

                            if doEvalTrain and ((not self.DDP) or is_rank0):
                                if is_rank0:
                                    print("[Total Train]", end="")
                                data_eval = self.calculate_y_prob_by_iterator(
                                    evalTrainStream, showProgress=is_rank0
                                )
                                metrictor.set_data(data_eval)
                                metrictor(report)

                            if doEvalValid and (evalValidStream is not None):
                                if is_rank0:
                                    print("[Total Valid]", end="")

                                mtc = self.stream_token_acc_ddp(
                                    evalValidStream, showProgress=is_rank0
                                )
                                res = {metrics: mtc}

                                if is_rank0:
                                    print(f" ACC={mtc:.4f}", end="")

                                if otherDataSet is not None and evalOtherStream is not None and is_rank0:
                                    print("[Total Other]", end="")
                                    if isinstance(evalOtherStream, list):
                                        print()
                                        for ei, eos in enumerate(evalOtherStream):
                                            print(f"{ei:<5d}", end="")
                                            data_other = self.calculate_y_prob_by_iterator(
                                                eos, showProgress=True
                                            )
                                            metrictor.set_data(data_other)
                                            metrictor(report)
                                    else:
                                        data_other = self.calculate_y_prob_by_iterator(
                                            evalOtherStream, showProgress=True
                                        )
                                        metrictor.set_data(data_other)
                                        metrictor(report)

                                if is_rank0:
                                    print("=================================")

                                    improved = (mtc > bestMtc) if isHigherBetter else (mtc < bestMtc)
                                    if improved:
                                        print(f"Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!")
                                        bestMtc = mtc
                                        self.save(f"{savePath}.pkl", e + 1, bestMtc)
                                        stopSteps = 0
                                    else:
                                        stopSteps += 1
                                        if stopSteps >= earlyStop:
                                            print(
                                                f"The val...for more than {earlyStop} evals in epoch {e+1}, stop training."
                                            )
                                            stop = True

                            self.to_train_mode()

                        if self.DDP:
                            stop_t = torch.tensor([1 if stop else 0], device=device, dtype=torch.int)
                            torch.distributed.broadcast(stop_t, src=0)
                            stop = bool(stop_t.item())
                            torch.distributed.barrier()

                        if stop:
                            break

            # 额外按 step 保存（可选）
            if saveSteps > 0 and (locStep % saveSteps == 0):
                if (not self.DDP) or (rank == 0):
                    self.save(f"{savePath}_step{locStep}.pkl", e + 1, -1)

            if stop:
                break

        

        # =========================
        # Final (DDP-safe)
        # =========================
        if self.DDP:
            torch.distributed.barrier()

        res = None
        import os

        ckpt_path = f"{savePath}.pkl"
        if os.path.exists(ckpt_path):
            with torch.no_grad():
                # 所有 rank 都 load 同一个 ckpt（保证后面的 all_reduce 能配对）
                self.load(ckpt_path)
                self.to_eval_mode()

                if is_rank0:
                    print("============ Result ============")

                # doEvalTrain 默认只让 rank0 做（不含 all_reduce）
                if doEvalTrain and ((not self.DDP) or is_rank0):
                    print("[Total Train]", end="")
                    data_eval = self.calculate_y_prob_by_iterator(evalTrainStream, showProgress=True)
                    metrictor.set_data(data_eval)
                    metrictor(report)

                # Valid：DDP 时必须所有 rank 都进入（内部 all_reduce）
                if doEvalValid and (evalValidStream is not None):
                    if is_rank0:
                        print("[Total Valid]", end="")
                    mtc = self.stream_token_acc_ddp(evalValidStream, showProgress=is_rank0)
                    res = {metrics: mtc}
                    if is_rank0:
                        print(f" ACC={mtc:.4f}", end="")

                # 先等所有 rank 都用完 ckpt_path，再让 rank0 重命名
                if self.DDP:
                    torch.distributed.barrier()

                if is_rank0:
                    new_path = f"{savePath}_{bestMtc:.2f}.pkl"
                    try:
                        os.rename(ckpt_path, new_path)
                    except OSError:
                        pass
                    print("================================")
        else:
            if is_rank0:
                print(f"[WARN] checkpoint {ckpt_path} not found. Skip load/rename/result-eval.")
            res = None

        if self.DDP:
            torch.distributed.barrier()

        return res



    def to_train_mode(self):
        self.model.train()  # set the module in training mode
        if self.collateFunc is not None:
            self.collateFunc.eval = False

    def to_eval_mode(self):
        self.model.eval()
        if self.collateFunc is not None:
            self.collateFunc.eval = True

    def _train_step(self, data):
        loss = self.calculate_loss(data)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def save(self, path, epochs, bestMtc=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc, 'model': self.model.state_dict()}
        torch.save(stateDict, path)
        print('Model saved in "%s".' % path)

    def load(self, path, key=None, map_location=None):
        parameters = torch.load(path, map_location=map_location)

        if key is None:
            self.model.load_state_dict(parameters['model'])
        else:
            tmp = parameters['model']
            for k in list(tmp.keys()):
                if key not in k:
                    tmp.pop(k)
                else:
                    tmp[k.replace('module.', '').replace(f'{key}.', '')] = tmp.pop(k)
            getattr(self.model, key).load_state_dict(tmp)

        print("%d epochs and %.3lf val Score 's model load finished." % (parameters['epochs'], parameters['bestMtc']))


class SequenceDiff(BaseClassifier):
    def __init__(self, model, optimizer=None, collateFunc=None, AMP=False, DDP=False):
        self.model = model
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
        self.collateFunc = collateFunc
        self.metrictor = Metrictor(batchList=True)
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters()) if optimizer is None else optimizer
        self.AMP = AMP
        if AMP:
            self.scaler = GradScaler()
        self.DDP = DDP

    def calculate_y_logit(self, data, **kwargs):
        return self.model(data, **kwargs)

    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data, T=128)['tkn_pre']
        return {'y_prob': F.softmax(Y_pre, dim=-1)}

    def calculate_y(self, data):
        Y_pre = self.calculate_y_prob(data)['y_prob']
        return {'y_pre': Y_pre.argmax(dim=-1)}

    def calculate_loss_by_iterator(self, dataStream):
        loss, cnt = 0, 0
        for data in dataStream:
            loss += self.calculate_loss(data) * len(data['tokenizedLabArr'])
            cnt += len(data['tokenizedLabArr'])
        return loss / cnt

    def calculate_y_prob_by_iterator(self, dataStream, showProgress=False):
        device = next(self.model.parameters()).device
        YArr, Y_preArr = [], []
        maskPAD = []
        if showProgress:
            dataStream = tqdm(dataStream)
        for data in dataStream:
            data = dict_to_device(data, device=device)
            Y_pre, Y = self.calculate_y_prob(data)['y_prob'].detach().cpu().data.numpy(), data['tokenizedSeqBArr'][
                'input_ids'].detach().cpu().data.numpy()
            # print(Y_pre.argmax(axis=-1), Y)
            maskpad = data['tokenizedSeqBArr']['attention_mask'].detach().cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
            maskPAD.append(maskpad)
        # YArr,Y_preArr = np.concatenate(YArr, axis=0).astype('float32'),np.concatenate(Y_preArr, axis=0).astype('float32')
        return {'y_prob': Y_preArr, 'y_true': YArr, 'maskPAD': maskPAD}

    def calculate_y_by_iterator(self, dataStream):
        tmp = self.calculate_y_prob_by_iterator(dataStream)
        Y_preArr, YArr = tmp['y_prob'], tmp['y_true']
        return {'y_pre': Y_preArr.argmax(dim=-1), 'y_true': YArr}
    def calculate_loss(self, data):
        res = self.calculate_y_logit(data)
        v, tkn_pre = res['y_logit'], res['tkn_pre']

        loss1 = self.criterion1(v, (res['x0'] - res['eps']))

        tknNum = getattr(self.model, "module", self.model).tknNum
        loss2 = self.criterion2(
            tkn_pre.reshape(-1, tknNum),
            data['tokenizedSeqBArr']['input_ids'].reshape(-1)
        )

        return {"MSE": loss1, "CE": loss2}


    def _train_step(self, data):
        self.optimizer.zero_grad()
        if self.AMP:
            with autocast():
                loss = self.calculate_loss(data)
            if isinstance(loss, dict):
                loss_ = 0
                for k in loss:
                    loss_ += loss[k]
                self.scaler.scale(loss_).backward()
            else:
                self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.calculate_loss(data)
            if isinstance(loss, dict):
                loss_ = 0
                for k in loss:
                    loss_ += loss[k]
                loss_.backward()
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()
        return {k: loss[k].detach().cpu().data.numpy() for k in loss} if isinstance(loss,
                                                                                    dict) else loss.detach().cpu().data.numpy()

    @torch.no_grad()
    def stream_token_acc_ddp(self, dataStream, showProgress=False):
        import torch.distributed as dist
        device = next(self.model.parameters()).device

        correct = 0
        total = 0

        if showProgress:
            from tqdm import tqdm
            dataStream = tqdm(dataStream)

        for data in dataStream:
            data = dict_to_device(data, device=device)
            tkn_pre = self.calculate_y_logit(data)["tkn_pre"]
            pred = tkn_pre.argmax(dim=-1)

            true = data["tokenizedSeqBArr"]["input_ids"]
            mask = data["tokenizedSeqBArr"]["attention_mask"].bool()

            correct += (pred[mask] == true[mask]).sum().item()
            total += mask.sum().item()

        ct = torch.tensor([correct, total], device=device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(ct, op=dist.ReduceOp.SUM)

        c, t = ct.tolist()
        return c / max(t, 1)
