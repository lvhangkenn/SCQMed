import logging
import json
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os
#import wandb
from datasets_mm import EmbDataset
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.labels_2 = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        # self.labels_3 = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        
        self.model = self.model.to(self.device)
        #self.optimizer.load_state_dict(state["optimizer"])
        self.trained_loss = {"total":[],"rqvae":[],"recon":[]}
        self.valid_collision_rate = {"val":[]}
        self.maxe = args.maxe

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            # optimizer = optim.AdamW([
            # {'params': self.model.parameters(), 'lr': learning_rate, 'weight_decay':weight_decay}, 
            # {'params': self.awl.parameters(), 'weight_decay':0}
            # ])
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        # x = data.cpu().detach().numpy()
        # data = self.embedding.weight.cpu().detach().numpy()
        x = data.astype(np.float64)

        n_samples = len(x)

        # 防止 n_clusters 大于样本数
        if n_clusters > n_samples:
            n_clusters = n_samples
            self.logger.warning(f"n_clusters ({n_clusters}) > n_samples ({n_samples}), reset to {n_samples}")

        # 安全计算 size_min: 至少为 1，最多为 n_samples // n_clusters
        size_min = max(1, min(n_samples // (n_clusters * 2), 10))

        # 安全计算 size_max: 至少为 size_min，最多为 n_samples
        size_max = min(n_clusters * 6, n_samples)

        # 确保 size_min <= size_max
        if size_min > size_max:
            size_min = size_max  # 或者直接设为 1，但要保证 <= size_max



        size_min = min(len(data) // (n_clusters * 2), 10)
        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

        return t_centers, t_labels
    
    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path_1,self.args.data_path_2)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)
        print(len(init_loader))
        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_data):
            text, kg, emb_idx = data[0], data[1], data[2]
            text, kg = text.to(self.device), kg.to(self.device)

            self.model.vq_initialization(text, kg)    

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        ##total_cf_loss = 0
        total_quant_loss = 0
        #print(len(train_data))
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )
        # embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]

        # for idx, emb in enumerate(embs):
        #     centers, labels = self.constrained_km(emb)
        #     self.labels[str(idx)] = labels

        # text
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.text_rq.vq_layers]

        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels

        # kg
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.kg_rq.vq_layers]

        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels_2[str(idx)] = labels

        for batch_idx, data in enumerate(iter_data):
            text,kg, emb_idx = data[0], data[1], data[2]
            text, kg = text.to(self.device),kg.to(self.device)
            self.optimizer.zero_grad()
            text_out, kg_out, rq_loss,rq_loss_2, indices, indices_2, dense_out,dense_out_2 = self.model(text,kg, self.labels,self.labels_2)

            loss,_, loss_recon, quant_loss = self.model.compute_loss(text_out,kg_out,rq_loss,rq_loss_2, emb_idx, dense_out,dense_out_2, text, kg)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            # iter_data.set_postfix_str("Loss: {:.4f}, RQ Loss: {:.4f}".format(loss.item(),rq_loss.item()))
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            ##total_cf_loss += (cf_loss.item() if cf_loss != 0 else cf_loss)
            total_quant_loss += quant_loss.item()
        
        text_usage = self.model.text_rq.get_codebook_usage()
        kg_usage = self.model.kg_rq.get_codebook_usage()

        print("Text codebook usage per layer:", [f"{u:.1%}" for u in text_usage])
        print("KG codebook usage per layer:", [f"{u:.1%}" for u in kg_usage])

        # 可选：重置计数器以统计下一个 epoch
        self.model.text_rq.reset_usage_counts()
        self.model.kg_rq.reset_usage_counts()

        return total_loss, total_recon_loss, None, quant_loss.item()

    # @torch.no_grad()
    # def _valid_epoch(self, valid_data):

    #     self.model.eval()

    #     iter_data =tqdm(
    #             valid_data,
    #             total=len(valid_data),
    #             ncols=100,
    #             desc=set_color(f"Evaluate   ", "pink"),
    #         )
    #     indices_set = set()

    #     num_sample = 0
    #     embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
    #     for idx, emb in enumerate(embs):
    #         centers, labels = self.constrained_km(emb)
    #         self.labels[str(idx)] = labels
    #     for batch_idx, data in enumerate(iter_data):

    #         data, emb_idx = data[0], data[1]
    #         num_sample += len(data)
    #         data = data.to(self.device)
    #         indices = self.model.get_indices(data, self.labels)
    #         indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
    #         for index in indices:
    #             code = "-".join([str(int(_)) for _ in index])
    #             indices_set.add(code)

    #     collision_rate = (num_sample - len(indices_set))/num_sample
    #     # balance_score = self.balance_overall(tokens_appearance)
    #     # wandb.log({"collision_rate": collision_rate, "balance_score": 0})


    #     return collision_rate

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()

        text_indices_list = []
        kg_indices_list = []

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluate Collision", "pink"),
        )

        # 获取当前码本聚类标签（与训练一致）
        # 注意：这里我们不需要重新聚类，直接用 self.labels / self.labels_2 即可
        for batch_idx, data in enumerate(iter_data):
            text, kg, emb_idx = data[0], data[1], data[2]
            text = text.to(self.device)
            kg = kg.to(self.device)

            # 编码
            text_z = self.model.text_encoder(text)
            kg_z = self.model.kg_encoder(kg)

            # 获取量化索引（use_sk=False 避免 Sinkhorn 干扰评估）
            _, _, indices_text = self.model.text_rq(text_z, self.labels, use_sk=False)
            _, _, indices_kg = self.model.kg_rq(kg_z, self.labels_2, use_sk=False)

            text_indices_list.append(indices_text.cpu())
            kg_indices_list.append(indices_kg.cpu())

        # 拼接所有 batch
        all_text_indices = torch.cat(text_indices_list, dim=0) # (N, L)
        all_kg_indices = torch.cat(kg_indices_list, dim=0) # (N, L)

        def compute_collision(indices_tensor):
            N = indices_tensor.size(0)
            # 转为字符串元组集合
            idx_tuples = [tuple(row.tolist()) for row in indices_tensor]
            unique_count = len(set(idx_tuples))
            collision_rate = 1.0 - unique_count / N if N > 0 else 1.0
            return collision_rate

        text_collision = compute_collision(all_text_indices)
        kg_collision = compute_collision(all_kg_indices)

        avg_collision = (text_collision + kg_collision) / 2.0

        print(
            f"Collision Rate — Text: {text_collision:.4f}, KG: {kg_collision:.4f}, Avg: {avg_collision:.4f}"
        )

        return avg_collision, text_collision, kg_collision

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss,):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        #train_loss_output += set_color("cf loss", "blue") + ": %.4f" % cf_loss
        return train_loss_output + "]"

    def fit(self, data, ori_data):

        cur_eval_step = 0
        self.vq_init()
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss,_ , quant_loss = self._train_epoch(data, epoch_idx)

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss,
            )
            self.logger.info(train_loss_output)

            # ====== 新增：定期验证碰撞率 ======
            avg_collision, text_cr, kg_cr = self._valid_epoch(data)  # 用训练集或传入 valid_data

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint(epoch=epoch_idx,ckpt_file=self.best_loss_ckpt)
                cur_eval_step = 0

                # new
                self.logger.info(set_color("Saving quantized embeddings (separate files) for best model...", "yellow"))
                self.model.eval()
                with torch.no_grad():
                    # 转为 tensor 并移到 GPU
                    text_all = torch.from_numpy(ori_data.text).to(self.device)
                    kg_all = torch.from_numpy(ori_data.kg).to(self.device)

                    # 编码 + 量化
                    text_z = self.model.text_encoder(text_all)
                    kg_z = self.model.kg_encoder(kg_all)

                    text_quantized, _, _ = self.model.text_rq(text_z, self.labels)
                    kg_quantized, _, _ = self.model.kg_rq(kg_z, self.labels_2)

                    assert text_quantized.size(0) == kg_quantized.size(0), "Batch size mismatch between modalities!"


                    fused_quantized = torch.cat([text_quantized, kg_quantized], dim=-1)  # shape: (N, D1 + D2)

                    fused_save_path = os.path.join(self.ckpt_dir, "fused_quantized_best.pt")

                    torch.save(fused_quantized.cpu(), fused_save_path)

                    self.logger.info(set_color(f"Saved: {fused_save_path}", "blue"))

                self.model.train()                

                '''
                #if train_loss
                collision_rate = self._valid_epoch(data)
                if collision_rate < self.best_collision_rate:
                    self.best_loss = train_loss
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt) 
                '''    
            else:
                cur_eval_step += 1

                if cur_eval_step >= self.maxe:
                    ckpt_path = os.path.join(self.ckpt_dir,self.best_loss_ckpt)
                    print(ckpt_path)
                    print("Finish!")
                    break
                '''
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)

                if epoch_idx>2500:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                '''

        return self.best_loss, self.best_collision_rate
        #return self.best_loss



