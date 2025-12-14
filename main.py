import argparse
import logging
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from nltk.translate.bleu_score import sentence_bleu

from module.model import BiLSTM, Transformer, CNN, BiLSTMAttn, BiLSTMCNN, BiLSTMConvAttRes
from module import Tokenizer, init_model_by_key
from module.metric import calc_bleu, calc_rouge_l
import csv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=768, type=int)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("-m", "--model", default='seq2seqgru', type=str)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--fp16_opt_level", default='O1', type=str)
    parser.add_argument("--max_grad_norm", default=3.0, type=float)
    parser.add_argument("--dir", default='dataset', type=str)
    parser.add_argument("--output", default='output', type=str)
    parser.add_argument("--logdir", default='runs', type=str)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--n_layer", default=1, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--ff_dim", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)

    parser.add_argument("--test_epoch", default=1, type=int)
    parser.add_argument("--save_epoch", default=10, type=int)

    parser.add_argument("--embed_drop", default=0.2, type=float)
    parser.add_argument("--hidden_drop", default=0.1, type=float)
    return parser.parse_args()

def auto_evaluate(model, testloader, tokenizer):
    bleus = []
    rls = []
    device = next(model.parameters()).device
    gen_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    gen_model.eval()

    comma_id = tokenizer.token_to_ix.get("，", None) if hasattr(tokenizer, "token_to_ix") else None
    period_id = tokenizer.token_to_ix.get("。", None) if hasattr(tokenizer, "token_to_ix") else None
    comma_en_id = tokenizer.token_to_ix.get(",", None) if hasattr(tokenizer, "token_to_ix") else None
    period_en_id = tokenizer.token_to_ix.get(".", None) if hasattr(tokenizer, "token_to_ix") else None

    for batch in testloader:
        batch = tuple(t.to(device) for t in batch)
        src_ids, src_mask, src_len, tgt_in, tgt_out, tgt_mask, tgt_len = batch

        B = src_ids.size(0)
        with torch.no_grad():
            for i in range(B):
                L = int(src_len[i].item())
                if L <= 0:
                    continue

                pred_i = gen_model.generate(
                    src_ids[i:i+1],
                    src_mask[i:i+1],
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    pad_id=tokenizer.pad_id,
                    force_len=L,
                    comma_id=comma_id,
                    period_id=period_id,
                    comma_en_id=comma_en_id,
                    period_en_id=period_en_id,
                )[0].tolist()

                # get content token
                pred_seq = [x for x in pred_i if x not in (tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)]
                pred_seq = pred_seq[:L]

                # reference
                ref_i = tgt_out[i].tolist()
                ref_seq = [x for x in ref_i if x not in (tokenizer.eos_id, tokenizer.pad_id)]
                ref_seq = ref_seq[:L]

                if len(ref_seq) == 0:
                    continue

                bleus.append(calc_bleu(pred_seq, ref_seq))
                rls.append(calc_rouge_l(pred_seq, ref_seq))

    return sum(bleus) / max(1, len(bleus)), sum(rls) / max(1, len(rls))


def predict_demos(model, tokenizer: Tokenizer, max_seq_len: int):
    demos = [
        "马 齿 草 焉 无 马 齿",
        "天 古 天 今 ， 地 中 地 外 ， 古 今 中 外 存 天 地",
        "笑 取 琴 书 温 旧 梦",
        "日 里 千 人 拱 手 划 船 ， 齐 歌 狂 吼 川 江 号 子",
        "我 有 诗 情 堪 纵 酒",
        "我 以 真 诚 溶 冷 血",
        "三 世 业 岐 黄 ， 妙 手 回 春 人 共 赞",
    ]

    device = next(model.parameters()).device
    gen_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    gen_model.eval()

    comma_id = tokenizer.token_to_ix.get("，", None) if hasattr(tokenizer, "token_to_ix") else None
    period_id = tokenizer.token_to_ix.get("。", None) if hasattr(tokenizer, "token_to_ix") else None
    comma_en_id = tokenizer.token_to_ix.get(",", None) if hasattr(tokenizer, "token_to_ix") else None
    period_en_id = tokenizer.token_to_ix.get(".", None) if hasattr(tokenizer, "token_to_ix") else None

    for s in demos:
        tokens = s.split()  
        src_len = min(len(tokens), max_seq_len)
        src_tokens = tokens[:src_len]
        src_ids_list = tokenizer.convert_tokens_to_ids(src_tokens)

        # pad to max_seq_len
        src_ids_list = src_ids_list + [tokenizer.pad_id] * (max_seq_len - src_len)
        src_ids = torch.tensor(src_ids_list, dtype=torch.long, device=device).unsqueeze(0)

        # mask: True=PAD
        src_mask = torch.ones((1, max_seq_len), dtype=torch.bool, device=device)
        src_mask[:, :src_len] = 0

        with torch.no_grad():
            pred = gen_model.generate(
                src_ids,
                src_mask,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
                force_len=src_len,      
                comma_id=comma_id,    
                period_id=period_id,
                comma_en_id=comma_en_id,
                period_en_id=period_en_id,
            )[0].tolist()

        # decode: drop BOS/EOS/PAD and cut to src_len
        out_ids = [x for x in pred if x not in (tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)]
        out_ids = out_ids[:src_len]
        out_text = "".join(tokenizer.convert_ids_to_tokens(out_ids))

        logger.info(f"上联：{s.replace(' ', '')}。 预测的下联：{out_text}")


def save_model(filename, model, args, tokenizer):
    info_dict = {
        'model': model.state_dict(),
        'args': args,
        'tokenzier': tokenizer
    }
    torch.save(info_dict, filename)

def run():
    args = get_args()
    fdir = Path(args.dir)

    # --- output + logging ---
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    tb = SummaryWriter(args.logdir)
    logger.info(args)

    # --- metrics csv ---
    import csv
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "bleu", "rouge_l", "lr", "best_bleu", "best_rouge_l"])

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # --- load tokenizer & dataset ---
    logger.info("loading vocab...")
    tokenizer = Tokenizer.from_pretrained(fdir / "vocab.pkl")

    logger.info("loading dataset...")
    train_dataset = torch.load(fdir / "train.pkl", weights_only=False)
    test_dataset = torch.load(fdir / "test.pkl", weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # --- init model ---
    logger.info("initializing model...")
    model = init_model_by_key(args, tokenizer)
    model.to(device)

    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # fp16 (optional)
    amp = None
    if args.fp16:
        try:
            from apex import amp as _amp
            _amp.register_half_function(torch, "einsum")
            model, optimizer = _amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            amp = _amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    logger.info(f"num gpu: {torch.cuda.device_count()}")

    # scheduler: use average train loss (more sensible than sum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    # --- best checkpoints ---
    best_bleu = -1.0
    best_rouge = -1.0
    best_bleu_path = output_dir / "best_bleu.bin"
    best_rouge_path = output_dir / "best_rouge.bin"

    global_step = 0

    for epoch in range(args.epochs):
        logger.info(f"***** Epoch {epoch + 1}/{args.epochs} *****")
        model.train()
        t1 = time.time()

        accu_loss = 0.0
        n_steps = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            src_ids, src_mask, src_len, tgt_in, tgt_out, tgt_mask, tgt_len = batch

            logits = model(src_ids, src_mask, tgt_in, tgt_mask)  # (B, Ttgt, V)

            loss = loss_function(
                logits.reshape(-1, tokenizer.vocab_size),
                tgt_out.reshape(-1)
            )

            # DataParallel may return per-device loss; be safe
            if torch.cuda.device_count() > 1:
                loss = loss.mean()

            if amp is not None:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            accu_loss += loss.item()
            n_steps += 1

            if step % 100 == 0:
                tb.add_scalar("train/step_loss", loss.item(), global_step)
                logger.info(f"[epoch]: {epoch + 1}, [batch]: {step}, [loss]: {loss.item():.6f}")
            global_step += 1

        avg_train_loss = accu_loss / max(1, n_steps)
        lr_now = optimizer.param_groups[0]["lr"]

        # scheduler uses average loss (not sum)
        scheduler.step(avg_train_loss)

        t2 = time.time()
        logger.info(f"epoch time: {t2 - t1:.3f}s, avg train loss: {avg_train_loss:.6f}, lr: {lr_now:.6g}")

        # tensorboard epoch logs
        tb.add_scalar("train/epoch_loss", avg_train_loss, epoch + 1)
        tb.add_scalar("train/lr", lr_now, epoch + 1)

        bleu, rl = None, None

        # --- evaluation ---
        if (epoch + 1) % args.test_epoch == 0:
            predict_demos(model, tokenizer, args.max_seq_len)
            bleu, rl = auto_evaluate(model, test_loader, tokenizer)
            logger.info(f"BLEU: {bleu:.9f}, Rouge-L: {rl:.8f}")

            tb.add_scalar("eval/bleu", bleu, epoch + 1)
            tb.add_scalar("eval/rouge_l", rl, epoch + 1)

            # save best by BLEU
            if bleu > best_bleu:
                best_bleu = bleu
                save_model(best_bleu_path, model, args, tokenizer)
                logger.info(f"Saved best BLEU checkpoint to {best_bleu_path} (BLEU={best_bleu:.6f})")

            # save best by Rouge-L
            if rl > best_rouge:
                best_rouge = rl
                save_model(best_rouge_path, model, args, tokenizer)
                logger.info(f"Saved best Rouge-L checkpoint to {best_rouge_path} (Rouge-L={best_rouge:.6f})")

        # --- periodic save ---
        if (epoch + 1) % args.save_epoch == 0:
            # DataParallel class name would be DataParallel; better use underlying module name if possible
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            filename = f"{base_model.__class__.__name__}_{epoch + 1}.bin"
            filename = output_dir / filename
            save_model(filename, model, args, tokenizer)
            logger.info(f"Saved checkpoint to {filename}")

        # --- append metrics.csv (write eval numbers if available) ---
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch + 1,
                avg_train_loss,
                "" if bleu is None else float(bleu),
                "" if rl is None else float(rl),
                lr_now,
                best_bleu,
                best_rouge
            ])

    tb.close()

if __name__ == "__main__":
    run()