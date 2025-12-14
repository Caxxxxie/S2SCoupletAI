import argparse
import torch
from flask import Flask, request, render_template
from module import init_model_by_key

def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}

def _tokenize_like_training(s: str):
    s = (s or "").strip()
    if not s:
        return []
    # if user provides whitespace-separated tokens, respect it
    if " " in s:
        return [t for t in s.split() if t]
    # otherwise char-level tokens
    return [ch for ch in s if not ch.isspace()]

class Context(object):
    def __init__(self, path: str, device: str = "cpu"):
        print(f"loading pretrained model from {path}")
        self.device = torch.device(device)
        model_info = torch.load(path, map_location=self.device, weights_only=False)

        self.tokenizer = model_info["tokenzier"]
        args = model_info["args"]

        self.model = init_model_by_key(args, self.tokenizer)
        state = _strip_module_prefix(model_info["model"])
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # optional punctuation ids for alignment constraints
        self.comma_id = self.tokenizer.token_to_ix.get("，", None)
        self.period_id = self.tokenizer.token_to_ix.get("。", None)
        self.comma_en_id = self.tokenizer.token_to_ix.get(",", None)
        self.period_en_id = self.tokenizer.token_to_ix.get(".", None)

    @torch.no_grad()
    def predict(self, s: str):
        tokens = _tokenize_like_training(s)
        if not tokens:
            return ""

        force_len = len(tokens)
        src_ids_list = self.tokenizer.convert_tokens_to_ids(tokens)
        src_ids = torch.tensor(src_ids_list, dtype=torch.long, device=self.device).unsqueeze(0)

        # (B, Tsrc) True=PAD; here no PAD in inference => all False
        src_mask = torch.zeros((1, src_ids.size(1)), dtype=torch.bool, device=self.device)

        pred_ids = self.model.generate(
            src_ids,
            src_mask,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            pad_id=self.tokenizer.pad_id,
            force_len=force_len,
            comma_id=self.comma_id,
            period_id=self.period_id,
            comma_en_id=self.comma_en_id,
            period_en_id=self.period_en_id,
        )[0].tolist()

        ignore = {self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id}
        out_ids = [x for x in pred_ids if x not in ignore][:force_len]
        out_tokens = self.tokenizer.convert_ids_to_tokens(out_ids, ignore_pad=False)
        return "".join(out_tokens)

def create_app(ctx: Context):
    app = Flask(__name__)

    # API: allow slashes and unicode by using <path:...>
    @app.route('/api/<path:coupletup>')
    def api(coupletup):
        return ctx.predict(coupletup)

    @app.route('/', methods=['GET', 'POST'])
    def index():
        # echo input and output on the page
        coupletup = ""
        coupletdown = ""
        if request.method == 'POST':
            coupletup = request.form.get("coupletup", "").strip()
            coupletdown = ctx.predict(coupletup) if coupletup else ""
        return render_template(
            "index.html",
            coupletup=coupletup,
            coupletdown=coupletdown,
        )

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    ctx = Context(args.model_path, device=device)
    app = create_app(ctx)
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
