import argparse
import torch
from module import init_model_by_key

def _build_src_mask(input_ids: torch.Tensor, pad_id: int):
    # input_ids: (B, T)
    # mask: True=PAD, False=valid
    return input_ids.eq(pad_id)

def _get_punct_ids(tokenizer):
    def tid(tok: str):
        return tokenizer.token_to_ix.get(tok, None)

    return {
        "comma_id": tid("，"),
        "period_id": tid("。"),
        "comma_en_id": tid(","),
        "period_en_id": tid("."),
    }

@torch.no_grad()
def predict_one(model, tokenizer, device, text: str, max_seq_len: int):
    model.eval()

    src = tokenizer.encode(text)
    # cut tomax_seq_len
    src = src[:max_seq_len]
    src_ids = torch.tensor(src, dtype=torch.long, device=device).unsqueeze(0)  # (1, Tsrc)

    # src_mask: True=PAD
    src_mask = torch.zeros_like(src_ids, dtype=torch.bool, device=device)
    punct = _get_punct_ids(tokenizer)
    force_len = src_ids.shape[1]

    #generation
    out_ids = model.generate(
        src_ids, src_mask,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        force_len=force_len,
        **punct
    )[0].tolist()

    ignore = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}
    out_ids = [i for i in out_ids if i not in ignore]

    return tokenizer.decode(out_ids)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, type=str, help="path to saved checkpoint .bin")
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("-s", "--stop_flag", default='q', type=str)
    parser.add_argument("-c", "--cuda", action='store_true')
    args = parser.parse_args()

    print("loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    model_info = torch.load(args.path, map_location=device, weights_only=False)

    tokenizer = model_info["tokenzier"]
    model = init_model_by_key(model_info["args"], tokenizer)
    model.load_state_dict(model_info["model"])
    model.to(device)
    model.eval()

    print("Ready. Type 'q' to quit.")
    while True:
        question = input("上联：").strip()
        if question.lower() == args.stop_flag.lower():
            print("Thank you!")
            break

        pred = predict_one(model, tokenizer, device, question, args.max_seq_len)
        print(f"下联：{pred}")

if __name__ == "__main__":
    run()
