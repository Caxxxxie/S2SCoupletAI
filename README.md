# S2SCoupletAI

This project reformulates Chinese couplet generation from **parallel sequence labeling** to a **Seq2Seq (Encoder–Decoder) framework with attention**, using RNN-based models (LSTM / GRU) with checkpoints saved in respective folders.

The project is based on the original implementation:  
https://github.com/liuslnlp/CoupletAI

## Structure

- `module/` : tokenizer, models (Seq2Seq LSTM / GRU), metrics
- `preprocess.py` : data preprocessing
- `main.py` : training and evaluation
- `webdemo.py` : simple Flask web demo
- `clidemo.py` : command-line interactive demo
- `dataset/` : processed datasets
- `output/` : model checkpoints and logs

## Training

Preprocess data:

```bash
python preprocess.py --input couplet --output dataset
```

Train the Seq2Seq–LSTM model:
```bash
python main.py -m seq2seq-rnn --dir dataset --output Seq2SeqLSTM
```
Train the Seq2Seq–GRU variant:
```bash
python main.py -m seq2seqgru --dir dataset --output Seq2SeqLSTM
```
## Web Demo

```bash
python webdemo.py --model_path Seq2SeqLSTM/best_bleu.bin
```
Then open the browser and visit:
```code
http://localhost:5000
```
## CLI Demo
Run an interactive command-line demo:
```bash
python clidemo.py -p Seq2SeqLSTM/best_bleu.bin
```

Type an upper couplet after the prompt, and the model will generate the corresponding lower couplet.
Enter q to exit the program.

## Notes
- Generation enforces equal length and punctuation alignment between the upper and lower couplets.
- BLEU and ROUGE-L are reported for reference, but human inspection is recommended for evaluating generation quality.

