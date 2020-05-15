import os
import torch
import argparse

from models.hier_attn import HAN
from dataloader import load_data, build_iterator
from sklearn import metrics
from tqdm import tqdm


def save_checkpoint(save_dir, model, epoch, loss, metric, f1_score):
    file_model = os.path.join(save_dir, "model.model")
    file_infor = os.path.join(save_dir, "summary.txt")
    print("saving %s" % file_model)
    torch.save(model.state_dict(), file_model)
    f = open(file_infor, 'w', encoding="utf-8")
    f.write("File model: {}\n".format(file_model))
    f.write("Epoch: {}\n".format(epoch))
    f.write("Loss: {}\n".format(loss))
    f.write("Evaluation: \n")
    f.write("F1 score: {}\n".format(f1_score))
    f.write(metric)
    f.close()
    print("saved model at epoch %d" % epoch)


def main(opts):
    if opts.device:
        device = opts.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(opts.saved_dir):
        os.makedirs(opts.saved_dir)
    if not os.path.exists(opts.log_path):
        os.makedirs(opts.log_path)

    train_dataset, valid_dataset, num_vocab, num_classes, pad_idx, vectors = load_data(train_file=opts.train_file,
                                                                                       test_file=opts.valid_file,
                                                                                       save_dir=opts.saved_dir)
    model = HAN(word_hidden_size=opts.word_hidden_size,
                sent_hidden_size=opts.sent_hidden_size,
                word_attn_size=opts.word_attn_size,
                sent_attn_size=opts.sent_attn_size,
                num_vocab=num_vocab,
                num_classes=num_classes,
                embedd_dim=opts.word_vec_size,
                pad_idx=pad_idx,
                init_weight=opts.init_weight, device=device,
                vectors=vectors)

    print("=" * 30 + "MODEL SUMMARY" + "=" * 30)
    print(model)
    print("=" * 73)

    if device == 'cuda':
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    if opts.optim == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opts.lr)
    elif opts.optim == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=opts.lr, momentum=opts.momentum)
    best_score = float('-inf')
    train_iter = build_iterator(train_dataset, batch_size=opts.batch_size, device=device, is_train=True)
    valid_iter = build_iterator(valid_dataset, batch_size=opts.batch_size, device=device, is_train=False)
    for epoch in range(opts.num_epoches):
        print(f"Epoch: {epoch}/{opts.num_epoches}")
        train_iter.init_epoch()
        total_epoch_loss = 0
        predicts = []
        actuals = []
        model.train()
        tqdm_bar = tqdm(enumerate(train_iter), total=len(train_iter), desc="Train")
        for idx, batch in tqdm_bar:
            optimizer.zero_grad()
            docs, doc_lens, sent_lens = batch.doc
            labels = batch.label
            prods, logits = model(docs, doc_lens, sent_lens)
            loss = criterion(prods, labels)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            predicts += [y.argmax().item() for y in prods]
            actuals += labels.tolist()

        acc_score = metrics.accuracy_score(actuals, predicts)
        macro_f1_score = metrics.f1_score(actuals, predicts, average="macro")
        print("TRAIN Macro F1 score: " + str(macro_f1_score))
        print("TRAIN Accurancy score: " + str(acc_score))
        print("TRAIN LOSS: {}".format(total_epoch_loss / len(train_iter)))

        if epoch % opts.valid_interval == 0:
            train_iter.init_epoch()
            total_epoch_loss = 0
            predicts = []
            actuals = []
            model.train()
            tqdm_bar = tqdm(enumerate(valid_iter), total=len(valid_iter), desc="Valid")
            for idx, batch in tqdm_bar:
                optimizer.zero_grad()
                docs, doc_lens, sent_lens = batch.doc
                labels = batch.label
                prods, logits = model(docs, doc_lens, sent_lens)
                loss = criterion(prods, labels)

                total_epoch_loss += loss.item()
                predicts += [y.argmax().item() for y in prods]
                actuals += labels.tolist()
        metric = metrics.classification_report(actuals, predicts)
        conf_matrix = metrics.confusion_matrix(actuals, predicts)
        acc_score = metrics.accuracy_score(actuals, predicts)
        macro_f1_score = metrics.f1_score(actuals, predicts, average="macro")
        print(metric)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("VALID Macro F1 score: " + str(macro_f1_score))
        print("VALID Accurancy score: " + str(acc_score))
        print("VALID LOSS: {}".format(total_epoch_loss / len(train_iter)))

        if macro_f1_score > best_score:
            save_checkpoint(opts.saved_dir, model, epoch, loss, metric, macro_f1_score)
            best_score = macro_f1_score
    return best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train.tsv", help="Path to the training file.")
    parser.add_argument("--valid_file", type=str, default="data/test.tsv", help="Path to the validation file.")
    parser.add_argument("--pretrain_embedding_file", type=str, default="data/word2vec.300d.txt",
                        help="Pre-train embeddings file.")
    parser.add_argument("--lang", type=str, default="vi", choices=['vi', 'en'],
                        help="Language used for Pre-train embeddings model.")

    parser.add_argument("--word_vec_size", type=int, default=300, help="Word embedding size.")
    parser.add_argument("--word_hidden_size", type=int, default=100, help="Size of word RNN hidden states.")
    parser.add_argument("--sent_hidden_size", type=int, default=100, help="Size of sentence RNN hidden states.")
    parser.add_argument("--word_attn_size", type=int, default=100, help="Size of word attention.")
    parser.add_argument("--sent_attn_size", type=int, default=100, help="Size of word attention.")

    parser.add_argument("--batch_size", type=int, default=16, help="Maximum batch size for training.")
    parser.add_argument("--num_epoches", type=int, default=100, help="Number of training epoches.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for training.")
    parser.add_argument("--optim", type=str, default='sgd', choices=['sgd', 'adam'], help="Optimization method.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help="Parameters are initialized over uniform distribution with support "
                             "(-init_weight, init_weight)")
    parser.add_argument("--valid_interval", type=int, default=1, help="Number of epoches between testing phases.")

    parser.add_argument("--log_path", type=str, default="outputs/logs",
                        help="Output logs to a file under this path.")
    parser.add_argument("--saved_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda', None],
                        help="Device for training phases.")
    args = parser.parse_args()
    main(args)
