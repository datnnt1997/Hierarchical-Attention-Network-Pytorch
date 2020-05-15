import os
import json

from torchtext.data import Field, NestedField, LabelField, RawField, Example, Dataset, BucketIterator
from torchtext.vocab import Vocab, FastText
from underthesea import sent_tokenize
# from nltk import sent_tokenize


def load_data(train_file, test_file, pretrain=None, save_dir=None):
    assert os.path.exists(train_file), f"{train_file} is not exist!"
    assert os.path.exists(test_file), f"{test_file} is not exist!"
    print("=" * 30 + "DATASET LOADER" + "=" * 30)
    sent_field = Field(tokenize=lambda x: x.split(), unk_token='<unk>', pad_token='<pad>',
                       init_token=None, eos_token=None)
    doc_field = NestedField(sent_field, tokenize=sent_tokenize, pad_token='<pad>', init_token=None,
                            eos_token=None, include_lengths=True)
    label_field = LabelField()
    fields = [("raw", RawField()), ("doc", doc_field), ("label", label_field)]
    print(f"Reading {train_file} ...")
    with open(train_file, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
        examples = []
        for line in lines:
            text, label = line.split('\t')
            examples.append(Example.fromlist([text, text.lower(), label], fields))
        train_dataset = Dataset(examples, fields)
        reader.close()
    print(f"\tNum of train examples: {len(examples)}")
    print(f"Reading {test_file} ...")
    with open(test_file, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
        examples = []
        for line in lines:
            text, label = line.split('\t')
            examples.append(Example.fromlist([text, text.lower(), label], fields))
        test_dataset = Dataset(examples, fields)
        reader.close()
    print(f"\tNum of valid examples: {len(examples)}")
    vectors = FastText('vi')
    doc_field.build_vocab(train_dataset, test_dataset, vectors=vectors)
    label_field.build_vocab(train_dataset, test_dataset)
    print(f"Building vocabulary ...")
    num_vocab = len(doc_field.vocab)
    num_classes = len(label_field.vocab)
    pad_idx = doc_field.vocab.stoi['<pad>']
    print(f"\tNum of vocabulary: {num_vocab}")
    print(f"\tNum of classes: {num_classes}")
    if save_dir:
        with open(save_dir + "/vocab.json", "w", encoding="utf-8") as fv:
            vocabs = {"word": doc_field.vocab.stoi,
                      "class": label_field.vocab.itos,
                      'pad_idx': pad_idx}
            json.dump(vocabs, fv)
    print("=" * 73)
    return train_dataset, test_dataset, num_vocab, num_classes, pad_idx, vectors.vectors


def build_iterator(dataset, batch_size, device="cuda", is_train=True):
    iterator = BucketIterator(
        dataset,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.doc),
        repeat=False,
        train=is_train,
        sort=True)
    return iterator


if __name__ == "__main__":
    train_dataset, test_dataset, num_vocab, num_classes, pad_idx, vectors = load_data("dataset/sample.train",
                                                                             "dataset/sample.test",
                                                                             "outputs")
    train_iter = build_iterator(train_dataset, 3)
    for b in train_iter:
        print(b.doc)
        print(b.label)