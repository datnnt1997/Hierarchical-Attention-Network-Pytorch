import os

from torchtext.data import Field, NestedField, LabelField, RawField, Example, Dataset, Iterator, BucketIterator
from underthesea import sent_tokenize


def load_data(train_file_path, test_file_path):
    assert os.path.exists(train_file_path), f"{train_file_path} is not exist!"
    assert os.path.exists(test_file_path), f"{test_file_path} is not exist!"

    sent_field = Field(tokenize=lambda x: x.split(), unk_token='<unk>', pad_token='<pad>',
                       init_token=None, eos_token=None)
    doc_field = NestedField(sent_field, tokenize=sent_tokenize, pad_token='<pad>', init_token=None,
                            eos_token=None, include_lengths=True)
    label_file = LabelField()
    fields = [("raw", RawField()), ("doc", doc_field), ("label", label_file)]

    with open(train_file_path, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
        examples = []
        for line in lines:
            text, label = line.split('\t')
            examples.append(Example.fromlist([text, text, label], fields))
        train_dataset = Dataset(examples, fields)
        reader.close()

    with open(test_file_path, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
        examples = []
        for line in lines:
            text, label = line.split('\t')
            examples.append(Example.fromlist([text, text, label], fields))
        test_dataset = Dataset(examples, fields)
        reader.close()

    doc_field.build_vocab(train_dataset, test_dataset, min_freq=1)
    label_file.build_vocab(train_dataset, train_dataset, min_freq=1)

    return train_dataset, test_dataset


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
    train_dataset, test_dataset = load_data("dataset/sample.train", "dataset/sample.test")
    train_iter = build_iterator(train_dataset, 3)
    for b in train_iter:
        print(b.doc)
        print(b.label)