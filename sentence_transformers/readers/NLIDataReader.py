from . import InputExample
import csv
import gzip
import os


class NLIDataReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self, dataset_folder, delimiter="\t", quoting=csv.QUOTE_NONE, header=False):
        self.dataset_folder = dataset_folder
        self.delimiter = delimiter
        self.quoting = quoting
        self.header = header

    def get_examples(self, filename, max_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """
        if self.header:
            data = csv.reader((x.replace('\0', '') for x in (open(os.path.join(self.dataset_folder, filename), encoding="utf-8"))),
                              delimiter=self.delimiter,
                              quoting=self.quoting)
            next(data)
        else:
            data = csv.reader((x.replace('\0', '') for x in (open(os.path.join(self.dataset_folder, filename), encoding="utf-8"))),
                              delimiter=self.delimiter,
                              quoting=self.quoting)

        examples = []
        id = 0
        for id, row in enumerate(data):
            s1 = row[0]
            s2 = row[1]
            label = int(row[2])
            examples.append(InputExample(guid=filename+str(id), texts=[s1, s2], label=label))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]
