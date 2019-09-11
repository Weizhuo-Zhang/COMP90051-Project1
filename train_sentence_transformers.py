from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.readers import NLIDataReader
from sentence_transformers.evaluation import BinaryEmbeddingSimilarityEvaluator, EmbeddingSimilarityEvaluator
from radam import RAdam
import logging
import torch

torch.backends.cudnn.benchmark = True

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

output_path = "checkpoints/sentence_transformers/bert_base_cased_200_v7.0C_epoch2"
num_epochs = 1
train_batch_size = 32
eval_batch_size = 256

# Apply mean pooling to get one fixed sized sentence vector
# word_embedding_model = models.BERT('bert-base-cased', do_lower_case=False)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
#                                pooling_mode_mean_tokens=True,
#                                pooling_mode_cls_token=False,
#                                pooling_mode_max_tokens=False)
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Load Old Model
model = SentenceTransformer('checkpoints/sentence_transformers/bert_base_cased_200_v7.0C_epoch1')


nli_reader = NLIDataReader('data/pair_v7', header=False)

logging.info("Read Dev dataset")
dev_data = SentencesDataset(examples=nli_reader.get_examples('paired_sentences_dev_fixed.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=eval_batch_size)
evaluator = BinaryEmbeddingSimilarityEvaluator(dev_dataloader)


logging.info("Read Train dataset")
train_data = SentencesDataset(examples=nli_reader.get_examples('paired_sentences_train.csv'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.SoftmaxLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=2)
                                # drop_out_rate=0.3)

# Train the model
# 1% of train data
warmup_steps = int(len(train_data)*num_epochs/train_batch_size*0.01)
# warmup_steps = 0
model.evaluate(evaluator)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          optimizer_class=RAdam,
          optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'betas': (0.0, 0.999)},
          evaluation_steps=len(train_data)/(train_batch_size*2)+2,
          warmup_steps=warmup_steps,
          output_path=output_path)
