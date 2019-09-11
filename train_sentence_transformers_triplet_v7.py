from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.readers import TripletReader
from sentence_transformers.evaluation import TripletEvaluator
from radam import RAdam
import logging
import torch

torch.backends.cudnn.benchmark = True

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

output_path = "checkpoints/sentence_transformers/roberta-base_v7_triplet_full_epoch1"
num_epochs = 1
train_batch_size = 16
eval_batch_size = 256

# Apply mean pooling to get one fixed sized sentence vector
word_embedding_model = models.RoBERTa('roberta-base', do_lower_case=False)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Load Old Model
# model = SentenceTransformer('checkpoints/sentence_transformers/roberta-base_v7_triplet_epoch1')


triplet_reader = TripletReader('data/v7')

logging.info("Read Train dataset")
train_data = SentencesDataset(examples=triplet_reader.get_examples('triplet_train_full.csv'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.TripletLoss(model=model, triplet_margin=1.0)

logging.info("Read Dev dataset")
dev_data = SentencesDataset(examples=triplet_reader.get_examples('triplet_dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=eval_batch_size)
evaluator = TripletEvaluator(dev_dataloader)

# Train the model
# 1% of train data
warmup_steps = int(len(train_data)*num_epochs/train_batch_size*0.05)
warmup_steps = 0
model.evaluate(evaluator)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          optimizer_class=RAdam,
          optimizer_params={'lr': 1e-5, 'eps': 1e-6, 'betas': (0.0, 0.999)},
          # evaluation_steps=len(train_data)/(train_batch_size*4)+2,
          warmup_steps=warmup_steps,
          output_path=output_path)
