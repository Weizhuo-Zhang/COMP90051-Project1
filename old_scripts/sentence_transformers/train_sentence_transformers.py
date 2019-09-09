from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.readers import TripletReader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

output_path = "checkpoints/sentence_transformers/bert_base_finetune"
num_epochs = 2
train_batch_size = 16

word_embedding_model = models.BERT('bert-base-uncased')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


triplet_reader = TripletReader('../data/triplet_transformer',
                               s1_col_idx=0,
                               s2_col_idx=1,
                               s3_col_idx=2)

logging.info("Read Triplet train dataset")
train_data = SentencesDataset(examples=triplet_reader.get_examples('triple_sentences.csv'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.TripletLoss(model=model)


logging.info("Read Triplet dev dataset")
dev_data = SentencesDataset(examples=triplet_reader.get_examples('dev_random.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = TripletEvaluator(dev_dataloader)


# Train the model
# 10% of train data
warmup_steps = int(len(train_data)*num_epochs/train_batch_size*0.1)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=output_path)
