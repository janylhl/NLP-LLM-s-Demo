from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from evaluation import compute_metrics
from transformers import AdamWeightDecay
from transformers import TFAutoModelForSeq2SeqLM
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback
from datasets import load_dataset

books = load_dataset("opus_books", "en-fr")
# Diviser le dataset en train et test
train_test_split = books['train'].train_test_split(test_size=0.2)

# Accéder aux ensembles d'entraînement et de test
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']


# checkpoint selection
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Languages and mode choice
source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

# Preprocessing the data
def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_books_train = train_dataset.map(preprocess_function, batched=True)
tokenized_books_test = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")

# Optimizer and Model selection
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Dataset preparation
tf_train_set = model.prepare_tf_dataset(
    tokenized_books_train,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    tokenized_books_test,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)


model.compile(optimizer=optimizer) 
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)
model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=[metric_callback])
model.save('./models/translator')