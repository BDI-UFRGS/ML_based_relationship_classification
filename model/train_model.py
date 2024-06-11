

"""
!pip install tensorflow-text
!pip install tf-models-official
!pip install scikit-learn
!pip install datasets
!pip install nltk
!pip install transformers==4.37.2
!pip install tensorflow==2.15.1
!pip install tf_keras==2.15.1

"""

import tensorflow as tf
print('TensorFlow version:', tf.__version__)
import keras
print('Keras version:', keras.__version__)
import transformers
print('Transformers version:', transformers.__version__)
import tensorflow as tf
#from official.nlp import optimization  # to create AdamW optimizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from scipy.sparse import load_npz
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
nltk.download('wordnet')
import transformers
from transformers import AutoTokenizer


tf.get_logger().setLevel('ERROR')

fold_num = "fold_ex"
# Load he data from each fold
X_train = pd.read_csv(f'{fold_num}_train.csv')
y_train = pd.read_csv(f'{fold_num}_train_labels.csv')
X_test = pd.read_csv(f'{fold_num}_test.csv')
y_test = pd.read_csv(f'{fold_num}_test_labels.csv')


X_train['text'] = X_train
X_test['text'] = X_test

X_train['labels'] = y_train
X_test['labels'] = y_test

X_train = X_train.drop(['def'], axis=1)
X_test = X_test.drop(['def'], axis=1)

np.random.seed(42)

# Select random indices for the training set
train_indices = np.random.choice(X_train.index, size=128, replace=False)

# Select the corresponding rows from X_train and y_train
X_train = X_train.loc[train_indices]

# Select random indices for the test set
test_indices = np.random.choice(X_test.index, size=64, replace=False)

# Select the corresponding rows from X_test and y_test
X_test = X_test.loc[test_indices]

# Verify the shapes of the resulting DataFrames
print(X_train.shape)
print(X_test.shape)

dataset_train = Dataset.from_pandas(X_train)
dataset_test = Dataset.from_pandas(X_test)

# Convert DataFrames to NumPy arrays
y_train = y_train.values.ravel()
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

class_weights

# here choose the model type: binary or multiclass depending on the chosen experiment
model_type = "multiclass"

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=256)


tokenized_train = dataset_train.map(tokenize_function, batched=True)
tokenized_test = dataset_test.map(tokenize_function, batched=True)

if model_type == "multiclass":
    label_mapping = {"Hypernyms": 1, "Holonyms": 0, "Unrelated": 2}
    depth = 3
    def encode_labels(example):
        labels = tf.one_hot(label_mapping[example["labels"]], depth=depth, dtype=tf.int64)  # One-hot encode the labels and cast to int64
        example["labels"] = tf.cast(labels, tf.int64)
        return example
    tokenized_train = tokenized_train.map(encode_labels)
    tokenized_test = tokenized_test.map(encode_labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_train.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_test_dataset = tokenized_test.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig

model_name = "bert-base-uncased"

# Load the tokenizer and configuration
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)

# Load BERT model with the provided configuration
bert_model = TFBertForSequenceClassification.from_pretrained(model_name, config=config)

# Function to tokenize and prepare input data
def tokenize_batch(batch):
    return tokenizer(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], padding=True, truncation=True)

# Create a simple model using BERT
input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
token_type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")

bert_output = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

# Add dropout layer
dropout_rate = 0.1  # You can adjust this value based on your needs
dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate, name="dropout_layer")(bert_output)

if model_type == "multiclass":
    num_classes = 3
    activation = 'softmax'
elif model_type == "binary":
    num_classes = 1
    activation='sigmoid'

dense_layer = tf.keras.layers.Dense(num_classes, activation=activation, name="dense_layer")(dropout_layer)

# Build the model
model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=dense_layer)

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

@tf.function
def f1_macro(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * ((precision_value * recall_value) / (precision_value + recall_value + tf.keras.backend.epsilon()))

# define out weighted loss function
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        # Convert y_true to one-hot encoding
        y_true_one_hot = tf.one_hot(tf.argmax(y_true, axis=-1), depth=tf.shape(y_pred)[-1])

        # Apply weights to the one-hot labels
        weights_per_class = tf.reduce_sum(weights * y_true_one_hot, axis=-1)

        # Compute cross-entropy loss with weighted targets
        unweighted_losses = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weighted_losses = unweighted_losses * weights_per_class
        return tf.reduce_mean(weighted_losses)

    return loss

import tensorflow as tf

def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        # Convert y_true to one-hot encoding
        y_true_one_hot = tf.one_hot(tf.argmax(y_true, axis=-1), depth=tf.shape(y_pred)[-1])

        # Apply weights to the one-hot labels
        weights_per_class = tf.reduce_sum(weights * y_true_one_hot, axis=-1)

        # Compute cross-entropy loss with weighted targets
        unweighted_losses = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weighted_losses = unweighted_losses * weights_per_class
        return tf.reduce_mean(weighted_losses)

    return loss


# Compile the model with weighted cross-entropy loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=weighted_categorical_crossentropy(class_weights),
    metrics=[f1_macro]
)

import tensorflow as tf


if model_type== "multiclass":
    loss_function = weighted_categorical_crossentropy(class_weights)
elif model== "binary":
    loss_function = 'binary_crossentropy'
# Compile the model with weighted cross-entropy loss

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=loss_function,
    metrics=[f1_macro]
)

if model_type== "multiclass":
    model.fit(tf_train_dataset, epochs=3, batch_size= 32)
elif model== "binary":
    model.fit(tf_train_dataset, epochs=3, batch_size= 32, class_weight= class_weights_dict)

model.evaluate(tf_test_dataset)

# prompt: give me a model predict adn compare to the correct labels give confusion matrix

y_pred = model.predict(tf_test_dataset)
y_pred = np.round(y_pred)
y_pred_binary = np.argmax(y_pred, axis=1)

test = tokenized_test["labels"]
test = np.argmax(test, axis=1)

# prompt: generate metrics for my test and y_pred_binary

from sklearn.metrics import classification_report

print(classification_report(test, y_pred_binary))

# prompt: classification report dict

from sklearn.metrics import classification_report

# Assuming you have the true labels in 'test' and predicted labels in 'y_pred_binary'
report = classification_report(test, y_pred_binary, output_dict=True)
report

