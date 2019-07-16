#!/usr/bin/env python
# coding: utf-8

# # TF Ranking
# 
# In this Notebook, we run through a simplified example to highlight some of the features of the TF Ranking library and demonstrate an end-to-end execution.
# 
# The general recipe is a short list of four main steps:
# 
# 1.   Compose a function to **read** input data and prepare a Tensorflow Dataset;
# 2.   Define a **scoring** function that, given a (set of) query-document feature vector(s), produces a score indicating the query's level of relevance to the document;
# 3.   Create a **loss** function that measures how far off the produced scores from step (2) are from the ground truth; and,
# 4.   Define evaluation **metrics**.
# 
# A final step makes use of standard Tensorflow API to create, train, and evaluate a model.
# 
# We have included in the TF Ranking library a default implementation of data readers (in the `tensorflow_ranking.data` module), loss functions (in `tensorflow_ranking.losses`), and popular evaluation metrics (in `tensorflow_ranking.metrics`) that may be further tailored to your needs as we shall show later in this Notebook.
# 
# ### Preparation
# 
# In what follows, we will assume the existence of a dataset that is split into training and test sets and that are stored at `data/train.txt` and `data/test.txt` respectively. We further assume that the dataset is in the LibSVM format and lines in the training and test files are sorted by query ID -- an assumption that holds for many popular learning-to-rank benchmark datasets.
# 
# We have included in our release a toy (randomly generated) dataset in the `data/` directory. However, to learn a more interesting model, you may copy your dataset of choice to the `data/` directory. Please ensure the format of your dataset conforms to the requirements above. Alternatively, you may edit this Notebook to plug in a customized input pipeline for a non-comformant dataset.

# # Get Started with TF Ranking

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_libsvm.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_libsvm.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>

# ### Dependencies and Global Variables
# 
# Let us start by importing libraries that will be used throughout this Notebook. We also enable the "eager execution" mode for convenience and demonstration purposes.

# In[1]:


get_ipython().system(' pip install tensorflow_ranking')


# In[1]:


import tensorflow as tf
import tensorflow_ranking as tfr
from gensim.models import KeyedVectors

tf.enable_eager_execution()
tf.executing_eagerly()


# In[3]:


ostop_words = ["!!", "?!", "??", "!?", "`", "``", "''", "-lrb-", "-rrb-", "-lsb-", "-rsb-", ",", ".", ":", ";", "\"",
                  "'", "?", "<", ">", "{", "}", "[", "]", "+", "-", "(", ")", "&", "%", "$", "@", "!", "^", "#", "*",
                  "..", "...", "'ll", "'s", "'m", "a", "about", "above", "after", "again", "against", "all", "am", "an",
                  "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
                  "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do",
                  "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
                  "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
                  "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
                  "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
                  "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
                  "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
                  "she'll", "she's", ",should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
                  "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
                  "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
                  "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
                  "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
                  "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
                  "yours", "yourself", "yourselves", "###", "return", "arent", "cant", "couldnt", "didnt", "doesnt",
                  "dont", "hadnt", "hasnt", "havent", "hes", "heres", "hows", "im", "isnt", "its", "lets", "mustnt",
                  "shant", "shes", "shouldnt", "thats", "theres", "theyll", "theyre", "theyve", "wasnt", "were",
                  "werent", "whats", "whens", "wheres", "whos", "whys", "wont", "wouldnt", "youd", "youll", "youre",
                  "youve"]


from stop_words import get_stop_words
istop_words = get_stop_words('en')

stop_words = list(set(ostop_words).union(set(istop_words)))

print("O : ",len(ostop_words),"\t I : ", len(istop_words),"\t All : ",len(stop_words))


# In[6]:


model = KeyedVectors.load_word2vec_format('numberbatch-en-17.04b.txt', binary=False)
vec = model['person']
print("Dimension : ",len(vec))


# ## Query the Data from the Embeddings and Create Dataset

# In[4]:


def remove_stops(string):
    return " ".join([word for word in string if word not in stop_words])


# In[ ]:


import sqlite3 as lite
# db_file = "/home/mulang/Desktop/Learning/EMNLP-19/ExCon_TG19/qae_sqlite/qae_lite/qae.sqlite"
db_file = "../../../qae_sqlite/qae.sqlite"
select_dev_query = "SELECT edq.q_id, edq.e_id, dq.q_string,dq.answer,dq.other_choices, e.explanation FROM Dev_Q_E_Join                     AS edq JOIN dev_question AS dq ON edq.q_id=dq.q_id, explanation AS e ON edq.e_id=e.e_id                     ORDER BY edq.q_id ASC;"

con = lite.connect(db_file)
curr_list = []

with con:
    cur = con.cursor()

    cur.execute(select_dev_query)
    data = cur.fetchall()

    for d in data:
        print(d)
        curr_list.append([remove_stops(val) for val in d])
con.close()

for ror in curr_list[:5]:
    print(row)


# Next, we will download a dummy dataset in LibSVM format.Note that you can replace these datasets with public or custom datasets. 
# 
# We also define some global parameters.

# In[5]:


get_ipython().system(' wget -O "/home/mulang/train.txt" "https://raw.githubusercontent.com/tensorflow/ranking/master/tensorflow_ranking/examples/data/train.txt"')
get_ipython().system(' wget -O "/home/mulang/test.txt" "https://raw.githubusercontent.com/tensorflow/ranking/master/tensorflow_ranking/examples/data/test.txt"')


# In[3]:


# Store the paths to files containing training and test instances.
# As noted above, we will assume the data is in the LibSVM format
# and that the content of each file is sorted by query ID.

# _TRAIN_DATA_PATH="data/train.txt"
# _TEST_DATA_PATH="data/test.txt"

_TRAIN_DATA_PATH="data/train-500-lemma-ts-affix-concepts-openie-conceptrel-wikicategories-wikititles-50-framenet.dat"
_TEST_DATA_PATH="data/dev-lemma-ts-affix-concepts-openie-conceptrel-wikicategories-wikititles-50-framenet.dat"

# Define a loss function. To find a complete list of available
# loss functions or to learn how to add your own custom function
# please refer to the tensorflow_ranking.losses module.
# _LOSS="pairwise_logistic_loss"
# _LOSS="list_mle_loss"
_LOSS = "sigmoid_cross_entropy_loss"

# In the TF-Ranking framework, a training instance is represented
# by a Tensor that contains features from a list of documents
# associated with a single query. For simplicity, we fix the shape
# of these Tensors to a maximum list size and call it "list_size,"
# the maximum number of documents per query in the dataset.
# In this demo, we take the following approach:
#   * If a query has fewer documents, its Tensor will be padded
#     appropriately.
#   * If a query has more documents, we shuffle its list of
#     documents and trim the list down to the prescribed list_size.
_LIST_SIZE=515

# The total number of features per query-document pair.
# We set this number to the number of features in the MSLR-Web30K
# dataset.
_NUM_FEATURES=136

# Parameters to the scoring function.
_BATCH_SIZE=32
_HIDDEN_LAYER_DIMS=["20", "10"]
_NUMBERBATCH_DIR = "../embeddings" 
# OUTPUT_DIR


# ### Input Pipeline
# 
# The first step to construct an input pipeline that reads your dataset and produces a `tensorflow.data.Dataset` object. In this example, we will invoke a LibSVM parser that is included in the `tensorflow_ranking.data` module to generate a `Dataset` from a given file.
# 
# We parameterize this function by a `path` argument so that the function can be used to read both training and test data files.

# In[4]:


def input_fn(path):
  train_dataset = tf.data.Dataset.from_generator(
      tfr.data.libsvm_generator(path, _NUM_FEATURES, _LIST_SIZE),
      output_types=(
          {str(k): tf.float32 for k in range(1,_NUM_FEATURES+1)},
          tf.float32
      ),
      output_shapes=(
          {str(k): tf.TensorShape([_LIST_SIZE, 1])
            for k in range(1,_NUM_FEATURES+1)},
          tf.TensorShape([_LIST_SIZE])
      )
  )

  train_dataset = train_dataset.shuffle(1000).repeat().batch(_BATCH_SIZE)
  return train_dataset.make_one_shot_iterator().get_next()


# ### Scoring Function
# 
# Next, we turn to the scoring function which is arguably at the heart of a TF Ranking model. The idea is to compute a relevance score for a (set of) query-document pair(s). The TF-Ranking model will use training data to learn this function.
# 
# Here we formulate a scoring function using a feed forward network. The function takes the features of a single example (i.e., query-document pair) and produces a relevance score.

# In[5]:


def example_feature_columns():
  """Returns the example feature columns."""
  feature_names = [
      "%d" % (i + 1) for i in range(0, _NUM_FEATURES)
  ]
  return {
      name: tf.feature_column.numeric_column(
          name, shape=(1,), default_value=0.0) for name in feature_names
  }

def make_score_fn():
  """Returns a scoring function to build `EstimatorSpec`."""

  def _score_fn(context_features, group_features, mode, params, config):
    """Defines the network to score a documents."""
    del params
    del config
    # Define input layer.
    example_input = [
        tf.layers.flatten(group_features[name])
        for name in sorted(example_feature_columns())
    ]
    input_layer = tf.concat(example_input, 1)

    cur_layer = input_layer
    for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
      cur_layer = tf.layers.dense(
          cur_layer,
          units=layer_width,
          activation="tanh")

    logits = tf.layers.dense(cur_layer, units=1)
    return logits

  return _score_fn


# ### Evaluation Metrics
# 
# We have provided an implementation of popular Information Retrieval evalution metrics in the TF Ranking library.

# In[6]:


def eval_metric_fns():
  """Returns a dict from name to metric functions.

  This can be customized as follows. Care must be taken when handling padded
  lists.

  def _auc(labels, predictions, features):
    is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
    clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
    clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
    return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
  metric_fns["auc"] = _auc

  Returns:
    A dict mapping from metric name to a metric function with above signature.
  """
  metric_fns = {}
  metric_fns.update({
      "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.NDCG, topn=topn)
      for topn in [1, 3, 5, 10]
  })

  return metric_fns


# ### Putting It All Together
# 
# We are now ready to put all of the components above together and create an `Estimator` that can be used to train and evaluate a model.

# In[7]:


def get_estimator(hparams):
  """Create a ranking estimator.

  Args:
    hparams: (tf.contrib.training.HParams) a hyperparameters object.

  Returns:
    tf.learn `Estimator`.
  """
  def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=hparams.learning_rate,
        optimizer="Adagrad")

  ranking_head = tfr.head.create_ranking_head(
      loss_fn=tfr.losses.make_loss_fn(_LOSS),
      eval_metric_fns=eval_metric_fns(),
      train_op_fn=_train_op_fn)

  return tf.estimator.Estimator(
      model_fn=tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          group_size=1,
          transform_fn=None,
          ranking_head=ranking_head),
      params=hparams)


# Let us instantiate and initialize the `Estimator` we defined above.

# In[8]:


hparams = tf.contrib.training.HParams(learning_rate=0.05)
ranker = get_estimator(hparams)


# Now that we have a correctly initialized `Estimator`, we will train a model using the training data. We encourage you to experiment with different number of steps here and below.

# In[9]:


ranker.train(input_fn=lambda: input_fn(_TRAIN_DATA_PATH), steps=100)


# Finally, let us evaluate our model on the test set.

# In[ ]:


ranker.evaluate(input_fn=lambda: input_fn(_TEST_DATA_PATH), steps=100)


# ### Visualization
# 
# The train and evaluation steps above by default store checkpoints, metrics, and other useful information about your network to a temporary directory on disk. We encourage you to visualize this data using [Tensorboard](http://www.tensorflow.org/guide/summaries_and_tensorboard). In particular, you can launch Tensorboard and point it to where your model data is stored as follows:
# 
# First, let's find out the path to the log directory created by the process above.

# In[ ]:


ranker.model_dir


# Launch Tensorboard in shell using:
# 
# $ tensorboard --logdir=<ranker.model_dir output>

# In[ ]:




