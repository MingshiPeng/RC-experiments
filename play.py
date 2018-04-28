import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell

context_bt = tf.placeholder(shape=(32, 300), dtype=tf.int32, name='context_bt')
embedding = tf.get_variable(shape=(5000, 200), dtype=tf.float32, name='embedding_matrix')

context_embed_btf = tf.nn.embedding_lookup(embedding, context_bt)

# lstm_cell = MultiRNNCell([LSTMCell(128) for _ in range(3)])
# outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell,
# 										 dtype=tf.float32,
# 										 inputs=context_embed_btf)


q_cell_fw = MultiRNNCell(cells=[LSTMCell(128) for _ in range(1)])
q_cell_bw = MultiRNNCell(cells=[LSTMCell(128) for _ in range(1)])
outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=q_cell_bw,
                                                       cell_fw=q_cell_fw,
                                                       dtype="float32",
                                                       inputs=context_embed_btf,
                                                       swap_memory=True)

q_encoded_bf = tf.concat([last_states[0][-1], last_states[1][-1]], axis=-1)

