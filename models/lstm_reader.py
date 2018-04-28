import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, GRUCell

from models.rc_base import RcBase
from utils.log import logger

_EPSILON = 10e-8


class LSTMReader(RcBase):
    """
    Simple baseline lstm reader
    """

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        #########################
        # b ... position of the example within the batch
        # t ... position of the word within the document/question
        # f ... features of the embedding vector or the encoded feature vector
        # i ... position of the word in candidates list
        #########################
        num_layers = self.args.num_layers
        hidden_size = self.args.hidden_size
        embedding_dim = self.args.embedding_dim
        assert embedding_dim == hidden_size * num_layers

        cell = LSTMCell if self.args.use_lstm else GRUCell

        # model input
        ques_doc_bt = tf.placeholder(dtype=tf.int32, shape=(None, self.qd_len), name="ques_doc_bt") 
        candidates_bi = tf.placeholder(dtype=tf.int32, shape=(None, self.dataset.A_len), name="candidates_bi")
        y_true_bi = tf.placeholder(shape=(None, self.dataset.A_len), dtype=tf.float32, name="y_true_bi")

        # shape=(None) the length of inputs
        ques_doc_lengths = tf.reduce_sum(tf.sign(tf.abs(ques_doc_bt)), 1)
        ques_doc_mask_bt = tf.sequence_mask(ques_doc_lengths, self.qd_len, dtype=tf.float32)

        init_embedding = tf.constant(self.embedding_matrix, dtype=tf.float32, name="embedding_init")
        embedding = tf.get_variable(initializer=init_embedding,
                                    name="embedding_matrix",
                                    dtype=tf.float32)

        with tf.variable_scope('qd_encoder', initializer=tf.orthogonal_initializer()):
            # encode question_document to fixed length of vector
            # output shape: (None, max_qd_length, embedding_dim)
            ques_doc_embed_btf = tf.nn.embedding_lookup(embedding, ques_doc_bt)
            logger("qd_embed_btf shape {}".format(ques_doc_embed_btf.get_shape()))
            lstm_cell = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
            outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                     dtype="float32",
                                                     sequence_length=ques_doc_lengths,
                                                     inputs=ques_doc_embed_btf,
                                                     swap_memory=True)
            # q_encoder output shape: (None, hidden_size * 2)
            qd_encoder_bf = tf.concat(last_states, axis=-1)
            logger("qd_encoded_bf shape {}".format(qd_encoded_bf.get_shape()))

        with tf.variable_scope('c_encoder', initializer=tf.orthogonal_initializer()):
            # encode each candidate to fixed length vector
            # output shape: (None, max_A_length, embedding_dim)
            c_embed_btf = tf.nn.embedding_lookup(embedding, candidates_bi)
            logger("c_embed_btf shape {}".format(c_embed_btf.get_shape()))

            # a fully connected layer to transform the shape 
            # from (None, max_A_length, embedding_dim)
            # to   (None, max_A_length, hidden_size * 2)
            # w1 = vs.get_variable('w1', [self.args.embedding_dim, hidden_size * num_layers], dtype=tf.float32)
            # b1 = vs.get_variable('b1', [1, hidden_size * num_layers], dtype=tf.loat32)

            # # d_encoder output shape: (None, max_d_length, hidden_size * 2)
            # c_encoded_btf = tf.nn.tanh(tf.matmul(c_embed_btf, w1) + b1)
            # logger("c_encoded_btf shape {}".format(c_encoded_btf.get_shape()))

        # output shape: (None, i) i = max_candidate_length = 10
        y_hat = tf.matmul(c_embed_btf, tf.expand_dims(qd_encoder_bf, -1))

        # crossentropy
        output = y_hat / tf.reduce_sum(y_hat, axis=-1, keep_dims=True)
        # manual computation of crossentropy
        epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype, name="epsilon")
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        self.loss = tf.reduce_mean(- tf.reduce_sum(y_true_bi * tf.log(output), axis=-1))

        # correct prediction nums
        self.correct_prediction = tf.reduce_sum(tf.sign(tf.cast(tf.equal(tf.argmax(y_hat, 1),
                                                                         tf.argmax(y_true_bi, 1)), "float")))
