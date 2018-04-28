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
        cell = LSTMCell if self.args.use_lstm else GRUCell

        # model input
        # questions_bt = tf.placeholder(dtype=tf.int32, shape=(None, self.q_len), name="questions_bt")
        # documents_bt = tf.placeholder(dtype=tf.int32, shape=(None, self.d_len), name="documents_bt")
        ques_doc_bt = tf.placeholder(dtype=tf.int32, shape=(None, self.qd_len), name="ques_doc_bt") 
        candidates_bi = tf.placeholder(dtype=tf.int32, shape=(None, self.dataset.A_len), name="candidates_bi")
        y_true_bi = tf.placeholder(shape=(None, self.dataset.A_len), dtype=tf.float32, name="y_true_bi")

        # shape=(None) the length of inputs
        # context_lengths = tf.reduce_sum(tf.sign(tf.abs(documents_bt)), 1)
        # question_lengths = tf.reduce_sum(tf.sign(tf.abs(questions_bt)), 1)
        ques_doc_lengths = tf.reduce_sum(tf.sign(tf.abs(ques_doc_bt)), 1)
        ques_doc_mask_bt = tf.sequence_mask(ques_doc_lengths, self.qd_len, dtype=tf.float32)

        init_embedding = tf.constant(self.embedding_matrix, dtype=tf.float32, name="embedding_init")
        embedding = tf.get_variable(initializer=init_embedding,
                                    name="embedding_matrix",
                                    dtype=tf.float32)

        # with tf.variable_scope('q_encoder', initializer=tf.orthogonal_initializer()):
        #     # encode question to fixed length of vector
        #     # output shape: (None, max_q_length, embedding_dim)
        #     question_embed_btf = tf.nn.embedding_lookup(embedding, questions_bt)
        #     logger("q_embed_btf shape {}".format(question_embed_btf.get_shape()))
        #     q_cell_fw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
        #     q_cell_bw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
        #     outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=q_cell_bw,
        #                                                            cell_fw=q_cell_fw,
        #                                                            dtype="float32",
        #                                                            sequence_length=question_lengths,
        #                                                            inputs=question_embed_btf,
        #                                                            swap_memory=True)
        #     # q_encoder output shape: (None, hidden_size * 2)
        #     q_encoded_bf = tf.concat([last_states[0][-1], last_states[1][-1]], axis=-1)
        #     logger("q_encoded_bf shape {}".format(q_encoded_bf.get_shape()))


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
            qd_encoder_bf = tf.concat([last_states[-1][0], last_states[-1][1]], axis=-1)
            logger("CAUTION!!: qd_encoded_bf shape {}".format(qd_encoded_bf.get_shape()))

        # with tf.variable_scope('d_encoder', initializer=tf.orthogonal_initializer()):
        #     # encode each document(context) word to fixed length vector
        #     # output shape: (None, max_d_length, embedding_dim)
        #     d_embed_btf = tf.nn.embedding_lookup(embedding, documents_bt)
        #     logger("d_embed_btf shape {}".format(d_embed_btf.get_shape()))
        #     d_cell_fw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
        #     d_cell_bw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
        #     outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=d_cell_bw,
        #                                                            cell_fw=d_cell_fw,
        #                                                            dtype="float32",
        #                                                            sequence_length=context_lengths,
        #                                                            inputs=d_embed_btf,
        #                                                            swap_memory=True)
        #     # d_encoder output shape: (None, max_d_length, hidden_size * 2)
        #     d_encoded_btf = tf.concat(outputs, axis=-1)
        #     logger("d_encoded_btf shape {}".format(d_encoded_btf.get_shape()))

        with tf.variable_scope('c_encoder', initializer=tf.orthogonal_initializer()):
            # encode each document(context) word to fixed length vector
            # output shape: (None, max_d_length, embedding_dim)
            d_embed_btf = tf.nn.embedding_lookup(embedding, candidates_bi)
            logger("d_embed_btf shape {}".format(d_embed_btf.get_shape()))
            d_cell_fw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
            d_cell_bw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=d_cell_bw,
                                                                   cell_fw=d_cell_fw,
                                                                   dtype="float32",
                                                                   sequence_length=context_lengths,
                                                                   inputs=d_embed_btf,
                                                                   swap_memory=True)
            # d_encoder output shape: (None, max_d_length, hidden_size * 2)
            d_encoded_btf = tf.concat(outputs, axis=-1)
            logger("d_encoded_btf shape {}".format(d_encoded_btf.get_shape()))

        def att_dot(x):
            """attention dot product function"""
            d_btf, q_bf = x
            res = tf.matmul(tf.expand_dims(q_bf, -1), d_btf, adjoint_a=True, adjoint_b=True)
            return tf.reshape(res, [-1, self.d_len])

        with tf.variable_scope('merge'):
            mem_attention_pre_soft_bt = att_dot([d_encoded_btf, q_encoded_bf])
            mem_attention_pre_soft_masked_bt = tf.multiply(mem_attention_pre_soft_bt,
                                                           context_mask_bt,
                                                           name="attention_mask")
            mem_attention_bt = tf.nn.softmax(logits=mem_attention_pre_soft_masked_bt, name="softmax_attention")

        # output shape: (None, i) i = max_candidate_length = 10
        y_hat = sum_probs_batch(candidates_bi, documents_bt, mem_attention_bt)

        # crossentropy
        output = y_hat / tf.reduce_sum(y_hat, axis=-1, keep_dims=True)
        # manual computation of crossentropy
        epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype, name="epsilon")
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        self.loss = tf.reduce_mean(- tf.reduce_sum(y_true_bi * tf.log(output), axis=-1))

        # correct prediction nums
        self.correct_prediction = tf.reduce_sum(tf.sign(tf.cast(tf.equal(tf.argmax(y_hat, 1),
                                                                         tf.argmax(y_true_bi, 1)), "float")))
