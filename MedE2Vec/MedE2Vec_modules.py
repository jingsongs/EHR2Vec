#encoding:utf8
#@Author:hadoop
#@File: MedE2Vec_modules.py

import logging
logging.basicConfig(level=logging.INFO)
import tensorflow as tf

def ln(inputs, epsilon=1e-8, scope="ln"):
    '''
    LayerNormaliztion
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    with tf.variable_scope("shared_weight_matrix",reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable(name='weight_mat',
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),embeddings[1:, :]), 0)
    return embeddings


def scaled_dot_product_attention(Q, K, V,dropout_rate=0.1,training=True,
                                 scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs /= d_k ** 0.5
        outputs = mask(outputs, Q, K, type="key")
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
        # query masking
        outputs = mask(outputs, Q, K, type="query")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        outputs = tf.matmul(outputs, V)
    return outputs


def mask(inputs, queries=None, keys=None, type=None):
    padding_num = -2 ** 32 + 1
    if type =="key":
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        masks = tf.expand_dims(masks, 1)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    elif type =="query":
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        masks = tf.expand_dims(masks, -1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])
        outputs = inputs * masks
    return outputs


def multihead_attention(queries, keys, values,num_heads=4,dropout_rate=0.1,training=True,
                        scope="multihead_attention"):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, d_model)
        K = tf.layers.dense(keys, d_model)
        V = tf.layers.dense(values, d_model)
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
        outputs = scaled_dot_product_attention(Q_, K_, V_, dropout_rate, training)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries
        outputs = ln(outputs)
    return outputs

def ff(inputs, num_units, scope="positionwise_feedforward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs += inputs
        outputs = ln(outputs)
    return outputs

class MedE2Vec(object):
    def __init__(self, n_input, d_model, batch_size, maxseq_len,d_ff,num_blocks,num_heads,dropout_rate,log_eps=1e-20,
                 optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.5), init_scale=0.01):
        self.n_input = n_input
        self.d_model = d_model
        self.log_eps = log_eps
        self.init_scale = init_scale
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.maxseq_len = maxseq_len
        self.d_ff=d_ff
        self.num_blocks=num_blocks
        self.num_heads=num_heads
        self.dropout_rate=dropout_rate
        self.embeddings = get_token_embeddings(self.n_input, self.d_model, zero_pad=True)

        self.i_vec = tf.placeholder(tf.int32)
        self.j_vec = tf.placeholder(tf.int32)
        self.idx = tf.placeholder(tf.int32, shape=[None, self.maxseq_len])
        self.vi = tf.placeholder(tf.int32)
        self.vj = tf.placeholder(tf.int32)
        self.v = self.encode(self.idx, self.embeddings,self.d_ff,self.num_blocks,
                             self.num_heads,self.dropout_rate) * 0.1
        self.emb_cost = self._initialize_entity_cost()
        self.vivlcost = self._initialize_visit_cost()
        self.cost = self.emb_cost + self.vivlcost
        self.optimizer = self.optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def encode(self, xs,embeddings,d_model,d_ff,num_blocks,num_heads,dropout_rate,training=True):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            enc = tf.nn.embedding_lookup(embeddings, xs)
            enc *= d_model**0.5
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=num_heads,
                                              dropout_rate=dropout_rate,
                                              training=training)
                    enc = ff(enc, num_units=[d_ff, d_model])
        return enc

    def _initialize_entity_cost(self):
        norms = tf.reduce_sum(tf.exp(tf.matmul(self.v, tf.transpose(self.v, [0, 2, 1]))), axis=2)
        wi_emb = tf.gather(self.v, self.i_vec, axis=1)
        wj_emb = tf.gather(self.v, self.j_vec, axis=1)
        exp = tf.exp(tf.reduce_sum(wi_emb * wj_emb, axis=2))
        norms2 = tf.gather(norms, self.i_vec, axis=1)
        log_sum = tf.reduce_sum(-tf.log(exp / norms2))
        return log_sum


    def _initialize_visit_cost(self):
        w_emb_relu = self.v
        norms = tf.reduce_sum(tf.exp(tf.matmul(w_emb_relu, tf.transpose(w_emb_relu, [0, 2, 1]))), axis=2)
        wi_emb = tf.gather(w_emb_relu, self.vi, axis=0)
        wj_emb = tf.gather(w_emb_relu, self.vj, axis=0)
        exp = tf.exp(tf.reduce_sum(wi_emb * wj_emb, axis=2))
        norms2 = tf.gather(norms, self.vj, axis=0)
        log_sum = tf.reduce_sum(-tf.log(exp / norms2))
        return log_sum


    def partial_fit(self, x=None, i_vec=None, j_vec=None, vi=None, vj=None):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict=
        {self.idx: x, self.i_vec: i_vec, self.j_vec: j_vec, self.vi: vi, self.vj: vj})
        return cost

    def get_visit_representation(self, x=None):
        visit_representation = self.sess.run(self.v, feed_dict={self.x: x})
        return visit_representation

    def get_weights_embeddings(self):
        return self.sess.run(self.embeddings)
