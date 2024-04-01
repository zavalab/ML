from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
from deephyper.nas import KSearchSpace
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import Zero, Connect, AddByProjecting
import collections, itertools

from gnn_uq.layers import (
    SPARSE_MPNN,
    GlobalAttentionPool,
    GlobalAttentionSumPool,
    GlobalAvgPool,
    GlobalMaxPool,
    GlobalSumPool
)
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
DistributionLambda = tfp.layers.DistributionLambda

tfd = tfp.distributions


def nll(y, rv_y):
    """Negative log likelihood for Tensorflow probability.

    Args:
        y: true data.
        rv_y: learned (predicted) probability distribution.
    """
    return -rv_y.log_prob(y)


def gnn(shapes, uq=1):
    """
    Construct a Graph Neural Network (GNN) model.

    Args:
        shapes (dict): Dictionary containing input and output shapes.
        uq (int, optional): Flag indicating whether to include uncertainty quantification. Defaults to 1.

    Returns:
        tf.keras.Model: GNN model.
    """
    input_shape = shapes["input_shape"]
    output_shape = shapes["output_shape"][0]
    
    node_ = layers.Input(input_shape[0])
    adj_ = layers.Input(input_shape[1], dtype=tf.int32)
    edge_ = layers.Input(input_shape[2])
    mask_ = layers.Input(input_shape[3])

    input_ = [node_, adj_, edge_, mask_]

    node = layers.BatchNormalization(axis=-1)(node_)
    edge = layers.BatchNormalization(axis=-1)(edge_)

    node = SPARSE_MPNN(state_dim=32,
                       T=1,
                       aggr_method='mean',
                       attn_method='const',
                       update_method='gru',
                       attn_head=1,
                       activation='relu')([node, adj_, edge, mask_])

    node = GlobalAttentionPool(128)(node)
    node = layers.Flatten()(node)
    node = layers.Dense(64, activation='relu')(node)
    node = layers.Dense(64, activation='relu')(node)

    if uq:
        output = layers.Dense(output_shape*2)(node)
        output = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(
                loc=t[..., :output_shape],
                scale=1e-3 + tf.math.softplus(0.05 * t[..., output_shape:]),  # positive constraint on the standard dev.
            )
        )(output)
        loss = nll
    else:
        output = layers.Dense(output_shape, activation='linear')(node)
        loss = 'mse'

    model = tf.keras.Model(input_, output)

    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer, loss=loss, metrics=['mae'])
    return model


class RegressionUQSpace(KSearchSpace):
    """
    Search space for regression models with uncertainty quantification.

    Args:
        input_shape (tuple): Shape of the input data.
        output_shape (tuple): Shape of the output data.
        seed (int, optional): Seed for random number generation. Defaults to None.
        num_layers (int, optional): Number of layers in the sub-graph. Defaults to 3.

    Attributes:
        num_layers (int): Number of layers in the sub-graph.

    """
    def __init__(self, input_shape, output_shape, seed=None, num_layers=3):
        super().__init__(input_shape, output_shape, seed=seed)
        self.num_layers = 3

    def build(self):
        """
        Build the search space.

        Returns:
            RegressionUQSpace: The constructed search space.

        """
        out_sub_graph = self.build_sub_graph(self.input_nodes, self.num_layers)
        output_dim = self.output_shape[0]
        output_dense = ConstantNode(op=Dense(output_dim * 2))
        self.connect(out_sub_graph, output_dense)

        output_dist = ConstantNode(
            op=DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :output_dim],
                    scale=1e-3 + tf.math.softplus(0.05 * t[..., output_dim:]),
                )
            )
        )
        self.connect(output_dense, output_dist)
        return self

    def build_sub_graph(self, input_nodes, num_layers=3):
        """
        Build the sub-graph.

        Args:
            input_nodes (list): List of input nodes.
            num_layers (int, optional): Number of layers in the sub-graph. Defaults to 3.

        Returns:
            VariableNode: The output node of the sub-graph.

        """
        source = prev_input = input_nodes[0]
        prev_input1 = input_nodes[1]
        prev_input2 = input_nodes[2]
        prev_input3 = input_nodes[3]

        anchor_points = collections.deque([source], maxlen=3)

        for _ in range(num_layers):
            graph_attn_cell = VariableNode()
            self.mpnn_cell(graph_attn_cell)
            self.connect(prev_input, graph_attn_cell)
            self.connect(prev_input1, graph_attn_cell)
            self.connect(prev_input2, graph_attn_cell)
            self.connect(prev_input3, graph_attn_cell)

            merge = ConstantNode()
            merge.set_op(AddByProjecting(self, [graph_attn_cell], activation="relu"))

            for node in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(self, node))
                self.connect(skipco, merge)

            prev_input = merge

            anchor_points.append(prev_input)

        global_pooling_node = VariableNode()
        self.gather_cell(global_pooling_node)
        self.connect(prev_input, global_pooling_node)
        prev_input = global_pooling_node

        flatten_node = ConstantNode()
        flatten_node.set_op(Flatten())
        self.connect(prev_input, flatten_node)
        prev_input = flatten_node

        return prev_input

    def mpnn_cell(self, node):
        """
        Add Message Passing Neural Network (MPNN) operations to the given node.

        Args:
            node (VariableNode): The node to which MPNN operations are added.

        """
        state_dims = [16, 32, 64]
        Ts = [1]
        attn_methods = ["const", "gat", "sym-gat", "linear", "gen-linear", "cos"]
        attn_heads = [1, 2, 3]
        aggr_methods = ["max", "mean", "sum"]
        update_methods = ["gru", "mlp"]
        activations = [tf.keras.activations.sigmoid,
                       tf.keras.activations.tanh,
                       tf.keras.activations.relu,
                       tf.keras.activations.linear,
                       tf.keras.activations.elu,
                       tf.keras.activations.softplus,
                       tf.nn.leaky_relu,
                       tf.nn.relu6]

        for hp in itertools.product(state_dims,
                                    Ts,
                                    attn_methods,
                                    attn_heads,
                                    aggr_methods,
                                    update_methods,
                                    activations):
            (state_dim, T, attn_method, attn_head, aggr_method, update_method, activation) = hp

            node.add_op(SPARSE_MPNN(state_dim=state_dim,
                                    T=T,
                                    attn_method=attn_method,
                                    attn_head=attn_head,
                                    aggr_method=aggr_method,
                                    update_method=update_method,
                                    activation=activation))

    def gather_cell(self, node):
        """
        Add operations for global pooling/readout to the given node.

        Args:
            node (VariableNode): The node to which global pooling/readout operations are added.
        """
        for functions in [GlobalSumPool, GlobalMaxPool, GlobalAvgPool]:
            for axis in [-1, -2]:
                node.add_op(functions(axis=axis))
        node.add_op(Flatten())

        for state_dim in [16, 32, 64]:
            node.add_op(GlobalAttentionPool(state_dim=state_dim))
        node.add_op(GlobalAttentionSumPool())