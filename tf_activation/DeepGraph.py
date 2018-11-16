import json
import decimal
import os
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from tf_activation.utils import dump_graph

from tf_activation.Layers import ConvolutionLayer, FullyConnectedLayer, PoolingLayer

from tf_activation import config

class DeepGraph(object):

    def __init__(self, network=[], conv_layer_name=None, fc_layer_name=None,
                    hidden_layer_name=None, pool_layer_name=None,
                    graph_layer_name=None, layer_counts=None, dump_dir=None):

        self.network = network

        self.conv_layer_name = conv_layer_name
        if conv_layer_name is None:
            self.conv_layer_name = config.CONV_LAYER_NAME
        self.fc_layer_name = fc_layer_name
        if fc_layer_name is None:
            self.fc_layer_name = config.FC_LAYER_NAME
        self.hidden_layer_name = hidden_layer_name
        if hidden_layer_name is None:
            self.hidden_layer_name = config.HIDDEN_LAYER_NAME
        self.pool_layer_name = pool_layer_name
        if pool_layer_name is None:
            self.pool_layer_name = config.POOL_LAYER_NAME
        if graph_layer_name is None:
            self.graph_layer_name = config.GRAPH_LAYER_NAME

        if dump_dir is None:
            self.dump_dir = config.DUMP_DIR

        self.layer_counts = layer_counts
        if layer_counts is None:
            self.layer_counts = self.create_empty_count_dict(self.conv_layer_name,
                                                            self.fc_layer_name,
                                                            self.hidden_layer_name,
                                                            self.pool_layer_name)

        self.name = config.NAME
        self.input_name = config.INPUT_NAME
        self.weight_name = config.WEIGHT_NAME
        self.output_name = config.OUTPUT_NAME

        self.G = nx.Graph()

    def create_empty_count_dict(self, conv_layer_name, fc_layer_name,
                                hidden_layer_name, pool_layer_name):
        return {conv_layer_name: 0,
                fc_layer_name: 0,
                hidden_layer_name: 0,
                pool_layer_name: 0
                }


    def get_default_layer_names(self):
        return {'CONV_LAYER_NAME' : config.CONV_LAYER_NAME,
                'FC_LAYER_NAME' : config.FC_LAYER_NAME,
                'HIDDEN_LAYER_NAME' : config.HIDDEN_LAYER_NAME,
                'POOL_LAYER_NAME' : config.POOL_LAYER_NAME
                }

    def get_default_network_names(self):
        return {'NAME' : config.NAME,
                'INPUT_NAME' : config.INPUT_NAME,
                'WEIGHT_NAME' : config.WEIGHT_NAME,
                'OUTPUT_NAME' : config.OUTPUT_NAME}

    def add_layer(self, name, I, W, O):
        """Creates a layer of type `name` which must match one of the config
        layer names, else raises `LayerNameNotFoundException`. Wraps
        `add_conv_layer`, `add_fc_layer`, and others.

        Args:
            - name (str): name of the layer
            - I (tf.Tensor): Input tensor
            - W (tf.Tensor): Weight tensor connecting input and output layers
            - O (tf.Tensor): Output tensor

        """
        if name == self.conv_layer_name:
            self.add_conv_layer(I, W, O)
        elif name == self.fc_layer_name:
            self.add_fc_layer(I, W, O)
        elif name == self.pool_layer_name:
            self.add_pool_layer(I, W, O)
        else:
            raise LayerNameNotFoundException('Exception in `add_layer`: layer \
name given in param `name` not found in default layer name options')


    def add_conv_layer(self, i, W, o, layer1, layer2, strides, construction='input'):
        """Adds a convolutional layer representation to the network

        Args:
            - i (tf.Tensor): input tensor
            - W (tf.Tensor): weight tensor
            - o (tf.Tensor): output tensor
            - layer1 (int): the layer number for the first (input) layer
            - layer2 (int): the layer number for the second (output) layer
            - construction (Optional[str]): the layer construction method to be
                passed to the layer object constructor.

        """
        t = {}
        t[self.name] = self.conv_layer_name + ':' + str(self.layer_counts[self.conv_layer_name] + 1)
        t[self.graph_layer_name] = ConvolutionLayer(i, W, o, layer1, layer2, strides, construction=construction)
        self.network.append(t)
        self.layer_counts[self.conv_layer_name] += 1

    def add_fc_layer(self, i, W, o, layer1, layer2, construction='input'):
        """Adds a fully-connected layer representation to the network

        Args:
            - i (tf.Tensor): input tensor
            - W (tf.Tensor): weight tensor
            - o (tf.Tensor): output tensor
            - layer1 (int): the layer number for the first (input) layer
            - layer2 (int): the layer number for the second (output) layer
            - construction (Optional[str]): the layer construction method to be
                passed to the layer object constructor.

        """
        t = {}
        t[self.name] = self.fc_layer_name + ':' + str(self.layer_counts[self.fc_layer_name] + 1)
        t[self.graph_layer_name] = FullyConnectedLayer(i, W, o, layer1, layer2, construction=construction)
        self.network.append(t)
        self.layer_counts[self.fc_layer_name] += 1

    def add_pool_layer(self, i, W, o, layer1, layer2, construction='input'):
        """Adds a pooling layer representation to the network

        Args:
            - i (tf.Tensor): input tensor
            - W (tf.Tensor): weight tensor
            - o (tf.Tensor): output tensor
            - layer1 (int): the layer number for the first (input) layer
            - layer2 (int): the layer number for the second (output) layer
            - construction (Optional[str]): the layer construction method to be
                passed to the layer object constructor.

        """
        t = {}
        t[self.name] = self.pool_layer_name + ':' + str(self.layer_counts[self.pool_layer_name] + 1)
        t[self.graph_layer_name] = PoolingLayer(i, W, o, layer1, layer2, construction=construction)
        self.network.append(t)
        self.layer_counts[self.pool_layer_name] += 1


    def print_dim_steps(self):
        """Prints out the dimensionality steps (upsample, downsample) of the
        entire network. Useful in sanity checks and debugging.
        """
        for layer in self.network:
            print(layer['name'] + ' input: ', layer[self.input_name].shape)
            print(layer['name'] + ' output: ', layer[self.output_name].shape)

    def validate_dim_steps(self):
        """Validation function that checks upsample and downsample dimensionality
        for any detectable errors given input and output dimensionality.

        Returns:
            - (bool): True if dimensionality steps are valid for the network

        """
        for i in range(1, len(self.network)-1):
            if (len(self.network[i-1][self.output_name].shape) > 3) and (len(self.network[i][self.input_name].shape) > 3):
                if (self.network[i-1][self.output_name].shape[1] != self.network[i][self.input_name].shape[1]) or (self.network[i-1][self.output_name].shape[2] != self.network[i][self.input_name].shape[2]):
                    raise DimValidationException('Dimensions not valid for ' +
                        self.network[i]['name'] + ' and ' + self.network[i-1]['name'])
            if (len(self.network[i-1][self.output_name].shape) == 2) and (len(self.network[i][self.input_name].shape) == 2):
                if (self.network[i-1][self.output_name].shape[1] != self.network[i][self.input_name].shape[1]):
                    raise DimValidationException('Dimensions not valid for ' +
                        self.network[i]['name'] + ' and ' + self.network[i-1]['name'])

        return True

    def layer_filter_by_percentile(self, percentile=99):
        """Filters each individual layer by setting to zero all edges below the
        `percentile` calculated for each layer individually

        Args:
            - percentile (Optional[int]): The percentile below which edge weights
                are set to zero

        """

        for i in range(len(self.network)):
            print('Filtering layer {}'.format(i))
            self.network[i][self.graph_layer_name].filter_by_percentile(percentile=percentile)
            self.network[i][self.graph_layer_name].relabel()

    def connect_layers(self):
        """Combines individual layer graphs into unified network using
        `nx.compose` function. Sets the result to `self.G`.
        """

        comp = self.network[0][self.graph_layer_name].G
        comp = self.network[0][self.graph_layer_name].G.__class__()
        del self.network[0][self.graph_layer_name].G
        for i in range(1, len(self.network)):
            comp = nx.compose(comp, self.network[i][self.graph_layer_name].G)
            del self.network[i][self.graph_layer_name].G
        self.G = comp
        # comp = self.network[0][self.graph_layer_name].G.__class__()
        # comp.name = "full network"
        # for i in range(len(self.network)):
        #     print('Connecting {}...'.format(i))
        #     comp.add_nodes_from(self.network[i][self.graph_layer_name].G.nodes())
        #     comp.add_edges_from(self.network[i][self.graph_layer_name].G.edges_iter(data=True))
        #     comp.node.update(self.network[i][self.graph_layer_name].G.node)
        #     comp.graph.update(self.network[i][self.graph_layer_name].G.graph)
        #     self.network[i][self.graph_layer_name].G = None
        # self.G = comp


    def dump(self, prefix=None):
        """Writes the entire network graph to json format with appended timestamp.
        Prefixes with 'full_network' if `prefix` kwarg not provided.

        Args:
            - prefix (Optional[str]): the prefix for the file name
        """
        if prefix is not None:
            dump_graph(prefix + '_' + time.strftime("%H:%M:%S_%d-%m-%y"), self.G)
        else:
            dump_graph('full_network' + '_' + time.strftime("%H:%M:%S_%d-%m-%y"), self.G)


    def create_int_map(self, start=0):
        ret = {}
        rev = {}
        for l in self.network:
            f, r, s = l[self.layer_name].create_int_map(start=start)
            ret += f
            rev += r
            start = s
        return ret, rev



class LayerNameNotFoundException(KeyError):
    '''Raise this when layer name
    not found in network'''


class DimValidationException(Exception):
    '''Rase this in `validate_dim_steps` if dim steps are determined invalid'''


def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, float):
        return decimal.Decimal(obj)


def sorted_network_edge_weights(G):
    """Returns list of network edges sorted by the associated weight

    Args:
        - (nx.Graph): The network representation from which to get sorted edges

    Returns:
        - (list): List of sorted network edges
    """

    s = sorted(G.edges(data=True), key=lambda t: t[2]['weight'])
    s.reverse()
    return s

def max_acts_network(G, top=10):
    """Returns a list of the maximum activations across the entire network

    Args:
        - G (nx.Graph): The network representation from which to find maximum activations
        - top (Options[int]): The number of activations to take

    Returns:
        - (list): a list of length `top`

    """

    edges = sorted_network_edge_weights(G)
    return edges[:top]
