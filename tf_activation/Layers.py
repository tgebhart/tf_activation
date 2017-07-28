import json
import decimal
import os
import time
import math

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import graph_writer

from utils import dump_graph

import config

class Layer(object):

    def __init__(self, layer_type, inp, Weight, output):

        self.layer_type = layer_type

        self.i = inp
        self.o = output
        self.W = Weight

        self.G = nx.Graph()
        self.labels = []


    def set_input(self, inp):
        self.i = inp

    def set_output(self, out):
        self.o = out

    def set_weight(self, W):
        self.W = W

    def filter_by_percentile(self, percentile=99, inplace=True):
        """Filters the adjacency matrix by setting to 0 all elements that are
        below the `percentile` cutoff value which is computed from the matrix
        itself.

        Args:
            - percentile (Optional[int]): the percentile from which a cutoff is
                to be calculated
            - inplace (Optional[bool]): `True` if the calculation should be written
                back to `self.G`. `False` returns this filtration.

        Returns:
            - (nx.Graph): if `inplace` is True.

        """

        adj = nx.to_numpy_matrix(self.G)
        a_adj = np.absolute(adj)
        p = np.percentile(a_adj, percentile)
        lows = a_adj < p
        adj[lows] = 0
        G = nx.from_numpy_matrix(adj)
        if inplace:
            self.G = G
        else:
            return G

    def set_labels(self):
        self.labels = self.G.nodes()

    def set_layer_attribute(self):
        """Sets the layer node attribute using the networkx built-in
        `set_node_attributes`.
        """

        mapping = {}
        for node in self.G:
            mapping[node] = node[len(node)-1]
        nx.set_node_attributes(self.G, 'layer', mapping)

    def set_channel_attribute(self):
        """Sets the channel node attribute by using the networkx built-in
        `set_node_attributes` and by recomputing the channel from the node labels.
        """

        mapping = {}
        for node in self.G:
            if len(node) > 3:
                mapping[node] = node[2]
            else:
                mapping[node] = 'input'
        nx.set_node_attributes(self.G, 'channel', mapping)

    def relabel(self, inplace=True):
        """Adds node labels to the graph assuming input adjacency matrix is of
        same dimensionality of original graph i.e. we can assign unique tuple
        labels as in `construct`.

        Args:
            - inplace (Optional[bool]): affects `self.G` if true, otherwise
                returns resultant nx.Graph object

        Returns:
            - (nx.Graph): if inplace is False
        """

        def mapping(x):
            return self.labels[x]
        if inplace:
            nx.relabel_nodes(self.G, mapping, copy=False)
            return None
        else:
            return nx.relabel_nodes(self.G, mapping, copy=True)

    def get_attr_matrix(self, attr):
        """Returns a numpy matrix using the attributes from `self.G`

        Args:
            - attr (str): the attribute name around which the matrix is constructed

        Returns:
            - (np.Matrix): attribute matrix
        """

        return nx.attr_matrix(self.G, attr=attr, rc_order=self.G.nodes())

    def sorted_edge_weights(self):
        """Returns list of layer edges sorted by the associated weight

        Returns:
            - (list): List of sorted network edges
        """

        s = sorted(self.G.edges(data=True), key=lambda (s,t,d): d['weight'])
        s.reverse()
        return s

    def max_acts(self):
        """Returns a list of the maximum activations across the layer.

        Args:
            - top (Options[int]): The number of activations to take

        Returns:
            - (list): a list of length `top`

        """

        edges = self.sorted_edge_weights()
        return edges[:top]

    def dump(self):
        """Wrapper write-to-file function on top of `utils.dump_graph`
        """

        dump_graph(self.layer_type + '_' + time.strftime("%H:%M:%S_%d-%m-%y"), self.G)

    def create_int_maps(self, start=0):
        ret = {}
        for node in self.G.nodes():
            ret[node] = start
            start = start + 1
        rev = {v: k for k, v in ret.iteritems()}
        return ret, rev, start


class ConvolutionLayer(Layer):

    def __init__(self, inp, Weight, output, layer1, layer2, strides, padding='SAME', construction='input'):

        super(ConvolutionLayer, self).__init__('convolution', inp, Weight, output)

        self.layer1 = layer1
        self.layer2 = layer2
        self.strides = strides
        self.padding = padding

        if construction == 'input':
            self.construct_input()
        else:
            self.construct()


    def construct(self):
        """Constructs the networkx representation of a convolutional layer in
        graphical form by iterating over the input data (height and width) and
        then iterating over the channels (if any). The nodes are labeled using
        tuples in (height, width, [channel], layer) format. The edges between
        nodes are weighted based on filter values for each channel.
        """

        G = nx.Graph()
        if self.padding == 'SAME':
            ohs = self.o.shape[1]
            ows = self.o.shape[2]
            ihs = self.i.shape[1]
            iws = self.i.shape[2]
            ochannels = self.o.shape[3]
            ichannels = self.i.shape[3]
            fh = self.W.shape[0]
            fw = self.W.shape[1]
            for ih in range(ihs):
                for iw in range(iws):
                    for ic in range(ichannels):
                        G.add_node((ih, iw, ic, self.layer1), layer=self.layer1)
                        for oc in range(ochannels):
                            G.add_node((ih, iw, oc, self.layer2), layer=self.layer2)
                            inputs = self.convolve_graph(ih, iw, ic, oc)
                            for info in inputs:
                                G.add_weighted_edges_from([((info['in_height'], info['in_width'], ic, self.layer1), (ih, iw, oc, self.layer2), info['weight'])])


        else:
            raise OptionNotImplemented('Error in `construct()`: Padding option of type ' + str(self.padding) + ' not yet available.')

        self.G = G
        self.set_labels()

    def construct_input(self, idx=0):
        """Constructs the networkx representation of a convolutional layer in
        graphical form by iterating over the input data (height and width) and
        then iterating over the channels (if any). The nodes are labeled using
        tuples in (height, width, [channel], layer) format. The edges between
        nodes are weighted based on the filter values multiplied by the input
        values for each channel.
        """

        G = nx.Graph()
        if self.padding == 'SAME':
            ohs = self.o.shape[1]
            ows = self.o.shape[2]
            ihs = self.i.shape[1]
            iws = self.i.shape[2]
            ochannels = self.o.shape[3]
            ichannels = self.i.shape[3]
            fh = self.W.shape[0]
            fw = self.W.shape[1]
            for ih in range(ihs):
                for iw in range(iws):
                    for ic in range(ichannels):
                        G.add_node((ih, iw, ic, self.layer1), layer=self.layer1)
                        for oc in range(ochannels):
                            G.add_node((ih, iw, oc, self.layer2), layer=self.layer2)
                            inputs = self.convolve_graph_inputs(ih, iw, ic, oc, idx=idx)
                            for info in inputs:
                                G.add_weighted_edges_from([((info['in_height'], info['in_width'], ic, self.layer1), (ih, iw, oc, self.layer2), info['weight'])])


        else:
            raise OptionNotImplemented('Error in `construct()`: Padding option of type ' + str(self.padding) + ' not yet available.')

        self.G = G
        self.set_labels()

    def convolve_graph(self, ih, iw, ic, oc):
        W = self.W
        fh = W.shape[0]
        fw = W.shape[1]
        mid_h = fh//2
        mid_w = fw//2
        inputs = []
        inputs.append({'in_height': ih, 'in_width': iw, 'weight': W[mid_h, mid_w, ic, oc]})
        for h in range(-mid_h, mid_h):
            h_idx = ih + h
            if h_idx >= 0 and h_idx < self.i.shape[1]:
                for w in range(-mid_w, mid_w):
                    w_idx = iw + w
                    if w_idx >= 0 and w_idx < self.i.shape[2]:
                        inputs.append({'in_height': h_idx, 'in_width': w_idx, 'weight': float(W[mid_h + h, mid_w + w, ic, oc])})
        return inputs

    def convolve_graph_inputs(self, ih, iw, ic, oc, idx=0):
        W = self.W
        fh = W.shape[0]
        fw = W.shape[1]
        mid_h = fh//2
        mid_w = fw//2
        inputs = []
        inputs.append({'in_height': ih, 'in_width': iw, 'weight': W[mid_h, mid_w, ic, oc] * float(self.i[ih, iw, ic, oc])})
        for h in range(-mid_h, mid_h):
            h_idx = ih + h
            if h_idx >= 0 and h_idx < self.i.shape[1]:
                for w in range(-mid_w, mid_w):
                    w_idx = iw + w
                    if w_idx >= 0 and w_idx < self.i.shape[2]:
                        inputs.append({'in_height': h_idx, 'in_width': w_idx, 'weight': float(W[mid_h+h,mid_w+w,ic,oc]) * float(self.i[idx,h_idx,w_idx,ic])})
        return inputs


class FullyConnectedLayer(Layer):

    def __init__(self, inp, Weight, output, layer1, layer2, construction='input'):

        super(FullyConnectedLayer, self).__init__('fully-connected', inp, Weight, output)

        self.layer1 = layer1
        self.layer2 = layer2

        if construction == 'input':
            self.construct_input()
        else:
            self.construct()

    def construct(self):
        """Constructs the networkx representation of a fully-connected layer in
        graphical form by iterating over the input data (height and width). The
        nodes are labeled in tuples (height, width, [channel], layer) format.
        The edges between nodes are weighted based on the weight matrix elements.
        """

        G = nx.Graph()
        print(self.i.shape, self.W.shape, self.o.shape)
        if len(self.i.shape) > 2:
            channels = self.i.shape[3]
            hs = self.i.shape[1]
            ws = self.i.shape[2]
        else:
            channels = 1
            hs = 1
            ws = self.i.shape[1]

        Ws = self.W.shape[1]

        for w in range(Ws):
            G.add_node((w, self.layer2), layer=self.layer2)
            for c in range(channels):
                for fh in range(hs):
                    for fw in range(ws):
                        G.add_node((fh, fw, c, self.layer1), layer=self.layer1)
                        channel_offset = ((ws * fh) + fw) + (ws * hs * c)
                        G.add_weighted_edges_from([((fh, fw, c, self.layer1), (w, self.layer2), float(self.W[channel_offset, w]))])

        self.G = G
        self.set_labels()



    def construct_input(self, idx=0):
        """Constructs the networkx representation of a fully-connected layer in
        graphical form by iterating over the input data (height and width). The
        nodes are labeled in tuples (height, width, [channel], layer) format.
        The edges between nodes are weighted based on the weight matrix elements
        multiplied by the input activations.
        """

        G = nx.Graph()
        print(self.i.shape, self.W.shape, self.o.shape)
        if len(self.i.shape) > 2:
            channels = self.i.shape[3]
            hs = self.i.shape[1]
            ws = self.i.shape[2]
        else:
            channels = 1
            hs = 1
            ws = self.i.shape[1]

        Ws = self.W.shape[1]

        for w in range(Ws):
            G.add_node((w, self.layer2), layer=self.layer2)
            for c in range(channels):
                for fh in range(hs):
                    for fw in range(ws):
                        G.add_node((fh, fw, c, self.layer1), layer=self.layer1)
                        channel_offset = ((ws * fh) + fw) + (ws * hs * c)
                        # print(self.i[idx, fh, fw, c], self.W[channel_offset, w])
                        try:
                            G.add_weighted_edges_from([((fh, fw, c, self.layer1), (w, self.layer2), float(self.i[idx,fh,fw,c]) * float(self.W[channel_offset, w]))])
                        except IndexError:
                            G.add_weighted_edges_from([((fh, fw, c, self.layer1), (w, self.layer2), float(self.i[idx,fw]) * float(self.W[channel_offset, w]))])
        self.G = G
        self.set_labels()

class PoolingLayer(Layer):

    def __init__(self, inp, Weight, output, layer1, layer2, construction='input'):

        super(PoolingLayer, self).__init__('pooling', inp, Weight, output)

        self.layer1 = layer1
        self.layer2 = layer2

        if construction == 'input':
            self.construct_input()
        else:
            self.construct()


    def construct(self):
        """Constructs the networkx representation of a pooling layer in
        graphical form by iterating over the input data (height and width). The
        nodes are labeled in tuples (height, width, [channel], layer) format.
        The edges between nodes are set to 1.
        """

        G = nx.Graph()
        print(self.i.shape, self.o.shape)
        channels = self.i.shape[3]
        ihs = self.i.shape[1]
        iws = self.i.shape[2]
        ohs = self.o.shape[1]
        ows = self.o.shape[2]
        downsampling_h = ihs//ohs
        downsampling_w = iws//ows
        for h in range(ihs):
            for w in range(iws):
                for c in range(channels):
                    if w % downsampling_w == 0:
                        d_w = w//downsampling_w
                        if h % downsampling_h == 0:
                            d_h = h//downsampling_h
                            G.add_node((d_h,d_w,c,self.layer2), layer=self.layer2)
                    G.add_node((h,w,c,self.layer1), layer=self.layer1)
                    G.add_weighted_edges_from([((h,w,c,self.layer1), (d_h,d_w,c,self.layer2), 1.0)])

        self.G = G
        self.set_labels()

    def construct_input(self, idx=0):
        """Constructs the networkx representation of a pooling layer in
        graphical form by iterating over the input data (height and width). The
        nodes are labeled in tuples (height, width, [channel], layer) format.
        The edges between nodes are set to the activation values to the
        input layer nodes.
        """

        G = nx.Graph()
        print(self.i.shape, self.o.shape)
        channels = self.i.shape[3]
        ihs = self.i.shape[1]
        iws = self.i.shape[2]
        ohs = self.o.shape[1]
        ows = self.o.shape[2]
        downsampling_h = ihs//ohs
        downsampling_w = iws//ows
        for h in range(ihs):
            for w in range(iws):
                for c in range(channels):
                    if w % downsampling_w == 0:
                        d_w = w//downsampling_w
                        if h % downsampling_h == 0:
                            d_h = h//downsampling_h
                            G.add_node((d_h,d_w,c,self.layer2), layer=self.layer2)
                    G.add_node((h,w,c,self.layer1), layer=self.layer1)
                    G.add_weighted_edges_from([((h,w,c,self.layer1), (d_h,d_w,c,self.layer2), float(self.i[idx,h,w,c]))])

        self.G = G
        self.set_labels()

class OptionNotImplemented(KeyError):
    '''Raise this when tensorflow option not yey available to model'''
