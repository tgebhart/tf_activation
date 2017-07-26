from networkx.readwrite import json_graph
import json
import os
import config

def dump_graph(filename, G):
    d = json_graph.node_link_data(G)
    json.dump(d, open(os.path.join(config.DIR_NAME, config.DUMP_DIR, filename + '.json'), 'w'))
