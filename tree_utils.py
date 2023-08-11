'''
Helper functions for parsing the abstract syntax trees (ASTs) in the public
Code.org Hour Of Code #4 dataset.
'''

import os
import json
import pickle

from zss import simple_distance, Node

from data_utils import *

def convert_to_tree(source):
    '''
    Convert a given AST into a zss tree, recursively processing the AST and
    handling special cases as specified.

    For example, with if statements: if someone is missing an IF altogether,
    then their distance will be at least 3 from something with the if - 1 error
    for the if condition, 1 error for the code if true, 1 error for the code if
    false.

    Also note that the zss.simple_distance metric, "swapping" two lines is TWO
    operations: re-labeling the first node and the second node.
    '''

    node_type = source['type']
    if node_type == 'statementList' or node_type == 'maze_turn':
        # Skip these nodes because they are redundant with their children
        node_list = []
        for child in source['children']:
            child_node = convert_to_tree(child)
            if type(child_node) == list:
                for c in child_node:
                    node_list.append(c)
            else:
                node_list.append(child_node)
        return node_list
    elif node_type == 'maze_forever':
        # This node type (loop until finish) is never used in HOC4
        assert len(source['children']) == 1, "While has more than 1 child"
        for child in source['children']: # There's only 1 child, the DO statement
            child_node = convert_to_tree(child)
            if type(child_node) == list: # This should never happen
                raise Exception("Should never be here")
                for list_elem in child_node:
                    node.addkid(list_elem)
            else:
                child_node.label = "while_" + child_node.label
                return child_node
    elif node_type == 'maze_ifElse':
        # This node type (if/else) is never used in HOC4. It should consist
        # of 3 children:
        # - the if condition
        # - the statements to be executed if the condition is met
        # - the statements to be executed if the condition is NOT met
        assert len(source['children']) == 3, "If/else has wrong number of children"

        condition = source['children'][0]['type']
        assert condition in ['isPathLeft', 'isPathRight', 'isPathForward'], "Bad condition"

        condition_node = Node(condition) # The condition of the IF statement

        if_stats = source['children'][1]
        if_stats_return = convert_to_tree(if_stats)
        if type(if_stats_return) == list:
            # Children if the IF statement is satisfied
            for c in if_stats_return:
                condition_node.addkid(c)
        else:
            condition_node.addkid(if_stats_return)

        else_stats = source['children'][2]
        else_stats_return = convert_to_tree(else_stats)
        if type(else_stats_return) == list:
            # Children if the IF statement is NOT satisfied
            for c in else_stats_return:
                condition_node.addkid(c)
        else:
            condition_node.addkid(else_stats_return)

        return condition_node
    else:
        node = Node(source['type'])
        if 'children' in source:
            for child in source['children']:
                child_node = convert_to_tree(child)
                if type(child_node) == list:
                    for list_elem in child_node:
                        node.addkid(list_elem)
                else:
                    node.addkid(child_node)
        return node

def print_tree(node, tab_level = 0):
    print("  " * tab_level, node.label)
    children = Node.get_children(node)
    for child in children:
        print_tree(child, tab_level + 1)

def write_trees():
    '''
    Iterate through all the ASTs in in_dir, and output their zss trees as .tree
    files in out_dir.
    '''

    in_dir = 'hoc_data/hoc4/asts/'
    out_dir = 'hoc_data/hoc4/trees/'
    asts = [x.strip('.json') for x in os.listdir(in_dir) if x != ".DS_Store"]
    asts.remove('counts.txt')
    asts.remove('unitTestResults.txt')
    for ast in sorted(asts):
        with open(in_dir + ast + '.json', 'r') as fin:
            with open(out_dir + ast + '.tree', 'wb') as fout:
                js = json.load(fin)
                tree = convert_to_tree(js)
                pickle.dump(tree, fout)

def compute_pairwise_distances(trees):
    '''
    Precompute the NxN distance matrix between all trees. Store this in a .dist
    file for later use.

    This is much faster than recomputing all the distances on-the-fly when using
    the HOC4 dataset in experiments, but still gives an accurate measure of
    the number of distance calls required (if using the precomputed distance
    matrix, it's the number of calls to the distance matrix).
    '''

    N = len(trees)
    dist_mat = -np.ones((N, N))
    for i in range(N):
        if i % 10 == 0: print(i)
        for j in range(i, N):
            if i == j:
                dist_mat[i, j] = 0
            else:
                i_j_dist = d_tree(trees[i], trees[j], metric = "TREE")
                dist_mat[i, j] = i_j_dist
                dist_mat[j, i] = i_j_dist # symmetric
    np.savetxt('tree-' + str(N) + '.dist', dist_mat)


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    assert args.dataset == "HOC4", "Can only do this for trees"

    write_trees() # Convert all ASTs to zss trees in .tree files
    trees, _1, _2 = load_data(args)
    compute_pairwise_distances(trees) # Write precomputed distance matrix to file
