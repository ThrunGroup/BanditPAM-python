import os
import json
import pickle

from zss import simple_distance, Node

def convert_to_tree(source):
    '''
    Small issue with if statements: if someone is missing an IF altogether, then
    their distance will be 3 from something with the if. Maybe that's what we want anyways.
    '''

    node_type = source['type']
    if node_type == 'statementList' or node_type == 'maze_turn':
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
        assert len(source['children']) == 1, "While has more than 1 child"
        for child in source['children']: # There's only 1 child, the DO
            child_node = convert_to_tree(child)
            if type(child_node) == list: # Not gonna happen
                raise Exception("Should never be here")
                for list_elem in child_node:
                    node.addkid(list_elem)
            else:
                child_node.label = "while_" + child_node.label
                return child_node
    elif node_type == 'maze_ifElse':
        assert len(source['children']) == 3, "If/else has wrong number of children"

        condition = source['children'][0]['type']
        assert condition in ['isPathLeft', 'isPathRight', 'isPathForward'], "Bad condition"

        condition_node = Node(condition)

        if_stats = source['children'][1]
        if_stats_return = convert_to_tree(if_stats)
        if type(if_stats_return) == list:
            for c in if_stats_return:
                condition_node.addkid(c)
        else:
            condition_node.addkid(if_stats_return)

        else_stats = source['children'][2]
        else_stats_return = convert_to_tree(else_stats)
        if type(else_stats_return) == list:
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
    in_dir = 'hoc_data/hoc18/asts/'
    out_dir = 'hoc_data/hoc18/trees/'
    asts = [x.strip('.json') for x in os.listdir(in_dir) if x != ".DS_Store"]
    asts.remove('counts.txt')
    asts.remove('unitTestResults.txt')
    for ast in sorted(asts):
        with open(in_dir + ast + '.json', 'r') as fin:
            with open(out_dir + ast + '.tree', 'wb') as fout:
                js = json.load(fin)
                tree = convert_to_tree(js)
                pickle.dump(tree, fout)

if __name__ == "__main__":
    write_trees()


# if __name__ == "__main__":
# # use this to spot-check edit distances
#     ast0 = "hoc_data/hoc4/asts/0.json"
#     ast1 = "hoc_data/hoc4/asts/1.json"
#     with open(ast0, 'r') as fin1:
#         with open(ast1, 'r') as fin2:
#             js = json.load(fin1)
#             tree0 = convert_to_tree(js)
#             print_tree(tree0)
#             print("\n")
#
#             js = json.load(fin2)
#             tree1 = convert_to_tree(js)
#             print_tree(tree1)
#             print("\n")
#
#             # Simple distance counts non-equal labels as 1, as desired. I verified this.
#             # Strangely, the editdist pkg that zss suggests installing doesn't seem to exist?
#             print(simple_distance(tree0, tree1))

    # # Use this to check that you can load pickle files
    # print("----------\n\n\n")
    # pik0_f = "hoc_data/hoc4/trees/0.tree"
    # pik1_f = "hoc_data/hoc4/trees/1.tree"
    #
    # with open(pik0_f, 'rb') as fin1:
    #     with open(pik1_f, 'rb') as fin2:
    #         tree0 = pickle.load(fin1)
    #         tree1 = pickle.load(fin2)
    #         print_tree(tree0)
    #         print("\n")
    #         print_tree(tree1)
    #         print(simple_distance(tree0, tree1))


# if __name__ == "__main__":
# # Use this to spot-check generated trees
#     for i in range(10000):
#         ast0 = "hoc_data/hoc18/asts/" + str(i) + ".json"
#         if os.path.exists(ast0):
#             print(i)
#             with open(ast0, 'r') as fin1:
#                 js = json.load(fin1)
#                 tree0 = convert_to_tree(js)
#                 print_tree(tree0)
#                 print("\n")
