import numpy as np
import networkx as nx

from utils.graph_utils import *
from utils.other_utils import *


class graph_allow_alias(nx.DiGraph):
    def __init__(self, G):
        self.G = G
        self.alias_to_main_nodes = {}
        self.main_to_alias_nodes = {}
        for key, value in self.G.nodes().items():
            main_key = key
            aliases = [key]
            if 'contraction' in value:
                for c_key, c_value in value['contraction'].items():
                    aliases.append(c_key)
            for a_k in aliases:
                self.alias_to_main_nodes[a_k] = main_key
            self.main_to_alias_nodes[main_key] = aliases

        self.alias_to_main_edges = {}
        self.main_to_alias_edges = {}
        for key, value in self.G.edges().items():
            main_key = key
            aliases = [key]
            u_aliases, v_aliases = self.main_to_alias_nodes[key[0]], self.main_to_alias_nodes[key[1]]
            for u_a in u_aliases:
                for v_a in v_aliases:
                    aliases.append((u_a, v_a))
            for a_k in aliases:
                self.alias_to_main_edges[a_k] = main_key
            self.main_to_alias_edges[main_key] = aliases

    def alias_node_key(self, node_key):
        return self.alias_to_main_nodes[node_key]

    def alias_node(self, node_key):
        return self.G.nodes[self.alias_to_main_nodes[node_key]]

    def alias_edge_key(self, u, v):
        return self.alias_to_main_edges[u, v]

    def alias_edge(self, u, v):
        return self.G.edges[self.alias_to_main_edges[u, v]]

    def add_node(self, node_name, **kwargs):
        self.G.add_node(node_name, **kwargs)
        self.alias_to_main_nodes[node_name] = node_name
        self.main_to_alias_nodes[node_name] = [node_name]

    def add_edge(self, n1, n2, **kwargs):
        u, v = self.alias_node_key(n1), self.alias_node_key(n2)
        self.G.add_edge(u, v, **kwargs)

        main_key = (u, v)
        aliases = [(u, v)]
        u_aliases, v_aliases = self.main_to_alias_nodes[u], self.main_to_alias_nodes[v]
        for u_a in u_aliases:
            for v_a in v_aliases:
                aliases.append((u_a, v_a))
        for a_k in aliases:
            self.alias_to_main_edges[a_k] = main_key
        self.main_to_alias_edges[main_key] = aliases

    def has_node(self, node_name):
        return node_name in self.alias_to_main_nodes

    def remove_node(self, node_name):
        try:
            removed_node_key = self.alias_to_main_nodes[node_name]
        except:
            return

        removed_edges = list(self.G.edges(removed_node_key))
        for u, v in removed_edges:
            u_key, v_key = self.alias_to_main_nodes[u], self.alias_to_main_nodes[v]
            del self.main_to_alias_edges[(u_key, v_key)]

            aliases = []
            u_aliases, v_aliases = self.main_to_alias_nodes[u_key], self.main_to_alias_nodes[v_key]
            for u_a in u_aliases:
                for v_a in v_aliases:
                    aliases.append((u_a, v_a))
            for a_k in aliases:
                del self.alias_to_main_edges[a_k]

        self.G.remove_node(removed_node_key)

        alias_list = self.main_to_alias_nodes[removed_node_key]
        for alias in alias_list:
            del self.alias_to_main_nodes[alias]
        del self.main_to_alias_nodes[removed_node_key]

def segment_augment(G2, seg_G, seg_name, aug_dev=0.0, aug_width=0.0,
                    lane_add=0, hole_add=0, edge_rewire=True, seg_G_add=True, hole_candids=None):
    road_name = seg_name.split('_')[0]
    if seg_name not in seg_G.nodes or len(seg_G.nodes[seg_name]['point_names']) == 0:
        return G2.copy(), seg_G.copy()

    seg_G2 = seg_G.copy()
    G_alias = graph_allow_alias(G2.copy())
    seg_list = [key for key, value in seg_G2.nodes().items() \
                if key.split('_')[0] == road_name and len(value['point_names']) > 0]

    nodes_in_road = {seg: seg_G2.nodes[seg]['point_names'].copy() for seg in seg_list}
    ref_seg_direc = int(seg_name.split('_')[-1])

    # node1_pos = G_alias.alias_node(nodes_in_road[seg_name][0])['pos']
    # node2_pos = G_alias.alias_node(nodes_in_road[seg_name][-1])['pos']

    # road_direc = (node1_pos - node2_pos) if ref_seg_direc > 0 else (node2_pos - node1_pos)
    # road_direc_norm = np.linalg.norm(road_direc)
    # road_direc = road_direc / road_direc_norm if road_direc_norm != 0 else road_direc
    # road_direc_perp = np.array([-road_direc[1], road_direc[0]])

    same_direct_seg_list = [seg for seg in seg_list if int(seg.split('_')[-1]) * ref_seg_direc > 0]
    same_direct_seg_list = sorted(same_direct_seg_list, key=lambda k: abs(int(k.split('_')[-1])))
    out_most_lane_id = same_direct_seg_list[-1]
    out_most_lane_num = out_most_lane_id.split('_')[-1]
    road_center = seg_G2.nodes[out_most_lane_id]['road_center'].copy()

    # add new lane
    if lane_add > 0:
        new_lane_num = int(out_most_lane_num) + 1. * (ref_seg_direc > 0) - 1. * (ref_seg_direc < 0)
        new_lane_id = road_name + '_' + str(int(new_lane_num))

        seg_frenet = seg_G2.nodes[out_most_lane_id]['frenet']
        new_seg_frenet = seg_frenet.copy()
        new_seg_frenet[:, 1] += 4 * (2. * seg_G2.nodes[out_most_lane_id]['is_inverted'] - 1.)
        new_seg_points = frenet_to_cartesian_approx(road_center, new_seg_frenet)

        # add nodes
        added_node = []
        for n, p in enumerate(new_seg_points):
            new_node_name = new_lane_id + "_" + str(n)
            G_alias.add_node(new_node_name, road=road_name, seg=new_lane_id, pos=new_seg_points[n],
                             order=n, frenet=new_seg_frenet[n], lane_section=None,
                             on_traffic_signal=seg_G2.nodes[out_most_lane_id]['on_traffic_signal'])
            added_node.append(new_node_name)
        nodes_in_road[new_lane_id] = added_node
        seg_list.append(new_lane_id)

        if seg_G_add:
            out_most_left_border = seg_G2.nodes[out_most_lane_id]['left_border'].copy()
            out_most_right_border = seg_G2.nodes[out_most_lane_id]['right_border'].copy()
            if seg_G2.nodes[out_most_lane_id]['is_inverted']:
                left_border = 2 * out_most_left_border - out_most_right_border
                right_border = out_most_left_border
            else:
                left_border = out_most_right_border
                right_border = 2 * out_most_right_border - out_most_left_border
            center_roadmark = seg_G2.nodes[out_most_lane_id]['center_roadmark'].copy()
            lane_roadmark = seg_G2.nodes[out_most_lane_id]['lane_roadmark'].copy()
            lane_center = (left_border + right_border) / 2

            seg_G2.add_node(new_lane_id, road=road_name,
                            is_inverted=seg_G2.nodes[out_most_lane_id]['is_inverted'],
                            on_traffic_signal=seg_G2.nodes[out_most_lane_id]['on_traffic_signal'],
                            lane_drivable=seg_G2.nodes[out_most_lane_id]['lane_drivable'].copy(),
                            frenet=new_seg_frenet, road_center=road_center,
                            left_border=left_border, right_border=right_border,
                            lane_roadmark=lane_roadmark, center_roadmark=center_roadmark,
                            point_names=added_node.copy(), lane_center=lane_center,
                            in_junction=seg_G2.nodes[out_most_lane_id]['in_junction'])

            changed_lane_roadmark = np.array(['1.2500000000000000e-1_white_broken'] * len(lane_roadmark))
            seg_G2.nodes[out_most_lane_id]['lane_roadmark'] = changed_lane_roadmark

        # change extorior edges
        if edge_rewire:
            try:
                start_node = G_alias.alias_node_key(nodes_in_road[out_most_lane_id][0])
                end_node = G_alias.alias_node_key(nodes_in_road[out_most_lane_id][-1])
                new_start_node = G_alias.alias_node_key(nodes_in_road[new_lane_id][0])
                new_end_node = G_alias.alias_node_key(nodes_in_road[new_lane_id][-1])
            except:
                print(out_most_lane_id, nodes_in_road[out_most_lane_id])
                return

            edges_connected_to_start_node = G_alias.G.in_edges(start_node)
            edges_connected_to_end_node = G_alias.G.out_edges(end_node)

            for n1, n2 in edges_connected_to_start_node:
                edge_direc = G_alias.alias_node(new_start_node)['pos'] - G_alias.alias_node(n1)['pos']
                edge_type = G_alias.alias_edge(n1, n2)['edge_type']
                edge_type_idx = G_alias.alias_edge(n1, n2)['edge_type_idx']
                if G_alias.has_node(n1) and G_alias.has_node(new_start_node):
                    G_alias.add_edge(n1, new_start_node, edge_type=edge_type, edge_type_idx=edge_type_idx,
                                     direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))

                if seg_G_add:
                    n1_seg = G_alias.alias_node(n1)['seg']
                    n2_seg = G_alias.alias_node(n2)['seg']
                    new_start_seg = G_alias.alias_node(new_start_node)['seg']
                    if new_start_seg not in [n1_seg, n2_seg] and seg_G2.has_edge(n1_seg, n2_seg):
                        if seg_G2.has_node(n1_seg) and seg_G2.has_node(new_start_seg):
                            seg_G2.add_edge(n1_seg, new_start_seg, \
                                            edge_type=seg_G2.edges[n1_seg, n2_seg]['edge_type'])

            for n1, n2 in edges_connected_to_end_node:
                edge_direc = G_alias.alias_node(n2)['pos'] - G_alias.alias_node(new_end_node)['pos']
                edge_type = G_alias.alias_edge(n1, n2)['edge_type']
                edge_type_idx = G_alias.alias_edge(n1, n2)['edge_type_idx']
                if G_alias.has_node(n2) and G_alias.has_node(new_end_node):
                    G_alias.add_edge(new_end_node, n2, edge_type=edge_type, edge_type_idx=edge_type_idx,
                                     direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))
                if seg_G_add:
                    n1_seg = G_alias.alias_node(n1)['seg']
                    n2_seg = G_alias.alias_node(n2)['seg']
                    new_end_seg = G_alias.alias_node(new_end_node)['seg']
                    if new_end_seg not in [n1_seg, n2_seg] and seg_G2.has_edge(n1_seg, n2_seg):
                        if seg_G2.has_node(n2_seg) and seg_G2.has_node(new_end_seg):
                            seg_G2.add_edge(new_end_seg, n2_seg, \
                                            edge_type=seg_G2.edges[n1_seg, n2_seg]['edge_type'])

        # linked_seg_list_at_start = [seg.nodes[n1] for n1, _ in edges_connected_to_start_node]
        # for n1,n2 in G_arrange.G.in_edges(nodes_in_road[out_most_lane_id][0]):
        # print(n1,n2)
        # linked_seg_list_at_start = []
        # linked_seg_list_at_start = []

        # add sequential edges
        for n in range(len(new_seg_points) - 1):
            prev_node_name = new_lane_id + "_" + str(n)
            next_node_name = new_lane_id + "_" + str(n + 1)
            edge_direc = G_alias.alias_node(next_node_name)['pos'] - G_alias.alias_node(prev_node_name)['pos']
            if G_alias.has_node(prev_node_name) and G_alias.has_node(next_node_name):
                G_alias.add_edge(prev_node_name, next_node_name, edge_type='sequential',
                                 edge_type_idx=np.array([1, 0, 0, 0, 0]),
                                 direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))

        # add lane change edges
        lane_change_progress_min_dist, lane_change_progress_max_dist = 0.0, 6.0
        lane_points = nodes_in_road[out_most_lane_id]
        is_inverted = seg_G2.nodes[out_most_lane_id]['is_inverted']
        changed_lane_points = [new_lane_id + "_" + str(n) for n in range(len(new_seg_points))]
        changed_lane_idx = 0
        for p_id in lane_points:
            while (changed_lane_idx < len(changed_lane_points)):
                changed_p_id = changed_lane_points[changed_lane_idx]
                if is_inverted:
                    lane_s = -G_alias.alias_node(p_id)['frenet'][0]
                    changed_lane_s = -G_alias.alias_node(changed_p_id)['frenet'][0]
                else:
                    lane_s = G_alias.alias_node(p_id)['frenet'][0]
                    changed_lane_s = G_alias.alias_node(changed_p_id)['frenet'][0]

                if changed_lane_s < lane_s + lane_change_progress_min_dist:
                    changed_lane_idx += 1
                else:
                    if changed_lane_s < lane_s + lane_change_progress_max_dist:
                        edge_direc = G_alias.alias_node(changed_p_id)['pos'] - G_alias.alias_node(p_id)['pos']
                        if G_alias.has_node(p_id) and G_alias.has_node(changed_p_id):
                            G_alias.add_edge(p_id, changed_p_id, edge_type='right',
                                             edge_type_idx=np.array([0, 0, 0, 1, 0]),
                                             direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))

                        p_seg = G_alias.alias_node(p_id)['seg']
                        changed_p_seg = G_alias.alias_node(changed_p_id)['seg']
                        if seg_G_add:
                            if seg_G2.has_node(p_seg) and seg_G2.has_node(changed_p_seg):
                                seg_G2.add_edge(p_seg, changed_p_seg, edge_type='right')
                    break
        lane_points = [new_lane_id + "_" + str(n) for n in range(len(new_seg_points))]
        is_inverted = seg_G2.nodes[out_most_lane_id]['is_inverted']
        changed_lane_points = nodes_in_road[out_most_lane_id]
        changed_lane_idx = 0
        for p_id in lane_points:
            while (changed_lane_idx < len(changed_lane_points)):
                changed_p_id = changed_lane_points[changed_lane_idx]
                if is_inverted:
                    lane_s = -G_alias.alias_node(p_id)['frenet'][0]
                    changed_lane_s = -G_alias.alias_node(changed_p_id)['frenet'][0]
                else:
                    lane_s = G_alias.alias_node(p_id)['frenet'][0]
                    changed_lane_s = G_alias.alias_node(changed_p_id)['frenet'][0]

                if changed_lane_s < lane_s + lane_change_progress_min_dist:
                    changed_lane_idx += 1
                else:
                    if changed_lane_s < lane_s + lane_change_progress_max_dist:
                        edge_direc = G_alias.alias_node(changed_p_id)['pos'] - G_alias.alias_node(p_id)['pos']
                        if G_alias.has_node(p_id) and G_alias.has_node(changed_p_id):
                            G_alias.add_edge(p_id, changed_p_id, edge_type='left',
                                             edge_type_idx=np.array([0, 0, 1, 0, 0]),
                                             direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))

                        p_seg = G_alias.alias_node(p_id)['seg']
                        changed_p_seg = G_alias.alias_node(changed_p_id)['seg']
                        if seg_G_add:
                            if seg_G2.has_node(p_seg) and seg_G2.has_node(changed_p_seg):
                                seg_G2.add_edge(p_seg, changed_p_seg, edge_type='left')
                    break

    elif lane_add < 0 and len(same_direct_seg_list) > 0:
        # change extorior edges
        if edge_rewire and len(same_direct_seg_list) > 1:
            second_out_most_lane_id = same_direct_seg_list[-2]

            start_node = G_alias.alias_node_key(nodes_in_road[out_most_lane_id][0])
            end_node = G_alias.alias_node_key(nodes_in_road[out_most_lane_id][-1])
            new_start_node = G_alias.alias_node_key(nodes_in_road[second_out_most_lane_id][0])
            new_end_node = G_alias.alias_node_key(nodes_in_road[second_out_most_lane_id][-1])

            edges_connected_to_start_node = G_alias.G.in_edges(start_node)
            edges_connected_to_end_node = G_alias.G.out_edges(end_node)

            for n1, n2 in edges_connected_to_start_node:
                edge_direc = G_alias.alias_node(new_start_node)['pos'] - G_alias.alias_node(n1)['pos']
                edge_type = G_alias.alias_edge(n1, n2)['edge_type']
                edge_type_idx = G_alias.alias_edge(n1, n2)['edge_type_idx']
                if G_alias.has_node(n1) and G_alias.has_node(new_start_node):
                    G_alias.add_edge(n1, new_start_node, edge_type=edge_type, edge_type_idx=edge_type_idx,
                                     direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))

                if seg_G_add:
                    n1_seg = G_alias.alias_node(n1)['seg']
                    n2_seg = G_alias.alias_node(n2)['seg']
                    new_start_seg = G_alias.alias_node(new_start_node)['seg']
                    if new_start_seg not in [n1_seg, n2_seg] and seg_G2.has_edge(n1_seg, n2_seg):
                        if seg_G2.has_node(n1_seg) and seg_G2.has_node(new_start_seg):
                            seg_G2.add_edge(n1_seg, new_start_seg, \
                                            edge_type=seg_G2.edges[n1_seg, n2_seg]['edge_type'])

            for n1, n2 in edges_connected_to_end_node:
                edge_direc = G_alias.alias_node(n2)['pos'] - G_alias.alias_node(new_end_node)['pos']
                edge_type = G_alias.alias_edge(n1, n2)['edge_type']
                edge_type_idx = G_alias.alias_edge(n1, n2)['edge_type_idx']
                if G_alias.has_node(n2) and G_alias.has_node(new_end_node):
                    G_alias.add_edge(new_end_node, n2, edge_type=edge_type, edge_type_idx=edge_type_idx,
                                     direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))
                if seg_G_add:
                    n1_seg = G_alias.alias_node(n1)['seg']
                    n2_seg = G_alias.alias_node(n2)['seg']
                    new_end_seg = G_alias.alias_node(new_end_node)['seg']
                    if new_end_seg not in [n1_seg, n2_seg] and seg_G2.has_edge(n1_seg, n2_seg):
                        if seg_G2.has_node(n2_seg) and seg_G2.has_node(new_end_seg):
                            seg_G2.add_edge(new_end_seg, n2_seg, \
                                            edge_type=seg_G2.edges[n1_seg, n2_seg]['edge_type'])

        # remove nodes
        for node in seg_G2.nodes[out_most_lane_id]['point_names']:
            for idx_a in G_alias.main_to_alias_nodes[G_alias.alias_node_key(node)]:
                idx_seg_name = '_'.join(idx_a.split('_')[:-1])
                if idx_seg_name in nodes_in_road.keys():
                    if idx_a in nodes_in_road[idx_seg_name]:
                        nodes_in_road[idx_seg_name].remove(idx_a)
                if idx_seg_name in seg_G2.nodes.keys():
                    if idx_a in seg_G2.nodes[idx_seg_name]['point_names']:
                        seg_G2.nodes[idx_seg_name]['point_names'].remove(idx_a)
            G_alias.remove_node(node)
        if out_most_lane_id in seg_list:
            seg_list.remove(out_most_lane_id)
        if out_most_lane_id in seg_G2.nodes:
            seg_G2.remove_node(out_most_lane_id)

    if hole_add > 0:
        new_hole_candids = set([G_alias.alias_node_key(c) for c in nodes_in_road[seg_name]])
        if hole_candids is not None:
            hole_candids = set([G_alias.alias_node_key(c) for c in hole_candids if G_alias.has_node(c)])
            hole_candids = hole_candids & new_hole_candids

            if len(hole_candids) == 0 and aug_dev == 0 and aug_width == 0 and lane_add == 0:
                assert False, 'hole_add error'
        new_hole_candids = list(new_hole_candids)

        hole_idxs = np.random.choice(new_hole_candids, hole_add, replace=False)
        for idx in hole_idxs:
            for idx_a in G_alias.main_to_alias_nodes[G_alias.alias_node_key(idx)]:
                idx_seg_name = '_'.join(idx_a.split('_')[:-1])
                if idx_seg_name in nodes_in_road.keys():
                    if idx_a in nodes_in_road[idx_seg_name]:
                        nodes_in_road[idx_seg_name].remove(idx_a)
                if idx_seg_name in seg_G2.nodes.keys():
                    if idx_a in seg_G2.nodes[idx_seg_name]['point_names']:
                        point_name_idx = seg_G2.nodes[idx_seg_name]['point_names'].index(idx_a)
                        seg_G2.nodes[idx_seg_name]['point_names'].remove(idx_a)
                        hole_pos = seg_G2.nodes[idx_seg_name]['point_line'][point_name_idx]
                        if 'hole_pos' in seg_G2.nodes[idx_seg_name]:
                            seg_G2.nodes[idx_seg_name]['hole_pos'].append(hole_pos)
                        else:
                            seg_G2.nodes[idx_seg_name]['hole_pos'] = [hole_pos]
            G_alias.remove_node(idx)

    already_moved = []
    # move nodes
    for seg in seg_list:
        for node_name in nodes_in_road[seg]:
            key_node_name = G_alias.alias_node_key(node_name)
            if key_node_name in already_moved:
                continue
            else:
                already_moved.append(key_node_name)

            node_pos = G_alias.alias_node(node_name)['pos'].copy().reshape([1, 2])
            node_point_frenet = cartesian_to_frenet_approx(road_center, node_pos)
            node_point_frenet[:, 1] *= (4. - aug_width) / 4.
            node_point_frenet[:, 1] += aug_dev * (2 * (ref_seg_direc > 0) - 1)
            pos_arrange = frenet_to_cartesian_approx(road_center, node_point_frenet).reshape([2])
            G_alias.alias_node(node_name)['pos'] = pos_arrange

        if seg_G_add:
            left_border = seg_G2.nodes[seg]['left_border'].copy()
            l_point_frenet = cartesian_to_frenet_approx(road_center, left_border)
            l_point_frenet[:, 1] *= (4. - aug_width) / 4.
            l_point_frenet[:, 1] += aug_dev * (2 * (ref_seg_direc > 0) - 1)
            left_border = frenet_to_cartesian_approx(road_center, l_point_frenet)

            right_border = seg_G2.nodes[seg]['right_border'].copy()
            r_point_frenet = cartesian_to_frenet_approx(road_center, right_border)
            r_point_frenet[:, 1] *= (4. - aug_width) / 4.
            r_point_frenet[:, 1] += aug_dev * (2 * (ref_seg_direc > 0) - 1)
            right_border = frenet_to_cartesian_approx(road_center, r_point_frenet)

            seg_road_center = seg_G2.nodes[seg]['road_center'].copy()
            s_point_frenet = cartesian_to_frenet_approx(road_center, seg_road_center)
            s_point_frenet[:, 1] = aug_dev * (2 * (ref_seg_direc > 0) - 1)
            seg_road_center = frenet_to_cartesian_approx(road_center, s_point_frenet)

            seg_G2.nodes[seg]['left_border'] = left_border
            seg_G2.nodes[seg]['right_border'] = right_border
            seg_G2.nodes[seg]['lane_center'] = (left_border + right_border) / 2
            seg_G2.nodes[seg]['road_center'] = seg_road_center

            if 'hole_pos' in seg_G2.nodes[seg]:
                hole_pos = np.stack(seg_G2.nodes[seg]['hole_pos'])
                h_point_frenet = cartesian_to_frenet_approx(road_center, hole_pos)
                h_point_frenet[:, 1] *= (4. - aug_width) / 4.
                h_point_frenet[:, 1] = aug_dev * (2 * (ref_seg_direc > 0) - 1)
                hole_pos = frenet_to_cartesian_approx(road_center, h_point_frenet)
                seg_G2.nodes[seg]['hole_pos'] = [h for h in hole_pos]
    return G_alias.G, seg_G2

def route_augment(G2, seg_G, seg_name_list, hole_candids=None):

    G_arrange = G2.copy()
    seg_G2 = seg_G.copy()

    result = []
    for seg_name in seg_name_list:
        rand_ = np.random.randint(8)
        try:
            if rand_ == 0:  # hole add
                aug_dev = 0; aug_width = 0; lane_add = 0; hole_add = 1
            elif rand_ == 1:  # lane add
                aug_dev = 0; aug_width = 0; lane_add = 1; hole_add = 0
            elif rand_ == 2:  # lane remove
                aug_dev = 0; aug_width = 0; lane_add = -1; hole_add = 0
            elif rand_ == 3:  # aug width
                aug_dev = 0; aug_width = 0; lane_add = 0; hole_add = 0
                aug_width = np.random.uniform(1.0, 1.5) * np.random.choice([1,-1])
            elif rand_ == 4:  # aug deviation
                aug_dev = 0; aug_width = 0; lane_add = 0; hole_add = 0
                aug_dev = np.random.uniform(1.5, 3.0) * np.random.choice([1,-1])
            elif rand_ == 5:  # hode add + combination
                aug_dev = 0; aug_width = 0; lane_add = 0; hole_add = 1
                aug_width = np.random.uniform(1.0, 1.5) * np.random.choice([1,0,-1])
                aug_dev = np.random.uniform(1.5, 3.0) * np.random.choice([1,0,-1])
            elif rand_ == 6:  # lane add + combination
                aug_dev = 0; aug_width = 0; lane_add = 1; hole_add = 0
                aug_width = np.random.uniform(1.0, 1.5) * np.random.choice([1,0,-1])
                aug_dev = np.random.uniform(1.5, 3.0) * np.random.choice([1,0,-1])
            elif rand_ == 7:  # lane remove + combination
                aug_dev = 0; aug_width = 0; lane_add = -1; hole_add = 0
                aug_width = np.random.uniform(1.0, 1.5) * np.random.choice([1,0,-1])
                aug_dev = np.random.uniform(1.5, 3.0) * np.random.choice([1,0,-1])
            else:
                pass

            G_arrange, seg_G2 = segment_augment(G_arrange, seg_G2, seg_name, aug_dev=aug_dev, aug_width=aug_width,
                                                lane_add=lane_add, hole_add=hole_add, hole_candids=hole_candids)
            result.append({'seg_name':seg_name, 'rand_':rand_, 'aug_dev':aug_dev, 'aug_width':aug_width, 'lane_add':lane_add, 'hole_add':hole_add})
        except Exception as e:
            result.append({'seg_name':seg_name, 'error':e, 'rand_':rand_, 'aug_dev':aug_dev, 'aug_width':aug_width, 'lane_add':lane_add, 'hole_add':hole_add})
            #print(seg_name, {'seg_name':seg_name, 'error':e, 'rand_':rand_, 'aug_dev':aug_dev, 'aug_width':aug_width, 'lane_add':lane_add, 'hole_add':hole_add})

    return G_arrange, seg_G2, result

'''
def segment_augment(G, seg_G, seg_name, rot_center, aug_deg=0.0, aug_dev=0.0, aug_width=0.0,
                    lane_add=0, add_to_same_direc=True):
    road_name = seg_name.split('_')[0]
    theta = aug_deg * np.pi / 180

    cos_theta = np.cos(theta).reshape([-1, 1])
    sin_theta = np.sin(theta).reshape([-1, 1])
    R = np.concatenate([cos_theta, sin_theta, -sin_theta, cos_theta], axis=1).reshape([2, 2])

    G_arrange = G.copy()
    seg_list = [key for key, value in seg_G.nodes().items() if value['road'] == road_name]
    nodes_in_road = {seg_name: [] for seg_name in seg_list}
    for key, value in G.nodes().items():
        if value['road'] == road_name:
            nodes_in_road[value['seg']].append(key)
        elif 'contraction' in value:
            for c_key, c_value in value['contraction'].items():
                if c_value['road'] == road_name:
                    # mapping = {key: c_key}
                    # G_arrange = nx.relabel_nodes(G_arrange, mapping)
                    G_arrange.nodes[key]['road'] = c_value['road']
                    G_arrange.nodes[key]['order'] = c_value['order']
                    G_arrange.nodes[key]['seg'] = c_value['seg']
                    G_arrange.nodes[key]['frenet'] = c_value['frenet']
                    G_arrange.nodes[key]['lane_section'] = c_value['lane_section']
                    nodes_in_road[c_value['seg']].append(key)
                    break

    # find nearest rot center
    for seg in seg_list:
        nodes_in_road[seg_name] = sorted(nodes_in_road[seg_name], key=lambda k: G_arrange.nodes[k]['order'])
    ref_seg_direc = int(seg_name.split('_')[-1])
    node1 = nodes_in_road[seg_name][0]
    node2 = nodes_in_road[seg_name][-1]
    if np.sum((node1_pos - rot_center) ** 2) > np.sum((node2_pos - rot_center) ** 2):
        toward_rot_center = True
    else:
        toward_rot_center = False

    road_direc = (node1_pos - node2_pos) if toward_rot_center else (node2_pos - node1_pos)
    road_direc_norm = np.linalg.norm(road_direc)
    road_direc = road_direc / road_direc_norm if road_direc_norm != 0 else road_direc
    road_direc = np.array([-road_direc[1], road_direc[0]])

    # add new lane
    if lane_add > 0:
        if ref_seg_direc > 0:
            out_most_lane_num = max(seg_list, key=lambda k: int(k.split('_')[-1])).split('_')[-1]
        elif ref_seg_direc < 0:
            out_most_lane_num = min(seg_list, key=lambda k: int(k.split('_')[-1])).split('_')[-1]
        out_most_lane_id = road_name + '_' + str(out_most_lane_num)
        new_lane_num = int(out_most_lane_num) + 1. * (ref_seg_direc > 0) - 1. * (ref_seg_direc < 0)
        new_lane_id = road_name + '_' + str(int(new_lane_num))

        road_center = seg_G.nodes[out_most_lane_id]['road_center']
        seg_frenet = seg_G.nodes[out_most_lane_id]['frenet']
        new_seg_frenet = seg_frenet.copy()
        new_seg_frenet[:, 1] += 4
        new_seg_points = frenet_to_cartesian_approx(road_center, new_seg_frenet)

        # add nodes
        added_node = []
        for n, p in enumerate(new_seg_points):
            new_node_name = new_lane_id + "_" + str(n)
            G_arrange.add_node(new_node_name, road=road_name, seg=new_lane_id, pos=new_seg_points[n],
                               order=n, frenet=new_seg_frenet[n], lane_section=None,
                               on_traffic_signal=seg_G.nodes[out_most_lane_id]['on_traffic_signal'])
            added_node.append(new_node_name)
        nodes_in_road[new_lane_id] = added_node
        seg_list.append(new_lane_id)

        # add sequential edges
        for n in range(len(new_seg_points) - 1):
            prev_node_name = new_lane_id + "_" + str(n)
            next_node_name = new_lane_id + "_" + str(n + 1)
            edge_direc = G_arrange.nodes[next_node_name]['pos'] - G_arrange.nodes[prev_node_name]['pos']
            G_arrange.add_edge(prev_node_name, next_node_name, edge_type='sequential',
                               direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))

        # add lane change edges
        lane_change_progress_min_dist, lane_change_progress_max_dist = 0.0, 6.0
        lane_points = nodes_in_road[out_most_lane_id]
        is_inverted = seg_G.nodes[out_most_lane_id]['is_inverted']
        changed_lane_points = [new_lane_id + "_" + str(n) for n in range(len(new_seg_points))]
        changed_lane_idx = 0
        for p_id in lane_points:
            while (changed_lane_idx < len(changed_lane_points)):
                changed_p_id = changed_lane_points[changed_lane_idx]
                if is_inverted:
                    lane_s = -G_arrange.nodes[p_id]['frenet'][0]
                    changed_lane_s = -G_arrange.nodes[changed_p_id]['frenet'][0]
                else:
                    lane_s = G_arrange.nodes[p_id]['frenet'][0]
                    changed_lane_s = G_arrange.nodes[changed_p_id]['frenet'][0]

                if changed_lane_s < lane_s + lane_change_progress_min_dist:
                    changed_lane_idx += 1
                else:
                    if changed_lane_s < lane_s + lane_change_progress_max_dist:
                        edge_direc = G_arrange.nodes[changed_p_id]['pos'] - G_arrange.nodes[p_id]['pos']
                        G_arrange.add_edge(p_id, changed_p_id, edge_type='right',
                                           direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))
                    break
        lane_points = [new_lane_id + "_" + str(n) for n in range(len(new_seg_points))]
        is_inverted = seg_G.nodes[out_most_lane_id]['is_inverted']
        changed_lane_points = nodes_in_road[out_most_lane_id]
        changed_lane_idx = 0
        for p_id in lane_points:
            while (changed_lane_idx < len(changed_lane_points)):
                changed_p_id = changed_lane_points[changed_lane_idx]
                if is_inverted:
                    lane_s = -G_arrange.nodes[p_id]['frenet'][0]
                    changed_lane_s = -G_arrange.nodes[changed_p_id]['frenet'][0]
                else:
                    lane_s = G_arrange.nodes[p_id]['frenet'][0]
                    changed_lane_s = G_arrange.nodes[changed_p_id]['frenet'][0]

                if changed_lane_s < lane_s + lane_change_progress_min_dist:
                    changed_lane_idx += 1
                else:
                    if changed_lane_s < lane_s + lane_change_progress_max_dist:
                        edge_direc = G_arrange.nodes[changed_p_id]['pos'] - G_arrange.nodes[p_id]['pos']
                        G_arrange.add_edge(p_id, changed_p_id, edge_type='left',
                                           direc=edge_direc, weight=float(np.linalg.norm(edge_direc)))
                    break

    elif lane_add < 0:
        pass

    # move nodes
    for seg in seg_list:
        seg_num = int(seg.split('_')[-1])
        same_direction = seg_num * ref_seg_direc > 0
        seg_toward_rot_center = toward_rot_center if same_direction else not toward_rot_center
        if seg_toward_rot_center:
            rot_center_node = nodes_in_road[seg][-1]
        else:
            rot_center_node = nodes_in_road[seg][0]
        rot_center_node_pos = G_arrange.nodes[rot_center_node]['pos']

        for node_name in nodes_in_road[seg]:
            pos_arrange = np.matmul(G_arrange.nodes[node_name]['pos'] - rot_center_node_pos, R)
            pos_arrange += rot_center_node_pos
            pos_arrange += road_direc * aug_dev
            pos_arrange -= (2. * same_direction - 1) * road_direc * aug_width * abs(seg_num)
            G_arrange.nodes[node_name]['pos'] = pos_arrange

    return G_arrange

'''