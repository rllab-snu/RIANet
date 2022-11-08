import math
import numpy as np
#import pymap3d as pm
import networkx as nx
import xml.etree.ElementTree as ET

def get_georeference(xml_root):
    geo_ref_txt = xml_root.find('header/geoReference').text
    geo_ref_dict = {}
    for item in geo_ref_txt.split(' +'):
        option = item.split('=')
        try:
            geo_ref_dict[option[0]] = float(option[1])
        except:
            try:
                geo_ref_dict[option[0]] = option[1]
            except:
                pass

    return geo_ref_dict

def geodetic2enu(latitude, longtitude, altitude, LAT_REF, LON_REF, ALT_REF):
    EARTH_RADIUS_EQUA = 6378137.0
 
    scale = math.cos(LAT_REF * math.pi / 180.0)
    basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * LON_REF
    basey = scale * EARTH_RADIUS_EQUA * math.log(
        math.tan((90.0 + LAT_REF) * math.pi / 360.0))

    x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longtitude - basex
    y = scale * EARTH_RADIUS_EQUA * math.log(
        math.tan((90.0 + latitude) * math.pi / 360.0)) - basey

    return x, y, altitude

def gnss_to_xy(gnss_data, geo_ref_dict=None, use_pm_lib=False):
    # convert to location using earth-curvature radii

    latitude = gnss_data[0]
    longitude = gnss_data[1]
    altitude = gnss_data[2]

    lat_0 = geo_ref_dict['lat_0'] if geo_ref_dict is not None else 0
    lon_0 = geo_ref_dict['lon_0'] if geo_ref_dict is not None else 0

    if use_pm_lib:
        carla_x, carla_y, carla_z = pm.geodetic2enu(latitude, longitude, altitude, \
                                    lat_0, lon_0, 0, \
                                    ell=pm.utils.Ellipsoid('wgs84'))
    else:
        carla_x, carla_y, carla_z = geodetic2enu(latitude, longitude, altitude, \
                                    lat_0, lon_0, 0)

    return np.array([carla_x, carla_y, carla_z])

def get_global_plan_pos(global_plan_gnss, geo_ref_dict=None, use_pm_lib=False):

    lat_0 = geo_ref_dict['lat_0'] if geo_ref_dict is not None else 0
    lon_0 = geo_ref_dict['lon_0'] if geo_ref_dict is not None else 0

    if use_pm_lib:
        global_plan_pos = [pm.geodetic2enu(gnss['lat'], gnss['lon'], gnss['z'], \
                            lat_0, lon_0, 0, \
                            ell=pm.utils.Ellipsoid('wgs84')) for gnss, _ in global_plan_gnss]
    else:
        global_plan_pos = [geodetic2enu(gnss['lat'], gnss['lon'], gnss['z'], \
                            lat_0, lon_0, 0) for gnss, _ in global_plan_gnss]

    global_plan_pos = np.array(global_plan_pos)[:,:2]

    return global_plan_pos

'''
def get_global_traj(global_plan_pos, G, seg_G, pos_kdtree_keys, pos_kdtree):
    # global_plan_pos : w x 2
    _, idxs = pos_kdtree.query(global_plan_pos, k=1)
    nearest_nodes_idxs = pos_kdtree_keys[idxs].reshape(-1)

    # TODO
    # nead to consider lane change cases

    seg_path = []
    for most_nearest_id in nearest_nodes_idxs:
        if not G.nodes[most_nearest_id]['in_junction']:
            waypoint_seg = G.nodes[most_nearest_id]['seg']
            seg_path.append(waypoint_seg)
            
    # find path between seg
    all_shortest_path = dict(nx.all_pairs_shortest_path(seg_G))
    seg_path_candid_ids = []
    for n in range(len(seg_path)-1):
        prev_seg = seg_path[n]
        next_seg = seg_path[n+1]
        short_path = all_shortest_path[prev_seg][next_seg]
        seg_path_candid_ids += short_path
        
    # remove consecutive duplicates
    i=0
    while i < len(seg_path_candid_ids)-1:
        if seg_path_candid_ids[i] == seg_path_candid_ids[i+1]:
            del seg_path_candid_ids[i]
        else:
            i += 1

    i, path_traj = 0, []
    while (i < len(seg_path_candid_ids)):
        if i != len(seg_path_candid_ids) - 1:
            seg1, seg2 = seg_path_candid_ids[i], seg_path_candid_ids[i + 1]

            if seg_G.get_edge_data(seg1, seg2)['edge_type'] in ['left','right']:  # TODO : Need to change condition
                path1 = seg_G.nodes[seg1]['point_line']
                path2 = seg_G.nodes[seg2]['point_line']
                rand_point = np.random.randint(1, 1 + len(path1) / 2)  # select random lane-change point
                path_traj.append(path1[:rand_point])
                change_pos = path1[rand_point]

                dist = []
                for node_pos in path2:
                    dist.append(np.linalg.norm(change_pos - node_pos))  # nearest point
                changed_point = np.argmin(dist) + 2  # 4
                changed_point = min(changed_point, len(path2) - 1)
                path_traj.append(path2[changed_point:])

                i += 2
                continue            

        seg_i = seg_path_candid_ids[i]
        path_traj.append(seg_G.nodes[seg_i]['point_line'])
        i += 1
        
    # remove consecutive duplicates
    path_traj = np.concatenate(path_traj)
    path_filter = np.concatenate([np.array([True]), \
                (np.sum(np.abs(path_traj[1:] - path_traj[:-1]), axis=1) > 0.1)])
    path_traj = path_traj[path_filter]

    return path_traj
'''
# remove consecutive duplicates
def remove_duplicate(path):
    i=0
    while i < len(path)-1:
        if path[i] == path[i+1]:
            del path[i]
        else:
            i += 1
    return path
'''
def find_shortest_seg_path_recur(start_seg, seg_candid_nested_list, all_shortest_path):
    # input : start_seg, [[candid11, candid12], [candid21, candid22], ...], all_shortest_path
    # result : [start seg, candid1, candid2, ---]
    shortest_path = []
    
    if len(seg_candid_nested_list)==0:
        return [start_seg], 1
    
    min_dist = np.inf
    final_shortest_path = []
    for seg in seg_candid_nested_list[0]:        
        try:
            short_path = all_shortest_path[start_seg][seg][:-1]   # without end seg
            short_path_dist = len(short_path)
        except:
            short_path = [start_seg]
            short_path_dist = np.inf

        last_path, last_path_dist = find_shortest_seg_path_recur(seg, seg_candid_nested_list[1:], all_shortest_path)
        dist = short_path_dist + last_path_dist
            
        if dist < min_dist:
            min_dist = dist
            final_shortest_path = short_path + last_path
        
    return remove_duplicate(final_shortest_path), len(final_shortest_path) if dist < np.inf else np.inf

def get_global_traj(global_plan_pos, G, seg_G, pos_kdtree_keys, pos_kdtree, candid_node_num=3, cutoff_dist=None):

    if cutoff_dist is not None:
        pos_dist = global_plan_pos - global_plan_pos[:1]
        pos_dist = pos_dist[:,0]**2 + pos_dist[:,1]**2
        global_plan_pos = global_plan_pos[pos_dist<=cutoff_dist**2,:]
        
    # global_plan_pos : w x 2
    # find path between seg
    dist, idxs = pos_kdtree.query(global_plan_pos, k=candid_node_num)
    nearest_nodes_idxs = pos_kdtree_keys[idxs]

    seg_candid_bound_dist = 2.0
    seg_candid_nested_list = []
    for n, most_nearest_id in enumerate(nearest_nodes_idxs):
        if G.nodes[most_nearest_id[0]]['in_junction']:
            candid_id = most_nearest_id[dist[n] < seg_candid_bound_dist]
            candid_id = most_nearest_id[:1] if len(candid_id) == 0 else candid_id
            seg_candid_nested_list.append([G.nodes[seg_id]['seg'] for seg_id in candid_id])      
        else:        
            seg_candid_nested_list.append([G.nodes[most_nearest_id[0]]['seg']])  
            
    # find path between seg
    all_shortest_path = dict(nx.all_pairs_shortest_path(seg_G))

    seg_path_candid_ids = []
    seg_path_result = []
    for n, seg_candid in enumerate(seg_candid_nested_list):
        seg_path_candid_ids.append(seg_candid)
        if len(seg_candid) == 1 or n == len(seg_candid_nested_list) - 1:
            if len(seg_path_candid_ids) == 1:
                if len(seg_path_result) == 0:
                    seg_path_result += seg_candid
                    seg_path_candid_ids = [] # reset
                else:                
                    start_seg = seg_path_result[-1]
                    try:
                        short_path = all_shortest_path[start_seg][seg_candid[0]]   # without end seg
                    except:
                        short_path = [start_seg, seg_candid[0]]
                    seg_path_result += short_path
                    seg_path_candid_ids = [] # reset
            else:
                if len(seg_path_result) == 0:                
                    min_dist = np.inf
                    final_shortest_path = []
                    for start_seg in seg_path_candid_ids[0]:
                        path, dist = find_shortest_seg_path_recur(start_seg, seg_path_candid_ids[1:], all_shortest_path)
                        if dist < min_dist:
                            min_dist = dist
                            final_shortest_path = path
                    seg_path_result += final_shortest_path    
                    seg_path_candid_ids = [] # reset
                else:
                    start_seg = seg_path_result[-1]
                    path, _ = find_shortest_seg_path_recur(start_seg, seg_path_candid_ids, all_shortest_path)
                    seg_path_result += path   
                    seg_path_candid_ids = [] # reset
    seg_path_result = remove_duplicate(seg_path_result)

    i, path_traj = 0, []
    while (i < len(seg_path_result)):
        if i != len(seg_path_result) - 1:
            seg1, seg2 = seg_path_result[i], seg_path_result[i + 1]

            if seg_G.get_edge_data(seg1, seg2)['edge_type'] in ['left','right']:  # TODO : Need to change condition
                path1 = seg_G.nodes[seg1]['point_line']
                path2 = seg_G.nodes[seg2]['point_line']
                rand_point = np.random.randint(1, 1 + len(path1) / 2)  # select random lane-change point
                path_traj.append(path1[:rand_point])
                change_pos = path1[rand_point]

                dist = []
                for node_pos in path2:
                    dist.append(np.linalg.norm(change_pos - node_pos))  # nearest point
                changed_point = np.argmin(dist) + 2  # 4
                changed_point = min(changed_point, len(path2) - 1)
                path_traj.append(path2[changed_point:])

                i += 2
                continue            

        seg_i = seg_path_result[i]
        path_traj.append(seg_G.nodes[seg_i]['point_line'])
        i += 1

    # remove consecutive duplicates
    path_traj = np.concatenate(path_traj)
    path_filter = np.concatenate([np.array([True]), \
                (np.sum(np.abs(path_traj[1:] - path_traj[:-1]), axis=1) > 0.1)])
    path_traj = path_traj[path_filter]
    
    return path_traj
'''

def find_shortest_path_recur(start_node, candid_nested_list, G):
    # input : start_node, [[candid11, candid12], [candid21, candid22], ...], G
    # result : [start node, candid1, candid2, ---]
    shortest_path = []
    
    if len(candid_nested_list) == 0:
        return [start_node], 0
    
    min_dist = np.inf
    final_shortest_path = []
    for node in candid_nested_list[0]:        
        try:
            short_path = nx.dijkstra_path(G, start_node, node)[:-1]   # without end seg
            short_path_dist = nx.dijkstra_path_length(G, start_node, node)
        except:
            short_path = [start_node]
            short_path_dist = np.inf
            
        last_path, last_path_dist = find_shortest_path_recur(node, candid_nested_list[1:], G)
        dist = short_path_dist + last_path_dist
            
        if dist < min_dist:
            min_dist = dist
            final_shortest_path = short_path + last_path
        
    return remove_duplicate(final_shortest_path), len(final_shortest_path) if dist < np.inf else np.inf

def get_global_traj_recur_method(global_plan_pos, G, seg_G, pos_kdtree_keys, pos_kdtree, \
                     candid_node_num=3, candid_bound_dist=2.0, smoothing_node_len=2, cutoff_dist=None):
    if cutoff_dist is not None:
        pos_dist = global_plan_pos - global_plan_pos[:1]
        pos_dist = pos_dist[:,0]**2 + pos_dist[:,1]**2
        global_plan_pos = global_plan_pos[pos_dist<=cutoff_dist**2,:]
        
    # global_plan_pos : w x 2
    # find path between seg
    dist, idxs = pos_kdtree.query(global_plan_pos, k=candid_node_num)
    nearest_nodes_idxs = pos_kdtree_keys[idxs]

    candid_nested_list = []
    for n, most_nearest_id in enumerate(nearest_nodes_idxs):
        if G.nodes[most_nearest_id[0]]['in_junction']:
            candid_id = most_nearest_id[dist[n] < candid_bound_dist]
            candid_id = most_nearest_id[:1] if len(candid_id) == 0 else candid_id
            candid_nested_list.append(list(candid_id))
        else:        
            candid_nested_list.append([most_nearest_id[0]])

    # find path between seg
    path_candid_ids = []
    path_result = []
    for n, candids in enumerate(candid_nested_list):
        path_candid_ids.append(candids)
        if len(candids) == 1 or n == len(candid_nested_list) - 1:
            if len(path_candid_ids) == 1:
                if len(path_result) == 0:
                    path_result += candids
                    path_candid_ids = [] # reset
                else:                
                    start_node = path_result[-1]
                    try:
                        short_path = nx.dijkstra_path(G, start_node, candids[0])   # without end seg
                    except:
                        short_path = [start_node, candids[0]]
                    path_result += short_path
                    path_candid_ids = [] # reset
            else:
                if len(path_result) == 0:                
                    min_dist = np.inf
                    final_shortest_path = []
                    for start_node in path_candid_ids[0]:
                        path, dist = find_shortest_path_recur(start_node, path_candid_ids[1:], G)
                        if dist < min_dist:
                            min_dist = dist
                            final_shortest_path = path
                    path_result += final_shortest_path
                    path_candid_ids = [] # reset
                else:
                    start_node = path_result[-1]
                    path, _ = find_shortest_path_recur(start_node, path_candid_ids, G)
                    path_result += path
                    path_candid_ids = [] # reset
    path_result = remove_duplicate(path_result)

    # smoothing path
    if smoothing_node_len > 0:
        i, smoothing_path = 0, []
        while i < len(path_result):
            if i <= len(path_result) - 2 * smoothing_node_len - 1:
                idx1 = path_result[i + smoothing_node_len - 1]
                idx2 = path_result[i + smoothing_node_len]
                if G.has_edge(idx1, idx2) and G.edges[idx1, idx2]['edge_type'] in ['left', 'right']:
                    i += 2 * smoothing_node_len
                    continue
            smoothing_path.append(path_result[i])
            i += 1
        path_result = smoothing_path

    # expand path
    most_front_node_seg = G.nodes[path_result[0]]['seg']
    most_front_node_order = G.nodes[path_result[0]]['order']
    front_addition = seg_G.nodes[most_front_node_seg]['point_names'][:most_front_node_order]

    most_end_node_seg = G.nodes[path_result[-1]]['seg']
    most_end_node_order = G.nodes[path_result[-1]]['order']
    end_addition = seg_G.nodes[most_end_node_seg]['point_names'][most_end_node_order+1:]
    
    if cutoff_dist is None:
        path_result = front_addition + path_result + end_addition
    else:
        path_result = front_addition + path_result

    # convert to position array
    path_traj = np.stack([G.nodes[node]['pos'] for node in path_result])

    return path_traj, path_result

def get_global_traj(global_plan_pos, G, seg_G, pos_kdtree_keys, pos_kdtree, \
                        candid_node_num=20, candid_bound_dist=3.0, smoothing_node_len=2,
                        path_ignore_distance_factor=10.0, cutoff_dist=None):
    global_plan_pos = global_plan_pos.copy()
    if cutoff_dist is not None:
        pos_dist = global_plan_pos - global_plan_pos[:1]
        pos_dist = pos_dist[:, 0] ** 2 + pos_dist[:, 1] ** 2
        cutoff_filter = np.cumprod(pos_dist <= cutoff_dist ** 2) == 1  # make False after if once a False value is found
        global_plan_pos = global_plan_pos[cutoff_filter, :]

    tmp_G = G.copy()
    # global_plan_pos : w x 2
    # find path between seg
    dist, idxs = pos_kdtree.query(global_plan_pos, k=candid_node_num)
    nearest_nodes_idxs = pos_kdtree_keys[idxs]

    tmp_node_id_list = []
    for n, most_nearest_id in enumerate(nearest_nodes_idxs):
        tmp_node_id = 'tmp_node_{}'.format(n)
        tmp_G.add_node(tmp_node_id, pos=global_plan_pos[n])
        tmp_node_id_list.append(tmp_node_id)

        candid_id = most_nearest_id[dist[n] < candid_bound_dist]
        candid_id = most_nearest_id[:1] if len(candid_id) == 0 else candid_id
        for node_id in candid_id:
            edge_direc = tmp_G.nodes[node_id]['pos'] - tmp_G.nodes[tmp_node_id]['pos']
            edge_weight = float(np.linalg.norm(edge_direc))
            tmp_G.add_edge(tmp_node_id, node_id, edge_type='tmp', direc=edge_direc, weight=edge_weight)
            tmp_G.add_edge(node_id, tmp_node_id, edge_type='tmp', direc=-edge_direc, weight=edge_weight)

    path_result = []
    prev_tmp_node_id = tmp_node_id_list[0]
    for tmp_node_id in tmp_node_id_list[1:]:
        try:
            short_path = nx.dijkstra_path(tmp_G, prev_tmp_node_id, tmp_node_id)
            short_path_len = nx.dijkstra_path_length(tmp_G, prev_tmp_node_id, tmp_node_id)
        except:
            short_path = None
            short_path_len = np.inf

        l2_dist = np.sqrt(np.sum((tmp_G.nodes[prev_tmp_node_id]['pos'] - tmp_G.nodes[tmp_node_id]['pos'])**2))
        if short_path_len < path_ignore_distance_factor * l2_dist:  # if a path was found
            for node in short_path:
                if node not in tmp_node_id_list:   # ignore tmp node
                    path_result.append(node)
            prev_tmp_node_id = tmp_node_id

    path_result = remove_duplicate(path_result)

    # smoothing path
    if smoothing_node_len > 0:
        i, smoothing_path = 0, []
        while i < len(path_result):
            if i <= len(path_result) - 2 * smoothing_node_len - 1:
                idx1 = path_result[i + smoothing_node_len - 1]
                idx2 = path_result[i + smoothing_node_len]
                if G.has_edge(idx1, idx2) and G.edges[idx1, idx2]['edge_type'] in ['left', 'right']:
                    i += 2 * smoothing_node_len
                    continue
            smoothing_path.append(path_result[i])
            i += 1
        path_result = smoothing_path

    # expand path
    most_front_node_seg = G.nodes[path_result[0]]['seg']
    most_front_node_order = G.nodes[path_result[0]]['order']
    front_addition = seg_G.nodes[most_front_node_seg]['point_names'][:most_front_node_order]

    most_end_node_seg = G.nodes[path_result[-1]]['seg']
    most_end_node_order = G.nodes[path_result[-1]]['order']
    end_addition = seg_G.nodes[most_end_node_seg]['point_names'][most_end_node_order + 1:]

    plan_start_idx = len(front_addition)
    plan_end_idx = plan_start_idx + len(path_result) - 1
    if cutoff_dist is None:
        path_result = front_addition + path_result + end_addition
    else:
        path_result = front_addition + path_result

    path_result = [node for node in path_result if G.has_node(node)]

    # convert to position array
    path_traj = np.stack([G.nodes[node]['pos'] for node in path_result if G.has_node(node)])

    # get progress info
    direc_vec = np.diff(path_traj, axis=0)
    line_len = np.hypot(direc_vec[:, 0], direc_vec[:, 1])
    line_len = np.concatenate([np.zeros(1), line_len])
    path_line_len = np.cumsum(line_len)

    return path_traj, path_result, path_line_len, (plan_start_idx, plan_end_idx)

def get_global_traj_with_error(global_plan_pos, G, seg_G, pos_kdtree_keys, pos_kdtree, G_origin, seg_G_origin, \
                        candid_node_num=20, candid_bound_dist=3.0, smoothing_node_len=2,
                        path_ignore_distance_factor=10.0, cutoff_dist=None):
    global_plan_pos = global_plan_pos.copy()
    if cutoff_dist is not None:
        pos_dist = global_plan_pos - global_plan_pos[:1]
        pos_dist = pos_dist[:, 0] ** 2 + pos_dist[:, 1] ** 2
        cutoff_filter = np.cumprod(pos_dist <= cutoff_dist ** 2) == 1  # make False after if once a False value is found
        global_plan_pos = global_plan_pos[cutoff_filter, :]

    tmp_G = G.copy()
    # global_plan_pos : w x 2
    # find path between seg
    dist, idxs = pos_kdtree.query(global_plan_pos, k=candid_node_num)
    nearest_nodes_idxs = pos_kdtree_keys[idxs]

    tmp_node_id_list = []
    for n, most_nearest_id in enumerate(nearest_nodes_idxs):
        tmp_node_id = 'tmp_node_{}'.format(n)
        tmp_G.add_node(tmp_node_id, pos=global_plan_pos[n])
        tmp_node_id_list.append(tmp_node_id)

        candid_id = most_nearest_id[dist[n] < candid_bound_dist]
        candid_id = most_nearest_id[:1] if len(candid_id) == 0 else candid_id
        for node_id in candid_id:
            edge_direc = tmp_G.nodes[node_id]['pos'] - tmp_G.nodes[tmp_node_id]['pos']
            edge_weight = float(np.linalg.norm(edge_direc))
            tmp_G.add_edge(tmp_node_id, node_id, edge_type='tmp', direc=edge_direc, weight=edge_weight)
            tmp_G.add_edge(node_id, tmp_node_id, edge_type='tmp', direc=-edge_direc, weight=edge_weight)

    path_result = []
    prev_tmp_node_id = tmp_node_id_list[0]
    for tmp_node_id in tmp_node_id_list[1:]:
        try:
            short_path = nx.dijkstra_path(tmp_G, prev_tmp_node_id, tmp_node_id)
            short_path_len = nx.dijkstra_path_length(tmp_G, prev_tmp_node_id, tmp_node_id)
        except:
            short_path = None
            short_path_len = np.inf

        l2_dist = np.sqrt(np.sum((tmp_G.nodes[prev_tmp_node_id]['pos'] - tmp_G.nodes[tmp_node_id]['pos'])**2))
        if short_path_len < path_ignore_distance_factor * l2_dist:  # if a path was found
            for node in short_path:
                if node not in tmp_node_id_list:   # ignore tmp node
                    path_result.append(node)
            prev_tmp_node_id = tmp_node_id

    path_result = remove_duplicate(path_result)

    # smoothing path
    if smoothing_node_len > 0:
        i, smoothing_path = 0, []
        while i < len(path_result):
            if i <= len(path_result) - 2 * smoothing_node_len - 1:
                idx1 = path_result[i + smoothing_node_len - 1]
                idx2 = path_result[i + smoothing_node_len]
                if G.has_edge(idx1, idx2) and G.edges[idx1, idx2]['edge_type'] in ['left', 'right']:
                    i += 2 * smoothing_node_len
                    continue
            smoothing_path.append(path_result[i])
            i += 1
        path_result = smoothing_path

    # expand path
    most_front_node_seg = G.nodes[path_result[0]]['seg'] if G.has_node(path_result[0]) else G_origin.nodes[path_result[0]]['seg']
    most_front_node_order = G.nodes[path_result[0]]['order'] if G.has_node(path_result[0]) else G_origin.nodes[path_result[0]]['order']
    front_addition = seg_G.nodes[most_front_node_seg]['point_names'][:most_front_node_order] \
        if seg_G.has_node(most_front_node_seg) else seg_G_origin.nodes[most_front_node_seg]['point_names'][:most_front_node_order]

    most_end_node_seg = G.nodes[path_result[-1]]['seg'] if G.has_node(path_result[-1]) else G_origin.nodes[path_result[-1]]['seg']
    most_end_node_order = G.nodes[path_result[-1]]['order'] if G.has_node(path_result[-1]) else G_origin.nodes[path_result[-1]]['order']
    end_addition = seg_G.nodes[most_end_node_seg]['point_names'][most_end_node_order + 1:] \
        if seg_G.has_node(most_end_node_seg) else seg_G_origin.nodes[most_end_node_seg]['point_names'][most_end_node_order + 1:]

    plan_start_idx = len(front_addition)
    plan_end_idx = plan_start_idx + len(path_result) - 1
    if cutoff_dist is None:
        path_result = front_addition + path_result + end_addition
    else:
        path_result = front_addition + path_result

    path_result = [node for node in path_result if G.has_node(node) or G_origin.has_node(node)]

    # convert to position array
    path_traj = np.stack([G.nodes[node]['pos'] if G.has_node(node) else G_origin.nodes[node]['pos'] \
                          for node in path_result if G.has_node(node) or G_origin.has_node(node)])

    # get progress info
    direc_vec = np.diff(path_traj, axis=0)
    line_len = np.hypot(direc_vec[:, 0], direc_vec[:, 1])
    line_len = np.concatenate([np.zeros(1), line_len])
    path_line_len = np.cumsum(line_len)

    return path_traj, path_result, path_line_len, (plan_start_idx, plan_end_idx)

def get_current_partial_traj(prev_st, path_traj, path_progress, margin_before=20, margin_after=50):
    prev_progress = prev_st[0]

    cond1 = path_progress >= prev_progress - margin_before
    cond2 = path_progress < prev_progress + margin_after
    margin_filter = np.logical_and(cond1, cond2)

    partial_traj = path_traj[margin_filter]
    start_progress = path_progress[margin_filter][0]   # how far from the initial pos

    return partial_traj, start_progress

'''
def build_graph_for_path(topology):

    """
    Accessor for topology.
    This function retrieves topology from the server as a list of
    road segments as pairs of waypoint objects, and processes the
    topology into a list of dictionary objects.

        :return topology: list of dictionary objects with the following attributes
            entry   -   waypoint of entry point of road segment
            entryxyz-   (x,y,z) of entry point of road segment
            exit    -   waypoint of exit point of road segment
            exitxyz -   (x,y,z) of exit point of road segment
            path    -   list of waypoints separated by 1m from entry
                        to exit
    """
    world = self.client.load_world(town)
    wmap = world.get_map()
    sampling_resolution = 1.0
    
    topology = []
    # Retrieving waypoints to construct a detailed topology
    for segment in wmap.get_topology():
        wp1, wp2 = segment[0], segment[1]
        l1, l2 = wp1.transform.location, wp2.transform.location
        # Rounding off to avoid floating point imprecision
        x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
        wp1.transform.location, wp2.transform.location = l1, l2
        seg_dict = dict()
        seg_dict['entry'], seg_dict['exit'] = wp1, wp2
        seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
        seg_dict['path'] = []
        endloc = wp2.transform.location
        if wp1.transform.location.distance(endloc) > self._sampling_resolution:
            w = wp1.next(sampling_resolution)[0]
            while w.transform.location.distance(endloc) > self._sampling_resolution:
                seg_dict['path'].append(w)
                w = w.next(sampling_resolution)[0]
        else:
            seg_dict['path'].append(wp1.next(self._sampling_resolution)[0])
        topology.append(seg_dict)

    """
    This function builds a networkx graph representation of topology.
    The topology is read from self._topology.
    graph node properties:
        vertex   -   (x,y,z) position in world map
    graph edge properties:
        entry_vector    -   unit vector along tangent at entry point
        exit_vector     -   unit vector along tangent at exit point
        net_vector      -   unit vector of the chord from entry to exit
        intersection    -   boolean indicating if the edge belongs to an
                            intersection
    return      :   graph -> networkx graph representing the world map,
                    id_map-> mapping from (x,y,z) to node id
                    road_id_to_edge-> map from road id to edge in the graph
    """
    graph = nx.DiGraph()
    id_map = dict()  # Map with structure {(x,y,z): id, ... }
    road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }

    for segment in topology:

        entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
        path = segment['path']
        entry_wp, exit_wp = segment['entry'], segment['exit']
        intersection = entry_wp.is_junction
        road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

        for vertex in entry_xyz, exit_xyz:
            # Adding unique nodes and populating id_map
            if vertex not in id_map:
                new_id = len(id_map)
                id_map[vertex] = new_id
                graph.add_node(new_id, vertex=vertex)
        n1 = id_map[entry_xyz]
        n2 = id_map[exit_xyz]
        if road_id not in road_id_to_edge:
            road_id_to_edge[road_id] = dict()
        if section_id not in road_id_to_edge[road_id]:
            road_id_to_edge[road_id][section_id] = dict()
        road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

        entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
        exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

        # Adding edge with attributes
        graph.add_edge(
            n1, n2,
            length=len(path) + 1, path=path,
            entry_waypoint=entry_wp, exit_waypoint=exit_wp,
            entry_vector=np.array(
                [entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
            exit_vector=np.array(
                [exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
            net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
            intersection=intersection, type=RoadOption.LANEFOLLOW)

    return graph, id_map, road_id_to_edge
'''
