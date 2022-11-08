import numpy as np
import torch
import math

import networkx as nx
import xml.etree.ElementTree as ET

from bisect import *
from sklearn.neighbors import KDTree
from scipy.interpolate import interp1d

def get_traj_len(line_center_xy):
    direc_vec = np.diff(line_center_xy, axis=0)
    line_len = np.hypot(direc_vec[:, 0], direc_vec[:, 1])

    return line_len.sum()

def cartesian_to_frenet_approx(line_center_xy, target_xy):
    # target_xy : t x 2, line_center_xy : l x 2
    direc_vec = np.diff(line_center_xy, axis=0)
    line_len = np.hypot(direc_vec[:, 0], direc_vec[:, 1])

    line_center_xy = line_center_xy[np.concatenate([np.array([True]), line_len > 0])]
    direc_vec = direc_vec[line_len > 0]
    line_len = line_len[line_len > 0]

    direc_vec_norm = np.divide(direc_vec, line_len.reshape(-1, 1))
    point_vec = np.expand_dims(target_xy, 1) - np.expand_dims(line_center_xy[:-1], 0)

    dot_product = direc_vec_norm[:, 0] * point_vec[:, :, 0] + direc_vec_norm[:, 1] * point_vec[:, :, 1]
    scale = np.clip(dot_product, 0, line_len)

    nearest_point = np.expand_dims(direc_vec_norm, 0) * np.expand_dims(scale, 2)
    sq_dist = np.sum((point_vec - nearest_point) ** 2, axis=2)

    midx = np.argmin(sq_dist, axis=1)

    line_len_cumsum = np.cumsum(np.concatenate([np.zeros(1), line_len]))
    run_length = line_len_cumsum[midx] + scale[np.arange(len(midx)), midx]
    sign_dist = direc_vec_norm[midx, 0] * point_vec[np.arange(len(midx)), midx, 1] \
                - direc_vec_norm[midx, 1] * point_vec[np.arange(len(midx)), midx, 0]

    run_length = (run_length == 0) * dot_product[:, 0] + \
                 (run_length == line_len_cumsum[-1]) * (line_len_cumsum[-2] + dot_product[:, -1]) + \
                 ((run_length > 0) * (run_length < line_len_cumsum[-1])) * run_length

    frenet_coord = np.stack([run_length, sign_dist]).T

    return frenet_coord

def frenet_to_cartesian_approx(line_center_xy, target_st):
    # target_st : t x 2, line_center_xy : l x 2
    run_length, sign_dist = target_st[:, :1], target_st[:, 1:]
    direc_vec = np.diff(line_center_xy, axis=0)
    line_len = np.hypot(direc_vec[:, 0], direc_vec[:, 1])

    line_center_xy = line_center_xy[np.concatenate([np.array([True]), line_len > 0])]
    direc_vec = direc_vec[line_len > 0]
    line_len = line_len[line_len > 0]

    direc_vec_norm = np.divide(direc_vec, line_len.reshape(-1, 1))
    norm_vec = np.stack([-direc_vec[:, 1], direc_vec[:, 0]], 1) / line_len.reshape(-1, 1)

    line_len_cumsum = np.cumsum(line_len)
    
    point_vec1 = line_center_xy[0] + direc_vec_norm[0] * run_length + norm_vec[0] * sign_dist
    point_vec2 = line_center_xy[-1] + direc_vec_norm[-1] * (run_length - line_len_cumsum[-1]) + norm_vec[-1] * sign_dist

    midx = np.sum(line_len_cumsum < run_length, axis=1) * (run_length.squeeze() <= line_len_cumsum[-1])
    progress = (run_length.squeeze() - line_len_cumsum[midx - 1]) * (midx > 0) + run_length.squeeze() * (midx == 0)
    point_vec3 = line_center_xy[midx] + direc_vec_norm[midx] * np.expand_dims(progress, 1) + norm_vec[midx] * sign_dist

    point_vec = (run_length < 0) * point_vec1 + \
                (run_length > line_len_cumsum[-1]) * point_vec2 + \
                (run_length >= 0) * (run_length <= line_len_cumsum[-1]) * point_vec3

    return point_vec

def frenet_to_cartesian_with_geometry(road_reference_traj, st_list):
    # st : set of st coordinate (n x 2)
    traj_list = []
    for geo in road_reference_traj:
        new_dict = {'s':float(geo.get('s')), 'x':float(geo.get('x')), 'y':float(geo.get('y')), \
                   'hdg':float(geo.get('hdg')), 'length':float(geo.get('length')), \
                    'type':'arc' if geo.find('arc') is not None else 'line', \
                   'curvature':float(geo.find('arc').get('curvature')) if geo.find('arc') is not None else 0.0}
        traj_list.append(new_dict)
        
    xy_list = []
    refer_progress = 0
    for n, st in enumerate(st_list):
        s, t = st[0], st[1]
        while True:
            if refer_progress == len(traj_list) - 1:
                break
            elif s >= float(traj_list[refer_progress + 1].get('s')): 
                refer_progress += 1
            else:
                break

        traj_part = traj_list[refer_progress]
        start_s = traj_part['s']
        start_x = traj_part['x']
        start_y = traj_part['y']
        hdg = traj_part['hdg']
        traj_len = traj_part['length']
        traj_type = traj_part['type']
        curvature = traj_part['curvature']
        
        if traj_type == 'line':
            ds = (s - start_s)
            x = start_x + ds * math.cos(hdg) - t * math.sin(hdg)
            y = start_y + ds * math.sin(hdg) + t * math.cos(hdg)
        else:  # arc
            ds = (s - start_s)
            c_inv = 1. / curvature
            phi = curvature * ds
            x = start_x + (c_inv - t) * math.sin(hdg + phi) - c_inv * math.sin(hdg)
            y = start_y - (c_inv - t) * math.cos(hdg + phi) + c_inv * math.cos(hdg)
        
        xy_list.append([x,y])
        
    xy_list = np.asarray(xy_list)
    
    return xy_list    

def calculate_line_offset(lane_offset, ref_line_s):
    # lane_offset can be <laneOffset> or <width>
    lane_offset_as_dict_list = []
    for l in lane_offset:  
        s = float(l.get('s')) if l.get('s') is not None else float(l.get('sOffset'))
        a = float(l.get('a'))
        b = float(l.get('b'))
        c = float(l.get('c'))
        d = float(l.get('d'))
        lane_offset_as_dict_list.append({'s':s,'a':a,'b':b,'c':c,'d':d})
    
    refer_progress = 0
    line_t = []
    for s in ref_line_s:
        while True:
            if refer_progress == len(lane_offset) - 1:
                break
            elif s >= lane_offset_as_dict_list[refer_progress + 1]['s']: 
                refer_progress += 1
            else:
                break

        offset = lane_offset_as_dict_list[refer_progress]
        a, b, c, d = offset['a'], offset['b'], offset['c'], offset['d']
        ds = s - offset['s']
        t = a + b*ds + c*ds**2 + d*ds**3
        line_t.append(t)
    
    line_t = np.array(line_t)
    line_st = np.stack([ref_line_s, line_t], axis=1)
    
    return line_st

def calculate_roadmark_offset(roadmark_offset, ref_line_s):
    # lane_offset can be <laneOffset> or <width>
    lane_offset_as_dict_list = []
    for l in roadmark_offset:
        s = float(l.get('s')) if l.get('s') is not None else float(l.get('sOffset'))
        width = l.get('width')
        color = l.get('color')
        m_type = l.get('type')
        lane_offset_as_dict_list.append({'s':s, 'width': width, 'color': color, 'm_type': m_type})

    refer_progress = 0
    line_roadmark = []
    for s in ref_line_s:
        while True:
            if refer_progress == len(roadmark_offset) - 1:
                break
            elif s >= lane_offset_as_dict_list[refer_progress + 1]['s']:
                refer_progress += 1
            else:
                break

        offset = lane_offset_as_dict_list[refer_progress]
        width, color, m_type = str(offset['width']), str(offset['color']), str(offset['m_type'])
        m_str = width + '_' + color + '_' + m_type
        line_roadmark.append(m_str)

    line_roadmark = np.array(line_roadmark)

    return line_roadmark

def extract_lane_center(road, min_ds=0.1):
    lane_offset = road.findall('lanes/laneOffset')
    road_name = "Road " + road.get('id')
    road_ref_traj_len = float(road.get('length'))
    road_reference_traj = road.findall('planView/geometry')
    road_reference_traj.sort(key = lambda geo: float(geo.get('s')))

    lane_section = road.findall('lanes/laneSection')
    lane_section.sort(key = lambda ls: float(ls.get('s')))
    lane_section_s = [float(ls.get('s')) for ls in lane_section]

    lane_offset = road.findall('lanes/laneOffset')
    lane_offset.sort(key = lambda offset: float(offset.get('s')))

    # get centerline
    if road_ref_traj_len <= min_ds:
        ref_line_s = np.arange(0, road_ref_traj_len, 0.999 * road_ref_traj_len)
    else:
        ref_line_s = np.arange(0, road_ref_traj_len, min_ds)

    centerline_st = calculate_line_offset(lane_offset, ref_line_s)
    centerline_xy = frenet_to_cartesian_with_geometry(road_reference_traj, centerline_st)

    # find all lane
    left_lane_ids, right_lane_ids = [], []
    left_lane_info_dict, right_lane_info_dict = {}, {}
    left_lane_xy_dict, right_lane_xy_dict = {}, {}
    left_lane_width_dict, right_lane_width_dict = {}, {}
    left_roadmark_dict, right_roadmark_dict = {}, {}
    left_lane_is_drivable_dict, right_lane_is_drivable_dict = {}, {}
    for section in lane_section:
        left_lanes = list(section.find('left')) if section.find('left') is not None else []
        right_lanes = list(section.find('right')) if section.find('right') is not None else []

        for lane in left_lanes:
            if lane.get('id') not in left_lane_ids:
                left_lane_xy_dict[lane.get('id')] = []
                left_lane_width_dict[lane.get('id')] = []
                left_roadmark_dict[lane.get('id')] = []
                left_lane_is_drivable_dict[lane.get('id')] = []
                if lane.find('userData/vectorLane') is not None and \
                        lane.find('userData/vectorLane').get('travelDir') == 'forward':
                    is_inverted = False
                else:
                    is_inverted = True  # default
                section_lane_type = {float(section.get('s')): lane.get('type')}
                left_lane_info_dict[lane.get('id')] = {'section_lane_type': section_lane_type,
                                                       'is_inverted': is_inverted}
                left_lane_ids.append(lane.get('id'))
            else:
                left_lane_info_dict[lane.get('id')]['section_lane_type'][float(section.get('s'))] = lane.get('type')

        for lane in right_lanes:
            if lane.get('id') not in right_lane_ids:
                right_lane_xy_dict[lane.get('id')] = []
                right_lane_width_dict[lane.get('id')] = []
                right_roadmark_dict[lane.get('id')] = []
                right_lane_is_drivable_dict[lane.get('id')] = []
                if lane.find('userData/vectorLane') is not None and \
                        lane.find('userData/vectorLane').get('travelDir') == 'backward':
                    is_inverted = True
                else:
                    is_inverted = False  # default
                section_lane_type = {float(section.get('s')): lane.get('type')}
                right_lane_info_dict[lane.get('id')] = {'section_lane_type': section_lane_type,
                                                        'is_inverted': is_inverted}
                right_lane_ids.append(lane.get('id'))
            else:
                right_lane_info_dict[lane.get('id')]['section_lane_type'][float(section.get('s'))] = lane.get('type')
                
    left_lane_ids.sort(key = lambda ls: abs(int(ls)))
    right_lane_ids.sort(key = lambda ls: abs(int(ls)))

    centerline_xy_cumulated = []
    centerline_roadmark_cumulated = []

    # get line info
    for n, section in enumerate(lane_section):
        left_lanes = list(section.find('left')) if section.find('left') is not None else [] 
        right_lanes = list(section.find('right')) if section.find('right') is not None else []    
        current_section_left_lanes = {lane.get('id') : lane for lane in left_lanes}
        current_section_right_lanes = {lane.get('id') : lane for lane in right_lanes}
            
        if n == len(lane_section) - 1:
            section_len = road_ref_traj_len - float(section.get('s'))
        else:
            section_len = float(lane_section[n+1].get('s')) - float(section.get('s'))

        if section_len <= min_ds:
            section_ref_line_s = np.arange(0, section_len, 0.999 * section_len)
        else:
            section_ref_line_s = np.arange(0, section_len, min_ds)
        
        cumulated_s = section_ref_line_s.copy()
        cumulated_s += float(section.get('s'))

        centerline_st_append = centerline_st.copy()
        centerline_st_append = np.concatenate([centerline_st_append[:1], \
                                               centerline_st_append, centerline_st_append[-1:]])
        centerline_st_append[0,0] -= 1
        centerline_st_append[-1,0] += 1
        width_intp_f = interp1d(centerline_st_append[:,0], centerline_st_append[:,1], kind='linear')
        cumulated_t = width_intp_f(cumulated_s)  # x : start_x ~ end_x, y: corresponding centerline t
        cumulated = np.stack([cumulated_s, cumulated_t], axis=1)
                
        centerline_xy_cumulated.append(frenet_to_cartesian_with_geometry(road_reference_traj, cumulated))

        center_lane = section.find('center')[0] if section.find('center') is not None else None
        if center_lane is None:
            centerline_roadmark = np.array(['None_None_None'] * len(section_ref_line_s))
        else:
            centerline_roadmark = center_lane.findall('roadMark')
            if len(centerline_roadmark) == 0:
                centerline_roadmark = np.array(['None_None_None'] * len(section_ref_line_s))
            else:
                centerline_roadmark.sort(key=lambda roadmark: float(roadmark.get('sOffset')))
                centerline_roadmark = calculate_roadmark_offset(centerline_roadmark, section_ref_line_s)
        centerline_roadmark_cumulated.append(centerline_roadmark)

        # get left lines
        cumulated_width_st_left = cumulated.copy()
        for lane_id in left_lane_ids:
            if lane_id in current_section_left_lanes.keys():
                lane = current_section_left_lanes[lane_id]
                # calculate left line frenet traj
                lane_width = lane.findall('width')
                lane_width.sort(key = lambda width: float(width.get('sOffset')))
                lane_width_st = calculate_line_offset(lane_width, section_ref_line_s)
                lane_width_st[:,0] += float(section.get('s'))  # add offset

                cumulated_width_st_left[:,1] += lane_width_st[:,1]
                cumulated_width_xy_left = frenet_to_cartesian_with_geometry(road_reference_traj, cumulated_width_st_left)

                left_lane_xy_dict[lane_id].append(cumulated_width_xy_left)
                left_lane_width_dict[lane_id].append(lane_width_st)

                lane_roadmark = lane.findall('roadMark')
                if len(lane_roadmark) == 0:
                    lane_roadmark = np.array(['None_None_None'] * len(section_ref_line_s))
                else:
                    lane_roadmark.sort(key = lambda roadmark: float(roadmark.get('sOffset')))
                    lane_roadmark = calculate_roadmark_offset(lane_roadmark, section_ref_line_s)
                left_roadmark_dict[lane_id].append(lane_roadmark)

                drivable_info = np.ones(len(lane_width_st)) if lane.get('type') == 'driving' else np.zeros(len(lane_width_st))
                left_lane_is_drivable_dict[lane_id].append(drivable_info)
            else:
                lane_width_st = np.stack([section_ref_line_s, np.zeros_like(section_ref_line_s)], axis=1)
                lane_width_st[:,0] += float(section.get('s'))  # add offset
                cumulated_width_xy_left = frenet_to_cartesian_with_geometry(road_reference_traj, cumulated_width_st_left)
                left_lane_xy_dict[lane_id].append(cumulated_width_xy_left)
                left_lane_width_dict[lane_id].append(lane_width_st)

                lane_roadmark = lane.findall('roadMark')
                if len(lane_roadmark) == 0:
                    lane_roadmark = np.array(['None_None_None'] * len(section_ref_line_s))
                else:
                    lane_roadmark.sort(key = lambda roadmark: float(roadmark.get('sOffset')))
                    lane_roadmark = calculate_roadmark_offset(lane_roadmark, section_ref_line_s)
                left_roadmark_dict[lane_id].append(lane_roadmark)

                drivable_info = np.zeros(len(lane_width_st))
                left_lane_is_drivable_dict[lane_id].append(drivable_info)

        # get right lines
        cumulated_width_st_right = cumulated.copy()
        for lane_id in right_lane_ids:
            if lane_id in current_section_right_lanes.keys():
                lane = current_section_right_lanes[lane_id]
                # calculate left line frenet traj
                lane_width = lane.findall('width')
                lane_width.sort(key = lambda width: float(width.get('sOffset')))
                lane_width_st = calculate_line_offset(lane_width, section_ref_line_s)
                lane_width_st[:,0] += float(section.get('s'))  # add offset

                cumulated_width_st_right[:,1] -= lane_width_st[:,1]
                cumulated_width_xy_right = frenet_to_cartesian_with_geometry(road_reference_traj, cumulated_width_st_right)

                right_lane_xy_dict[lane_id].append(cumulated_width_xy_right)
                right_lane_width_dict[lane_id].append(lane_width_st)

                lane_roadmark = lane.findall('roadMark')
                if len(lane_roadmark) == 0:
                    lane_roadmark = np.array(['None_None_None'] * len(section_ref_line_s))
                else:
                    lane_roadmark.sort(key = lambda roadmark: float(roadmark.get('sOffset')))
                    lane_roadmark = calculate_roadmark_offset(lane_roadmark, section_ref_line_s)
                right_roadmark_dict[lane_id].append(lane_roadmark)

                drivable_info = np.ones(len(lane_width_st)) if lane.get('type') == 'driving' else np.zeros(len(lane_width_st))
                right_lane_is_drivable_dict[lane_id].append(drivable_info)
            else:
                lane_width_st = np.stack([section_ref_line_s, np.zeros_like(section_ref_line_s)], axis=1)
                lane_width_st[:,0] += float(section.get('s'))  # add offset
                cumulated_width_xy_right = frenet_to_cartesian_with_geometry(road_reference_traj, cumulated_width_st_right)
                right_lane_xy_dict[lane_id].append(cumulated_width_xy_right)
                right_lane_width_dict[lane_id].append(lane_width_st)

                lane_roadmark = lane.findall('roadMark')
                if len(lane_roadmark) == 0:
                    lane_roadmark = np.array(['None_None_None'] * len(section_ref_line_s))
                else:
                    lane_roadmark.sort(key = lambda roadmark: float(roadmark.get('sOffset')))
                    lane_roadmark = calculate_roadmark_offset(lane_roadmark, section_ref_line_s)
                right_roadmark_dict[lane_id].append(lane_roadmark)

                drivable_info = np.zeros(len(lane_width_st))
                right_lane_is_drivable_dict[lane_id].append(drivable_info)

    centerline_xy_cumulated = np.concatenate(centerline_xy_cumulated)
    centerline_roadmark_cumulated = np.concatenate(centerline_roadmark_cumulated)
    for lane_id in left_lane_ids:
        left_lane_xy_dict[lane_id] = np.concatenate(left_lane_xy_dict[lane_id])
        left_lane_width_dict[lane_id] = np.concatenate(left_lane_width_dict[lane_id])
        left_lane_is_drivable_dict[lane_id] = np.concatenate(left_lane_is_drivable_dict[lane_id])
        left_roadmark_dict[lane_id] = np.concatenate(left_roadmark_dict[lane_id])
    for lane_id in right_lane_ids:
        right_lane_xy_dict[lane_id] = np.concatenate(right_lane_xy_dict[lane_id])
        right_lane_width_dict[lane_id] = np.concatenate(right_lane_width_dict[lane_id])
        right_lane_is_drivable_dict[lane_id] = np.concatenate(right_lane_is_drivable_dict[lane_id])
        right_roadmark_dict[lane_id] = np.concatenate(right_roadmark_dict[lane_id])

    # calcualte each lane center
    lane_center_dict_list = []

    for n, lane_id in enumerate(left_lane_ids):
        left_lane_info = left_lane_info_dict[lane_id]
        is_drivable = any([lane_type == 'driving' for lane_type in left_lane_info['section_lane_type'].values()])
        if is_drivable:
            left_border = left_lane_xy_dict[left_lane_ids[n]]
            right_border = left_lane_xy_dict[left_lane_ids[n-1]] if n > 0 else centerline_xy_cumulated

            lane_center = 0.5 * (left_border + right_border)
            lane_width = left_lane_width_dict[lane_id]
            lane_roadmark = left_roadmark_dict[lane_id]
            lane_drivable = left_lane_is_drivable_dict[lane_id]
            if left_lane_info['is_inverted']:
                lane_center = np.flip(lane_center, axis=0)
                lane_width = np.flip(lane_width, axis=0)
                lane_roadmark = np.flip(lane_roadmark, axis=0)
                lane_drivable = np.flip(lane_drivable, axis=0)

            new_lane_dict = {'road_name':road_name, 'lane_id':lane_id,
                       'left_border':left_border, 'right_border':right_border,
                       'lane_width':lane_width, 'is_inverted':left_lane_info['is_inverted'],
                        'lane_drivable':lane_drivable,
                        'lane_center':lane_center, 'road_center':centerline_xy_cumulated,
                        'lane_section_s':lane_section_s, 'lane_center_s':lane_width[:,0],
                        'lane_roadmark':lane_roadmark, 'center_roadmark':centerline_roadmark_cumulated}

            lane_center_dict_list.append(new_lane_dict)

    for n, lane_id in enumerate(right_lane_ids):
        right_lane_info = right_lane_info_dict[lane_id]
        is_drivable = any([lane_type == 'driving' for lane_type in right_lane_info['section_lane_type'].values()])
        if is_drivable:
            left_border = right_lane_xy_dict[right_lane_ids[n-1]] if n > 0 else centerline_xy_cumulated
            right_border = right_lane_xy_dict[right_lane_ids[n]]

            lane_center = 0.5 * (left_border + right_border)
            lane_width = right_lane_width_dict[lane_id]
            lane_roadmark = right_roadmark_dict[lane_id]
            lane_drivable = right_lane_is_drivable_dict[lane_id]
            if right_lane_info['is_inverted']:
                lane_center = np.flip(lane_center, axis=0)
                lane_width = np.flip(lane_width, axis=0)
                lane_roadmark = np.flip(lane_roadmark, axis=0)
                lane_drivable = np.flip(lane_drivable, axis=0)

            new_lane_dict = {'road_name':road_name, 'lane_id':lane_id,
                       'left_border':left_border, 'right_border':right_border,
                       'lane_width':lane_width, 'is_inverted':right_lane_info['is_inverted'],
                        'lane_drivable':lane_drivable,
                        'lane_center':lane_center, 'road_center':centerline_xy_cumulated,
                        'lane_section_s':lane_section_s, 'lane_center_s':lane_width[:,0],
                        'lane_roadmark':lane_roadmark, 'center_roadmark':centerline_roadmark_cumulated}

            lane_center_dict_list.append(new_lane_dict)
            
    return lane_center_dict_list, (centerline_xy_cumulated, left_lane_xy_dict, right_lane_xy_dict)

def extract_graph_info_partial(xml_root, node_dist=3.0, min_width=2.0,
                       lane_change_progress_min_dist=0.0, lane_change_progress_max_dist=6.0,
                       successor_min_dist=1.0, adjacency_min_dist=1.0, dist_knn=5,
                       prev_seg_G=None, prev_G=None, processed_roads=None,
                       signal_object_dict=None, signal_reference_dict=None, cutoff_dist=None, vehicle_pos=None):
    seg_G = nx.DiGraph() if prev_seg_G is None else prev_seg_G.copy()
    G = nx.DiGraph() if prev_G is None else prev_G.copy()
    processed_roads = [] if processed_roads is None else processed_roads
    new_processed_roads, not_processed_roads = [], []
    signal_object_dict = {} if signal_object_dict is None else signal_object_dict
    signal_reference_dict = {} if signal_reference_dict is None else signal_reference_dict

    for road in xml_root.findall('road'):
        road_name = "Road " + road.get('id')
        if road_name in processed_roads:
            continue
        elif cutoff_dist is not None:
            xy_list = [(float(g.get('x')), float(g.get('y'))) for g in road.findall('planView/geometry')]
            dist = [(vehicle_pos[0] - x) ** 2 + (vehicle_pos[1] - y) ** 2 > cutoff_dist ** 2 for x, y in xy_list]
            if all(dist):
                not_processed_roads.append(road_name)
                continue

        new_processed_roads.append(road_name)

        lane_center_dict_list, (ref_centerline, _, _) = extract_lane_center(road)

        # get traffic signal info
        signals = road.find('signals')
        signal_objects = signals.findall('signal') if signals is not None else []
        signal_references = signals.findall('signalReference') if signals is not None else []

        for signal_object in signal_objects:
            signal_id = int(signal_object.get('id'))
            signal_st = np.asfarray([[signal_object.get('s'), signal_object.get('t')]])
            signal_xy = frenet_to_cartesian_approx(ref_centerline, signal_st).squeeze()
            hOffset = float(signal_object.get('hOffset'))
            height = float(signal_object.get('height'))
            signal_object_dict[signal_id] = {'road_name':road_name, 'signal_xy': signal_xy, 'hOffset': hOffset, 'height': height}

        for signal_reference in signal_references:
            validity = signal_reference.find('validity')
            fromLane_validity = int(validity.get('fromLane'))
            toLane_validity = int(validity.get('toLane'))
            signal_id = int(signal_reference.get('id'))
            for i in range(min(fromLane_validity, toLane_validity), max(fromLane_validity, toLane_validity) + 1):
                valid_lane_id = road_name + "_" + str(i)  # Road N_m
                signal_reference_dict[valid_lane_id] = signal_id

        # get lane seg info
        for lane_seg_info in lane_center_dict_list:
            # register seg
            road_name = lane_seg_info['road_name']  # Road Nb
            lane_id = lane_seg_info['lane_id']
            seg_name = road_name + "_" + lane_id  # Road N_m
            lane_width = lane_seg_info['lane_width']
            lane_center = lane_seg_info['lane_center']
            left_border = lane_seg_info['left_border']
            right_border = lane_seg_info['right_border']
            road_center = lane_seg_info['road_center']
            is_inverted = lane_seg_info['is_inverted']
            lane_drivable = lane_seg_info['lane_drivable']
            lane_section_s = lane_seg_info['lane_section_s']
            lane_center_s = lane_seg_info['lane_center_s']

            lane_roadmark = lane_seg_info['lane_roadmark']
            center_roadmark = lane_seg_info['center_roadmark']

            lane_center_as_ref_line_frenet = cartesian_to_frenet_approx(ref_centerline, lane_center)

            lane_center_s_in_order = np.flip(lane_center_s, axis=0) if is_inverted else lane_center_s
            section_starting_points = [np.clip(bisect_left(lane_center_s_in_order, s), 0, len(lane_center_s_in_order)-1) for s in lane_section_s]
            lane_center_as_ref_line_frenet_in_order = np.flip(lane_center_as_ref_line_frenet, axis=0) if is_inverted else lane_center_as_ref_line_frenet
            section_starting_points_as_frenet = lane_center_as_ref_line_frenet_in_order[section_starting_points]

            # register node
            direc_vec = np.diff(lane_center, axis=0)
            line_length = np.sum(np.hypot(direc_vec[:, 0], direc_vec[:, 1]))

            node_line_s = np.concatenate([np.arange(0, line_length, node_dist), np.array([line_length])])
            ii = np.searchsorted(node_line_s, section_starting_points_as_frenet[:,0])
            node_line_s = np.unique(np.insert(node_line_s, ii, section_starting_points_as_frenet[:,0]))
            node_line_t = np.zeros_like(node_line_s)
            node_line_st = np.stack([node_line_s, node_line_t], axis=1)
            node_line_xy = frenet_to_cartesian_approx(lane_center, node_line_st)

            # filter nodes based on width
            has_not_enough_width = any(lane_width[:, 1] < min_width)
            is_not_drivable = any(lane_drivable < 0.5)
            if has_not_enough_width and is_not_drivable:
                node_line_as_refer_line_frenet = cartesian_to_frenet_approx(ref_centerline, node_line_xy)
                lane_width_append = lane_width.copy()
                lane_width_append[:, 0] = lane_center_as_ref_line_frenet[:,0]
                lane_width_append = np.concatenate([lane_width_append[:1], lane_width_append, lane_width_append[-1:]])
                lane_width_append[0, 0] += 20 if lane_width_append[0, 0] > lane_width_append[-1, 0] else -20
                lane_width_append[-1, 0] += 20 if lane_width_append[0, 0] < lane_width_append[-1, 0] else -20

                width_intp_f = interp1d(lane_width_append[:, 0], lane_width_append[:, 1], kind='nearest')
                width_intp = width_intp_f(node_line_as_refer_line_frenet[:, 0])
                width_filter = width_intp >= min_width

                lane_drivable_append = lane_width_append.copy()
                lane_drivable_append[1:-1,1] = lane_drivable
                lane_drivable_append[0,1] = lane_drivable[0]
                lane_drivable_append[-1,1] = lane_drivable[-1]

                drivable_intp_f = interp1d(lane_drivable_append[:, 0], lane_drivable_append[:, 1], kind='nearest')
                drivable_intp = drivable_intp_f(node_line_as_refer_line_frenet[:, 0])
                drivable_filter = drivable_intp > 0.5

                final_node_filter = np.logical_and(width_filter, drivable_filter)
                node_line_xy = node_line_xy[final_node_filter]
            else:
                final_node_filter = None

            if len(node_line_xy) == 0:
                continue

            seg_G.add_node(seg_name, road=road_name,
                           pos=lane_center[lane_center.shape[0] // 2],
                           lane_center=lane_center, left_border=left_border, right_border=right_border,
                           road_center=road_center, lane_width=lane_width,
                           is_inverted=is_inverted, valid_node_filter=final_node_filter,
                           lane_drivable=lane_drivable, lane_section_s=lane_section_s,
                           lane_roadmark=lane_roadmark, center_roadmark=center_roadmark)

            node_line_road_frenet = cartesian_to_frenet_approx(road_center, node_line_xy)

            for n, node_xy in enumerate(node_line_xy):
                node_name = seg_name + "_" + str(n)  # Road N_m_l
                node_frenet = node_line_road_frenet[n]
                node_section_num = np.clip(np.sum(node_frenet[0] >= lane_section_s) - 1, 0, None)
                G.add_node(node_name, road=road_name, seg=seg_name, pos=node_xy, order=n,
                           frenet=node_line_road_frenet[n], lane_section=node_section_num)

            # additional node info
            seg_G.nodes[seg_name]['frenet'] = node_line_road_frenet
            seg_G.nodes[seg_name]['point_line'] = node_line_xy
            seg_G.nodes[seg_name]['point_names'] = [seg_name + "_" + str(n) for n in range(len(node_line_xy))]
            seg_G.nodes[seg_name]['in_junction'] = True  # default value, it will be changed in next code
            for node_name in seg_G.nodes[seg_name]['point_names']:
                G.nodes[node_name]['in_junction'] = True  # default value, it will be changed in next code

    # link edges
    for road in xml_root.findall('road'):
        road_name = "Road " + road.get('id')
        if road_name in not_processed_roads:
            continue

        # find predecessor and successor
        link_predecessor = road.find('link/predecessor')
        if link_predecessor is not None:
            link_predecessor_type = link_predecessor.get('elementType')
            link_predecessor_name = "Road " + link_predecessor.get('elementId')
            link_predecessor_contact = link_predecessor.get('contactPoint')

        link_successor = road.find('link/successor')
        if link_successor is not None:
            link_successor_type = link_successor.get('elementType')
            link_successor_name = "Road " + link_successor.get('elementId')
            link_successor_contact = link_successor.get('contactPoint')

        # check each lane link
        lane_section = road.findall('lanes/laneSection')
        lane_section.sort(key=lambda ls: float(ls.get('s')))

        left_lane_ids, right_lane_ids = [], []
        left_lane_info_dict, right_lane_info_dict = {}, {}
        for section in lane_section:
            left_lanes = list(section.find('left')) if section.find('left') is not None else []
            right_lanes = list(section.find('right')) if section.find('right') is not None else []

            for lane in left_lanes:
                if lane.get('id') not in left_lane_ids:
                    if lane.find('userData/vectorLane') is not None and \
                            lane.find('userData/vectorLane').get('travelDir') == 'forward':
                        is_inverted = False
                    else:
                        is_inverted = True  # default
                    left_lane_info_dict[lane.get('id')] = {'type': lane.get('type'), 'is_inverted': is_inverted,
                                                           'predecessor': lane.find('link/predecessor'),
                                                           'successor': lane.find('link/successor')}
                    left_lane_ids.append(lane.get('id'))
                else:
                    if left_lane_info_dict[lane.get('id')]['predecessor'] is None:
                        left_lane_info_dict[lane.get('id')]['predecessor'] = lane.find('link/predecessor')
                    if left_lane_info_dict[lane.get('id')]['successor'] is None:
                        left_lane_info_dict[lane.get('id')]['successor'] = lane.find('link/successor')

            for lane in right_lanes:
                if lane.get('id') not in right_lane_ids:
                    if lane.find('userData/vectorLane') is not None and \
                            lane.find('userData/vectorLane').get('travelDir') == 'backward':
                        is_inverted = True
                    else:
                        is_inverted = False  # default
                    right_lane_info_dict[lane.get('id')] = {'type': lane.get('type'), 'is_inverted': is_inverted,
                                                            'predecessor': lane.find('link/predecessor'),
                                                            'successor': lane.find('link/successor')}
                    right_lane_ids.append(lane.get('id'))
                else:
                    if right_lane_info_dict[lane.get('id')]['predecessor'] is None:
                        right_lane_info_dict[lane.get('id')]['predecessor'] = lane.find('link/predecessor')
                    if right_lane_info_dict[lane.get('id')]['successor'] is None:
                        right_lane_info_dict[lane.get('id')]['successor'] = lane.find('link/successor')

        left_lane_ids.sort(key=lambda ls: abs(int(ls)))
        right_lane_ids.sort(key=lambda ls: abs(int(ls)))

        # seg edge between inside road
        for n, lane_id in enumerate(left_lane_ids):
            if n != 0 and seg_G.has_node(road_name + "_" + left_lane_ids[n]):
                if seg_G.has_node(road_name + "_" + left_lane_ids[n - 1]):
                    left_seg_name = road_name + "_" + left_lane_ids[n - 1]  # Road N_m
                    right_seg_name = road_name + "_" + left_lane_ids[n]  # Road N_m
                    if not seg_G.has_edge(left_seg_name, right_seg_name):
                        seg_G.add_edge(left_seg_name, right_seg_name, edge_type='right')
                    if not seg_G.has_edge(right_seg_name, left_seg_name):
                        seg_G.add_edge(right_seg_name, left_seg_name, edge_type='left')

        for n, lane_id in enumerate(right_lane_ids):
            if n != 0 and seg_G.has_node(road_name + "_" + right_lane_ids[n]):
                if seg_G.has_node(road_name + "_" + right_lane_ids[n - 1]):
                    left_seg_name = road_name + "_" + right_lane_ids[n - 1]  # Road N_m
                    right_seg_name = road_name + "_" + right_lane_ids[n]  # Road N_m
                    if not seg_G.has_edge(left_seg_name, right_seg_name):
                        seg_G.add_edge(left_seg_name, right_seg_name, edge_type='right')
                    if not seg_G.has_edge(right_seg_name, left_seg_name):
                        seg_G.add_edge(right_seg_name, left_seg_name, edge_type='left')

        all_lane_section = left_lane_ids + right_lane_ids
        all_lane_info_dict = {**left_lane_info_dict, **right_lane_info_dict}

        # edge between inside road
        for lane_id in all_lane_section:
            seg_name = road_name + "_" + lane_id  # Road N_m
            if seg_G.has_node(seg_name):
                # point edge inside segment
                point_names = seg_G.nodes[seg_name]['point_names']
                valid_node_filter = seg_G.nodes[seg_name]['valid_node_filter']
                if valid_node_filter is not None:
                    is_connected_with_prev_node = (valid_node_filter[1:] == valid_nodea_filter[:-1])
                    is_connected_with_prev_node = np.concatenate([np.array([True]), is_connected_with_prev_node])
                    is_connected_with_prev_node = is_connected_with_prev_node[valid_node_filter]
                for n in range(len(point_names)):
                    if n != 0:
                        prev_node_name = point_names[n - 1]
                        next_node_name = point_names[n]
                        if valid_node_filter is None or is_connected_with_prev_node[n]:
                            G.add_edge(prev_node_name, next_node_name, edge_type='sequential')

                # point and seg edge between linked roads
                predecessor = all_lane_info_dict[lane_id]['predecessor']
                successor = all_lane_info_dict[lane_id]['successor']

                if predecessor is not None and link_predecessor_type == 'road':
                    predecessor_id = predecessor.get('id')
                    predecessor_name = link_predecessor_name + "_" + predecessor_id
                    if predecessor_name in seg_G.nodes:
                        seg_G.nodes[predecessor_name]['in_junction'] = False  # additional node info
                        for node_name in seg_G.nodes[predecessor_name]['point_names']:
                            G.nodes[node_name]['in_junction'] = False

                        if seg_G.has_node(predecessor_name):
                            if all_lane_info_dict[lane_id]['is_inverted']:
                                if not seg_G.has_edge(seg_name, predecessor_name):
                                    seg_G.add_edge(seg_name, predecessor_name, edge_type='successor')
                                    prev_point_name = seg_G.nodes[seg_name]['point_names'][-1]
                                    next_point_name = seg_G.nodes[predecessor_name]['point_names'][0]
                                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos'])**2)
                                    if dist < successor_min_dist ** 2:
                                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')
                            else:
                                if not seg_G.has_edge(predecessor_name, seg_name):
                                    seg_G.add_edge(predecessor_name, seg_name, edge_type='successor')
                                    prev_point_name = seg_G.nodes[predecessor_name]['point_names'][-1]
                                    next_point_name = seg_G.nodes[seg_name]['point_names'][0]
                                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos'])**2)
                                    if dist < successor_min_dist ** 2:
                                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')

                if successor is not None and link_successor_type == 'road':
                    successor_id = successor.get('id')
                    successor_name = link_successor_name + "_" + successor_id
                    if successor_name in seg_G.nodes:
                        seg_G.nodes[successor_name]['in_junction'] = False  # additional node info
                        for node_name in seg_G.nodes[successor_name]['point_names']:
                            G.nodes[node_name]['in_junction'] = False

                        if seg_G.has_node(successor_name):
                            if all_lane_info_dict[lane_id]['is_inverted']:
                                if not seg_G.has_edge(successor_name, seg_name):
                                    seg_G.add_edge(successor_name, seg_name, edge_type='successor')
                                    prev_point_name = seg_G.nodes[successor_name]['point_names'][-1]
                                    next_point_name = seg_G.nodes[seg_name]['point_names'][0]
                                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos'])**2)
                                    if dist < successor_min_dist ** 2:
                                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')
                            else:
                                if not seg_G.has_edge(seg_name, successor_name):
                                    seg_G.add_edge(seg_name, successor_name, edge_type='successor')
                                    prev_point_name = seg_G.nodes[seg_name]['point_names'][-1]
                                    next_point_name = seg_G.nodes[successor_name]['point_names'][0]
                                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos'])**2)
                                    if dist < successor_min_dist ** 2:
                                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')

    # check each junction link
    for junction in xml_root.findall('junction'):
        for connection in junction.findall('connection'):
            prev_road_name = "Road " + connection.get('incomingRoad')
            next_road_name = "Road " + connection.get('connectingRoad')

            if prev_road_name in not_processed_roads or next_road_name in not_processed_roads:
                continue

            for lanelink in connection.findall("laneLink"):
                prev_lane_id = lanelink.get('from')
                next_lane_id = lanelink.get('to')
                prev_seg_name = prev_road_name + "_" + prev_lane_id
                next_seg_name = next_road_name + "_" + next_lane_id

                if not seg_G.has_edge(prev_seg_name, next_seg_name) \
                        and seg_G.has_node(prev_seg_name) \
                        and seg_G.has_node(next_seg_name):
                    seg_G.add_edge(prev_seg_name, next_seg_name, edge_type='successor')
                    prev_point_name = seg_G.nodes[prev_seg_name]['point_names'][-1]
                    next_point_name = seg_G.nodes[next_seg_name]['point_names'][0]
                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos']) ** 2)
                    if dist < successor_min_dist ** 2:
                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')

    # assign signal info
    for node in seg_G.nodes():
        seg_G.nodes[node]['on_traffic_signal'] = None
    for node in G.nodes():
        G.nodes[node]['on_traffic_signal'] = None
    for signal_ref_lane in signal_reference_dict.keys():
        if seg_G.has_node(signal_ref_lane):
            seg_G.nodes[signal_ref_lane]['on_traffic_signal'] = signal_reference_dict[signal_ref_lane]
            for point_name in seg_G.nodes[signal_ref_lane]['point_names']:
                G.nodes[point_name]['on_traffic_signal'] = signal_reference_dict[signal_ref_lane]

    pos_kdtree_keys = np.stack(list(dict(G.nodes('pos')).keys()))
    pos_kdtree = KDTree(np.stack(list(dict(G.nodes('pos')).values())))

    # add distance based edge
    graph_node_index = list(dict(G.nodes()).keys())
    for idx1 in graph_node_index:
        xy = G.nodes[idx1]['pos'].reshape([1, 2])
        dist, idxs = pos_kdtree.query(xy, k=dist_knn)
        for d, i in zip(dist[0], idxs[0]):
            idx2 = pos_kdtree_keys[i]
            if idx1 == idx2:
                pass
            elif d < adjacency_min_dist and not seg_G.has_edge(G.nodes[idx1]['seg'], G.nodes[idx2]['seg']) \
                    and seg_G.nodes[G.nodes[idx1]['seg']]['road'] != seg_G.nodes[G.nodes[idx2]['seg']]['road']:
                G.add_edge(idx1, idx2, edge_type='adjacency')
                seg_G.add_edge(G.nodes[idx1]['seg'], G.nodes[idx2]['seg'], edge_type='adjacency')

    # add lane_change based edge
    for seg_edge in seg_G.edges():
        if seg_G.nodes[seg_edge[0]]['road'] not in new_processed_roads and seg_G.nodes[seg_edge[1]]['road'] not in new_processed_roads:
            continue
        if seg_G.edges[seg_edge]['edge_type'] in ['left', 'right']:
            lane_id = seg_edge[0]
            changed_lane_id = seg_edge[1]

            lane_points = seg_G.nodes[lane_id]['point_names']
            changed_lane_points = seg_G.nodes[changed_lane_id]['point_names']

            is_inverted = seg_G.nodes[lane_id]['is_inverted']

            changed_lane_idx = 0
            for p_id in lane_points:
                while (changed_lane_idx < len(changed_lane_points)):
                    changed_p_id = changed_lane_points[changed_lane_idx]

                    if is_inverted:
                        lane_s = -G.nodes[p_id]['frenet'][0]
                        changed_lane_s = -G.nodes[changed_p_id]['frenet'][0]
                    else:
                        lane_s = G.nodes[p_id]['frenet'][0]
                        changed_lane_s = G.nodes[changed_p_id]['frenet'][0]

                    if changed_lane_s < lane_s + lane_change_progress_min_dist:
                        changed_lane_idx += 1
                    else:
                        if changed_lane_s < lane_s + lane_change_progress_max_dist:
                            G.add_edge(p_id, changed_p_id, edge_type=seg_G.edges[seg_edge]['edge_type'])
                        break

    for edge in G.edges():
        if 'direc' in G.edges[edge].keys():
            continue
        edge_direc = G.nodes[edge[1]]['pos'] - G.nodes[edge[0]]['pos']
        G.edges[edge]['direc'] = edge_direc
        G.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))

    processed_roads += new_processed_roads
    return G, seg_G, pos_kdtree_keys, pos_kdtree, processed_roads, signal_object_dict, signal_reference_dict

def extract_graph_info(xml_root, node_dist=3.0, min_width=2.0,
                       lane_change_progress_min_dist=0.0, lane_change_progress_max_dist=6.0,
                       successor_min_dist=1.0, adjacency_min_dist=1.0, dist_knn=5, cutoff_dist=None, vehicle_pos=None):
    seg_G = nx.DiGraph()
    G = nx.DiGraph()

    for road in xml_root.findall('road'):
        if cutoff_dist is not None:
            xy_list = [(float(g.get('x')), float(g.get('y'))) for g in road.findall('planView/geometry')]
            dist = [(vehicle_pos[0] - x) ** 2 + (vehicle_pos[1] - y) ** 2 > cutoff_dist ** 2 for x, y in xy_list]
            if all(dist):
                continue

        lane_center_dict_list, (ref_centerline, _, _) = extract_lane_center(road)
        road_length = float(road.get('length'))

        for lane_seg_info in lane_center_dict_list:
            # register seg
            road_name = lane_seg_info['road_name']  # Road Nb
            lane_id = lane_seg_info['lane_id']
            seg_name = road_name + "_" + lane_id  # Road N_m
            lane_width = lane_seg_info['lane_width']
            lane_center = lane_seg_info['lane_center']
            left_border = lane_seg_info['left_border']
            right_border = lane_seg_info['right_border']
            road_center = lane_seg_info['road_center']
            is_inverted = lane_seg_info['is_inverted']
            lane_drivable = lane_seg_info['lane_drivable']
            lane_section_s = lane_seg_info['lane_section_s']
            lane_center_s = lane_seg_info['lane_center_s']

            lane_roadmark = lane_seg_info['lane_roadmark']
            center_roadmark = lane_seg_info['center_roadmark']

            lane_center_as_ref_line_frenet = cartesian_to_frenet_approx(ref_centerline, lane_center)

            lane_center_s_in_order = np.flip(lane_center_s, axis=0) if is_inverted else lane_center_s
            section_starting_points = [np.clip(bisect_left(lane_center_s_in_order, s), 0, len(lane_center_s_in_order)-1) for s in lane_section_s]
            lane_center_as_ref_line_frenet_in_order = np.flip(lane_center_as_ref_line_frenet, axis=0) if is_inverted else lane_center_as_ref_line_frenet
            section_starting_points_as_frenet = lane_center_as_ref_line_frenet_in_order[section_starting_points]

            # register node
            direc_vec = np.diff(lane_center, axis=0)
            line_length = np.sum(np.hypot(direc_vec[:, 0], direc_vec[:, 1]))

            node_line_s = np.concatenate([np.arange(0, line_length, node_dist), np.array([line_length])])
            ii = np.searchsorted(node_line_s, section_starting_points_as_frenet[:,0])
            node_line_s = np.unique(np.insert(node_line_s, ii, section_starting_points_as_frenet[:,0]))
            node_line_t = np.zeros_like(node_line_s)
            node_line_st = np.stack([node_line_s, node_line_t], axis=1)
            node_line_xy = frenet_to_cartesian_approx(lane_center, node_line_st)

            # filter nodes based on width
            has_not_enough_width = any(lane_width[:, 1] < min_width)
            is_not_drivable = any(lane_drivable < 0.5)
            if has_not_enough_width and is_not_drivable:
                node_line_as_refer_line_frenet = cartesian_to_frenet_approx(ref_centerline, node_line_xy)
                lane_width_append = lane_width.copy()
                lane_width_append[:, 0] = lane_center_as_ref_line_frenet[:,0]
                lane_width_append = np.concatenate([lane_width_append[:1], lane_width_append, lane_width_append[-1:]])
                lane_width_append[0, 0] += 20 if lane_width_append[0, 0] > lane_width_append[-1, 0] else -20
                lane_width_append[-1, 0] += 20 if lane_width_append[0, 0] < lane_width_append[-1, 0] else -20

                width_intp_f = interp1d(lane_width_append[:, 0], lane_width_append[:, 1], kind='nearest')
                width_intp = width_intp_f(node_line_as_refer_line_frenet[:, 0])
                width_filter = width_intp >= min_width

                lane_drivable_append = lane_width_append.copy()
                lane_drivable_append[1:-1,1] = lane_drivable
                lane_drivable_append[0,1] = lane_drivable[0]
                lane_drivable_append[-1,1] = lane_drivable[-1]

                drivable_intp_f = interp1d(lane_drivable_append[:, 0], lane_drivable_append[:, 1], kind='nearest')
                drivable_intp = drivable_intp_f(node_line_as_refer_line_frenet[:, 0])
                drivable_filter = drivable_intp > 0.5

                final_node_filter = np.logical_and(width_filter, drivable_filter)
                node_line_xy = node_line_xy[final_node_filter]
            else:
                final_node_filter = None

            if len(node_line_xy) == 0:
                continue

            seg_G.add_node(seg_name, road=road_name,
                           pos=lane_center[lane_center.shape[0] // 2],
                           lane_center=lane_center, left_border=left_border, right_border=right_border,
                           road_center=road_center, lane_width=lane_width,
                           is_inverted=is_inverted, valid_node_filter=final_node_filter,
                           lane_drivable=lane_drivable, lane_section_s=lane_section_s,
                           lane_roadmark=lane_roadmark, center_roadmark=center_roadmark)

            node_line_road_frenet = cartesian_to_frenet_approx(road_center, node_line_xy)

            for n, node_xy in enumerate(node_line_xy):
                node_name = seg_name + "_" + str(n)  # Road N_m_l
                node_frenet = node_line_road_frenet[n]
                node_section_num = np.clip(np.sum(node_frenet[0] >= lane_section_s) - 1, 0, None)
                G.add_node(node_name, road=road_name, seg=seg_name, pos=node_xy, order=n,
                           frenet=node_line_road_frenet[n], lane_section=node_section_num)

            # additional node info
            seg_G.nodes[seg_name]['frenet'] = node_line_road_frenet
            seg_G.nodes[seg_name]['point_line'] = node_line_xy
            seg_G.nodes[seg_name]['point_names'] = [seg_name + "_" + str(n) for n in range(len(node_line_xy))]
            seg_G.nodes[seg_name]['in_junction'] = True  # default value, it will be changed in next code
            for node_name in seg_G.nodes[seg_name]['point_names']:
                G.nodes[node_name]['in_junction'] = True  # default value, it will be changed in next code

    # link edges
    for road in xml_root.findall('road'):
        if cutoff_dist is not None:
            xy_list = [(float(g.get('x')), float(g.get('y'))) for g in road.findall('planView/geometry')]
            dist = [(vehicle_pos[0] - x) ** 2 + (vehicle_pos[1] - y) ** 2 > cutoff_dist ** 2 for x, y in xy_list]
            if all(dist):
                continue

        road_name = "Road " + road.get('id')

        # find predecessor and successor
        link_predecessor = road.find('link/predecessor')
        if link_predecessor is not None:
            link_predecessor_type = link_predecessor.get('elementType')
            link_predecessor_name = "Road " + link_predecessor.get('elementId')
            link_predecessor_contact = link_predecessor.get('contactPoint')

        link_successor = road.find('link/successor')
        if link_successor is not None:
            link_successor_type = link_successor.get('elementType')
            link_successor_name = "Road " + link_successor.get('elementId')
            link_successor_contact = link_successor.get('contactPoint')

        # check each lane link
        lane_section = road.findall('lanes/laneSection')
        lane_section.sort(key=lambda ls: float(ls.get('s')))

        left_lane_ids, right_lane_ids = [], []
        left_lane_info_dict, right_lane_info_dict = {}, {}
        for section in lane_section:
            left_lanes = list(section.find('left')) if section.find('left') is not None else []
            right_lanes = list(section.find('right')) if section.find('right') is not None else []

            for lane in left_lanes:
                if lane.get('id') not in left_lane_ids:
                    if lane.find('userData/vectorLane') is not None and \
                            lane.find('userData/vectorLane').get('travelDir') == 'forward':
                        is_inverted = False
                    else:
                        is_inverted = True  # default
                    left_lane_info_dict[lane.get('id')] = {'type': lane.get('type'), 'is_inverted': is_inverted,
                                                           'predecessor': lane.find('link/predecessor'),
                                                           'successor': lane.find('link/successor')}
                    left_lane_ids.append(lane.get('id'))
                else:
                    if left_lane_info_dict[lane.get('id')]['predecessor'] is None:
                        left_lane_info_dict[lane.get('id')]['predecessor'] = lane.find('link/predecessor')
                    if left_lane_info_dict[lane.get('id')]['successor'] is None:
                        left_lane_info_dict[lane.get('id')]['successor'] = lane.find('link/successor')

            for lane in right_lanes:
                if lane.get('id') not in right_lane_ids:
                    if lane.find('userData/vectorLane') is not None and \
                            lane.find('userData/vectorLane').get('travelDir') == 'backward':
                        is_inverted = True
                    else:
                        is_inverted = False  # default
                    right_lane_info_dict[lane.get('id')] = {'type': lane.get('type'), 'is_inverted': is_inverted,
                                                            'predecessor': lane.find('link/predecessor'),
                                                            'successor': lane.find('link/successor')}
                    right_lane_ids.append(lane.get('id'))
                else:
                    if right_lane_info_dict[lane.get('id')]['predecessor'] is None:
                        right_lane_info_dict[lane.get('id')]['predecessor'] = lane.find('link/predecessor')
                    if right_lane_info_dict[lane.get('id')]['successor'] is None:
                        right_lane_info_dict[lane.get('id')]['successor'] = lane.find('link/successor')

        left_lane_ids.sort(key=lambda ls: abs(int(ls)))
        right_lane_ids.sort(key=lambda ls: abs(int(ls)))

        # seg edge between inside road
        for n, lane_id in enumerate(left_lane_ids):
            if n != 0 and seg_G.has_node(road_name + "_" + left_lane_ids[n]):
                if seg_G.has_node(road_name + "_" + left_lane_ids[n - 1]):
                    left_seg_name = road_name + "_" + left_lane_ids[n - 1]  # Road N_m
                    right_seg_name = road_name + "_" + left_lane_ids[n]  # Road N_m
                    if not seg_G.has_edge(left_seg_name, right_seg_name):
                        seg_G.add_edge(left_seg_name, right_seg_name, edge_type='right')
                    if not seg_G.has_edge(right_seg_name, left_seg_name):
                        seg_G.add_edge(right_seg_name, left_seg_name, edge_type='left')

        for n, lane_id in enumerate(right_lane_ids):
            if n != 0 and seg_G.has_node(road_name + "_" + right_lane_ids[n]):
                if seg_G.has_node(road_name + "_" + right_lane_ids[n - 1]):
                    left_seg_name = road_name + "_" + right_lane_ids[n - 1]  # Road N_m
                    right_seg_name = road_name + "_" + right_lane_ids[n]  # Road N_m
                    if not seg_G.has_edge(left_seg_name, right_seg_name):
                        seg_G.add_edge(left_seg_name, right_seg_name, edge_type='right')
                    if not seg_G.has_edge(right_seg_name, left_seg_name):
                        seg_G.add_edge(right_seg_name, left_seg_name, edge_type='left')

        all_lane_section = left_lane_ids + right_lane_ids
        all_lane_info_dict = {**left_lane_info_dict, **right_lane_info_dict}

        # edge between inside road
        for lane_id in all_lane_section:
            seg_name = road_name + "_" + lane_id  # Road N_m
            if seg_G.has_node(seg_name):
                # point edge inside segment
                point_names = seg_G.nodes[seg_name]['point_names']
                valid_node_filter = seg_G.nodes[seg_name]['valid_node_filter']
                if valid_node_filter is not None:
                    is_connected_with_prev_node = (valid_node_filter[1:] == valid_node_filter[:-1])
                    is_connected_with_prev_node = np.concatenate([np.array([True]), is_connected_with_prev_node])
                    is_connected_with_prev_node = is_connected_with_prev_node[valid_node_filter]
                for n in range(len(point_names)):
                    if n != 0:
                        prev_node_name = point_names[n - 1]
                        next_node_name = point_names[n]
                        if valid_node_filter is None or is_connected_with_prev_node[n]:
                            G.add_edge(prev_node_name, next_node_name, edge_type='sequential')

                # point and seg edge between linked roads
                predecessor = all_lane_info_dict[lane_id]['predecessor']
                successor = all_lane_info_dict[lane_id]['successor']

                if predecessor is not None and link_predecessor_type == 'road':
                    predecessor_id = predecessor.get('id')
                    predecessor_name = link_predecessor_name + "_" + predecessor_id
                    if predecessor_name in seg_G.nodes:
                        seg_G.nodes[predecessor_name]['in_junction'] = False  # additional node info
                        for node_name in seg_G.nodes[predecessor_name]['point_names']:
                            G.nodes[node_name]['in_junction'] = False

                        if seg_G.has_node(predecessor_name):
                            if all_lane_info_dict[lane_id]['is_inverted']:
                                if not seg_G.has_edge(seg_name, predecessor_name):
                                    seg_G.add_edge(seg_name, predecessor_name, edge_type='successor')
                                    prev_point_name = seg_G.nodes[seg_name]['point_names'][-1]
                                    next_point_name = seg_G.nodes[predecessor_name]['point_names'][0]
                                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos'])**2)
                                    if dist < successor_min_dist ** 2:
                                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')
                            else:
                                if not seg_G.has_edge(predecessor_name, seg_name):
                                    seg_G.add_edge(predecessor_name, seg_name, edge_type='successor')
                                    prev_point_name = seg_G.nodes[predecessor_name]['point_names'][-1]
                                    next_point_name = seg_G.nodes[seg_name]['point_names'][0]
                                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos'])**2)
                                    if dist < successor_min_dist ** 2:
                                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')

                if successor is not None and link_successor_type == 'road':
                    successor_id = successor.get('id')
                    successor_name = link_successor_name + "_" + successor_id
                    if successor_name in seg_G.nodes:
                        seg_G.nodes[successor_name]['in_junction'] = False  # additional node info
                        for node_name in seg_G.nodes[successor_name]['point_names']:
                            G.nodes[node_name]['in_junction'] = False

                        if seg_G.has_node(successor_name):
                            if all_lane_info_dict[lane_id]['is_inverted']:
                                if not seg_G.has_edge(successor_name, seg_name):
                                    seg_G.add_edge(successor_name, seg_name, edge_type='successor')
                                    prev_point_name = seg_G.nodes[successor_name]['point_names'][-1]
                                    next_point_name = seg_G.nodes[seg_name]['point_names'][0]
                                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos'])**2)
                                    if dist < successor_min_dist ** 2:
                                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')
                            else:
                                if not seg_G.has_edge(seg_name, successor_name):
                                    seg_G.add_edge(seg_name, successor_name, edge_type='successor')
                                    prev_point_name = seg_G.nodes[seg_name]['point_names'][-1]
                                    next_point_name = seg_G.nodes[successor_name]['point_names'][0]
                                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos'])**2)
                                    if dist < successor_min_dist ** 2:
                                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')

    # check each junction link
    for junction in xml_root.findall('junction'):
        for connection in junction.findall('connection'):
            prev_road_name = "Road " + connection.get('incomingRoad')
            next_road_name = "Road " + connection.get('connectingRoad')

            for lanelink in connection.findall("laneLink"):
                prev_lane_id = lanelink.get('from')
                next_lane_id = lanelink.get('to')
                prev_seg_name = prev_road_name + "_" + prev_lane_id
                next_seg_name = next_road_name + "_" + next_lane_id

                if not seg_G.has_edge(prev_seg_name, next_seg_name) \
                        and seg_G.has_node(prev_seg_name) \
                        and seg_G.has_node(next_seg_name):
                    seg_G.add_edge(prev_seg_name, next_seg_name, edge_type='successor')
                    prev_point_name = seg_G.nodes[prev_seg_name]['point_names'][-1]
                    next_point_name = seg_G.nodes[next_seg_name]['point_names'][0]
                    dist = np.sum((G.nodes[prev_point_name]['pos'] - G.nodes[next_point_name]['pos']) ** 2)
                    if dist < successor_min_dist ** 2:
                        G.add_edge(prev_point_name, next_point_name, edge_type='successor')

    pos_kdtree_keys = np.stack(list(dict(G.nodes('pos')).keys()))
    pos_kdtree = KDTree(np.stack(list(dict(G.nodes('pos')).values())))

    # add distance based edge
    graph_node_index = list(dict(G.nodes()).keys())
    for idx1 in graph_node_index:
        xy = G.nodes[idx1]['pos'].reshape([1, 2])
        dist, idxs = pos_kdtree.query(xy, k=dist_knn)
        for d, i in zip(dist[0], idxs[0]):
            idx2 = pos_kdtree_keys[i]
            if idx1 == idx2:
                pass
            elif d < adjacency_min_dist and not seg_G.has_edge(G.nodes[idx1]['seg'], G.nodes[idx2]['seg']) \
                    and seg_G.nodes[G.nodes[idx1]['seg']]['road'] != seg_G.nodes[G.nodes[idx2]['seg']]['road']:
                G.add_edge(idx1, idx2, edge_type='adjacency')
                #seg_G.add_edge(G.nodes[idx1]['seg'], G.nodes[idx2]['seg'], edge_type='adjacency')

    # add lane_change based edge
    for seg_edge in seg_G.edges():
        if seg_G.edges[seg_edge]['edge_type'] in ['left', 'right']:
            lane_id = seg_edge[0]
            changed_lane_id = seg_edge[1]

            lane_points = seg_G.nodes[lane_id]['point_names']
            changed_lane_points = seg_G.nodes[changed_lane_id]['point_names']

            is_inverted = seg_G.nodes[lane_id]['is_inverted']

            changed_lane_idx = 0
            for p_id in lane_points:
                while (changed_lane_idx < len(changed_lane_points)):
                    changed_p_id = changed_lane_points[changed_lane_idx]

                    if is_inverted:
                        lane_s = -G.nodes[p_id]['frenet'][0]
                        changed_lane_s = -G.nodes[changed_p_id]['frenet'][0]
                    else:
                        lane_s = G.nodes[p_id]['frenet'][0]
                        changed_lane_s = G.nodes[changed_p_id]['frenet'][0]

                    if changed_lane_s < lane_s + lane_change_progress_min_dist:
                        changed_lane_idx += 1
                    else:
                        if changed_lane_s < lane_s + lane_change_progress_max_dist:
                            G.add_edge(p_id, changed_p_id, edge_type=seg_G.edges[seg_edge]['edge_type'])
                        break

    # calculate weights
    for edge in G.edges():
        edge_direc = G.nodes[edge[1]]['pos'] - G.nodes[edge[0]]['pos']
        G.edges[edge]['direc'] = edge_direc
        G.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))

    return G, seg_G, pos_kdtree_keys, pos_kdtree

def contract_graph(G, pos_kdtree_keys, pos_kdtree, min_node_dist=0.75):
    G2 = G.copy()

    graph_node_index = list(dict(G2.nodes()).keys())
    for idx in graph_node_index:
        same_pos_list = []
        if idx not in G2.nodes():
            continue
        xy = G2.nodes[idx]['pos'].reshape([1, 2])
        dist, idxs = pos_kdtree.query(xy, k=5)
        for d, i in zip(dist[0], idxs[0]):
            idx2 = pos_kdtree_keys[i]
            if d < min_node_dist and idx2 in G2.nodes() and idx != idx2:
                same_pos_list.append(idx2)  

        for same_node in same_pos_list:
            if 'contraction' in G2.nodes[same_node].keys():
                if 'contraction' in G2.nodes[idx].keys():
                    G2.nodes[idx]['contraction'] = {**G2.nodes[same_node]['contraction'], **G2.nodes[idx]['contraction']}
                else:
                    G2.nodes[idx]['contraction'] = G2.nodes[same_node]['contraction']
                del G2.nodes[same_node]['contraction']
            G2 = nx.contracted_nodes(G2, idx, same_node, self_loops=False)

    # reset graph edge info
    for edge in G2.edges():
        edge_type = G2.edges[edge]['edge_type']
        if edge_type == 'sequential':
            G2.edges[edge]['edge_type_idx'] = np.array([1,0,0,0,0])
        elif edge_type == 'successor':
            G2.edges[edge]['edge_type_idx'] = np.array([0,1,0,0,0])
        elif edge_type == 'left':
            G2.edges[edge]['edge_type_idx'] = np.array([0,0,1,0,0])
        elif edge_type == 'right':
            G2.edges[edge]['edge_type_idx'] = np.array([0,0,0,1,0])
        elif edge_type == 'adjacency':
            G2.edges[edge]['edge_type_idx'] = np.array([0,0,0,0,1])
        else:
            G2.edges[edge]['edge_type_idx'] = np.array([0,0,0,0,0])
            
        edge_direc = G2.nodes[edge[1]]['pos'] - G2.nodes[edge[0]]['pos']
        G2.edges[edge]['direc'] = edge_direc
        G2.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))

    pos_kdtree_keys2 = np.stack(list(dict(G2.nodes('pos')).keys()))
    pos_kdtree2 = KDTree(np.stack(list(dict(G2.nodes('pos')).values())))

    return G2, pos_kdtree_keys2, pos_kdtree2

def convert_graph_as_torch(graph, device):
    graph = graph.copy()
    for node in graph.nodes.values():
        for k in node.keys():
            if type(node[k]) == np.ndarray:
                node[k] = torch.from_numpy(node[k].copy()).to(device)
                
    for edge in graph.edges.values():
        for k in edge.keys():
            if type(edge[k]) == np.ndarray:
                edge[k] = torch.from_numpy(edge[k].copy()).to(device)
                
    return graph

def convert_graph_as_numpy(graph):
    graph = graph.copy()
    for node in graph.nodes.values():
        for k in node.keys():
            if type(node[k]) == torch.Tensor:
                node[k] = node[k].cpu().numpy()

    return graph

def get_current_frenet_coord(xy, G, seg_G, most_nearest_node_id):
    xy = xy.reshape([1,2])
    
    most_nearest_seg_id = G.nodes[most_nearest_node_id]['seg']

    point_line = seg_G.nodes[most_nearest_seg_id]['point_line']
    st = cartesian_to_frenet_approx(point_line, xy)

    return st.squeeze()
    
def get_k_nearest_nodes(xy, G, pos_kdtree, pos_kdtree_keys, k):
    xy = xy.reshape([1,2])

    _, idxs = pos_kdtree.query(xy, k=k)

    nearest_node_ids = pos_kdtree_keys[idxs].reshape(-1)
    #sub_graph = torch_G.subgraph(nearest_node_ids)

    return nearest_node_ids

def get_subgraph(center_pos_xy, G, pos_kdtree_keys, pos_kdtree, nearest_node_num=96):
    _, idxs = pos_kdtree.query(center_pos_xy.reshape(1, -1), k=nearest_node_num)
    nearest_node_ids = pos_kdtree_keys[idxs].reshape(-1)
    H = G.subgraph(nearest_node_ids)
    return H

def get_filtered_subgraph(center_pos_xy, G, pos_kdtree_keys, pos_kdtree, nearest_node_num=96, cutoff_dist=30):
    dist, idxs = pos_kdtree.query(center_pos_xy.reshape(1, -1), k=nearest_node_num)
    nearest_node_ids = pos_kdtree_keys[idxs[dist <= cutoff_dist]].reshape(-1)
    H = G.subgraph(nearest_node_ids)
    return H

def get_front_filtered_subgraph(center_pos_xy, theta, G, pos_kdtree_keys, pos_kdtree, nearest_node_num=96, cutoff_dist=30, margin=-10):
    dist, idxs = pos_kdtree.query(center_pos_xy.reshape(1, -1), k=nearest_node_num)
    nearest_node_ids = pos_kdtree_keys[idxs[dist <= cutoff_dist]].reshape(-1)
    sub_graph = G.subgraph(nearest_node_ids)

    # front filtering
    sub_pos_kdtree_values = np.array(list(dict(sub_graph.nodes('pos')).values()))
    sub_pos_kdtree_keys = np.array(list(dict(sub_graph.nodes('pos')).keys()))

    node_direc = sub_pos_kdtree_values - center_pos_xy.reshape(1, -1)
    vehicle_direc = np.array([np.cos(theta), np.sin(theta)])

    filter_idx = np.where(np.dot(node_direc, vehicle_direc) > margin)[0]
    filtered_nearest_node_ids = sub_pos_kdtree_keys[filter_idx].reshape(-1)
    filtered_sub_graph = sub_graph.subgraph(filtered_nearest_node_ids)

    return filtered_sub_graph

def get_only_frontal_subgraph(center_pos_xy, theta, G, margin=1):
    # front filtering
    sub_pos_kdtree_values = np.array(list(dict(G.nodes('pos')).values()))
    sub_pos_kdtree_keys = np.array(list(dict(G.nodes('pos')).keys()))

    node_direc = sub_pos_kdtree_values - center_pos_xy.reshape(1, -1)
    vehicle_direc = np.array([np.cos(theta), np.sin(theta)])

    filter_idx = np.where(np.dot(node_direc, vehicle_direc) > margin)[0]
    filtered_nearest_node_ids = sub_pos_kdtree_keys[filter_idx].reshape(-1)
    filtered_sub_graph = G.subgraph(filtered_nearest_node_ids)

    return filtered_sub_graph

def extract_graph_features(center_pos_xy, theta, speed, G, pos_kdtree_keys, pos_kdtree, traj_node_id,
                           nearest_node_num=96, cutoff_dist=30, use_node_filter=True):
    cos_theta = np.cos(theta).reshape([-1, 1])
    sin_theta = np.sin(theta).reshape([-1, 1])
    R = np.concatenate([cos_theta, -sin_theta, sin_theta, cos_theta], axis=1).reshape(
        [2, 2])

    # construct graph
    if use_node_filter:
        H = get_front_filtered_subgraph(center_pos_xy, theta, G, pos_kdtree_keys, pos_kdtree, nearest_node_num, cutoff_dist)  # get filtered subgraph
    else:
        H = get_subgraph(center_pos_xy, G, pos_kdtree_keys, pos_kdtree, nearest_node_num, cutoff_dist)  # get subgraph

    adjacency_matrix = nx.to_numpy_array(H, weight=None)  # n x n

    _, idxs = pos_kdtree.query(center_pos_xy.reshape(1, -1))
    most_nearest_node_id = pos_kdtree_keys[idxs].reshape(-1)[0]

    node_num = len(H.nodes())
    edge_feature_num = 2  # direction
    node_feature_num = 6  # node position + on_traffic + on_path + speed + is_nearest

    # construct edge feature matrix
    edge_feature_matrix = np.zeros([node_num, node_num, edge_feature_num])
    nodes = list(H.nodes())
    for edge in H.edges():
        key1, key2 = edge
        i, j = nodes.index(key1), nodes.index(key2)
        diff = H.edges[edge]['direc']
        diff = np.matmul(diff, R)  # apply diff_rot
        edge_feature_matrix[i, j] = diff

    # construct node feature matrix
    node_feature_matrix = np.zeros([node_num, node_feature_num])
    for i, key in enumerate(H.nodes()):
        pos = H.nodes[key]['pos'] - center_pos_xy
        pos_rot = np.matmul(pos, R)  # apply diff_rot
        node_feature_matrix[i, :2] = pos_rot
        node_feature_matrix[i, 2] = 1 if H.nodes[key]['on_traffic_signal'] is not None else 0
        on_path = (key in traj_node_id)
        if 'contraction' in H.nodes[key]:
            on_path = on_path or any([(contraction in traj_node_id) for contraction in H.nodes[key]['contraction']])
        node_feature_matrix[i, 3] = 1 if on_path else 0
        if key == most_nearest_node_id:
            node_feature_matrix[i, 4] = speed / 5.0
            node_feature_matrix[i, 5] = 1

    adjacency_matrix_appended = np.zeros([nearest_node_num, nearest_node_num])
    node_feature_matrix_appended = np.zeros([nearest_node_num, node_feature_num])
    edge_feature_matrix_appended = np.zeros([nearest_node_num, nearest_node_num, edge_feature_num])
    adjacency_matrix_appended[:node_num, :node_num] = adjacency_matrix
    node_feature_matrix_appended[:node_num, :] = node_feature_matrix
    edge_feature_matrix_appended[:node_num, :node_num, :] = edge_feature_matrix

    graph_features = {'adjacency_matrix': adjacency_matrix_appended, \
                  'node_feature_matrix': node_feature_matrix_appended, \
                  'edge_feature_matrix': edge_feature_matrix_appended, \
                  'node_num': node_num, \
                  'most_nearest_node_id':most_nearest_node_id}

    return graph_features, H
