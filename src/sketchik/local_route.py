"""One local motion contract for normal stages and unfinished-suffix repair."""
from __future__ import annotations

from copy import deepcopy

from .tools import obj


def route_schema(point_schema, max_points=6):
    if type(max_points) is not int or max_points < 1:
        raise ValueError('max_route_points must be a positive integer')
    return obj({'waypoints': {'type': 'array', 'items': deepcopy(point_schema),
        'minItems': 1, 'maxItems': max_points,
        'description': 'Ordered full-pose waypoints of this move. Gripper changes are separate actions in the enclosing sketch.'}}, ['waypoints'])


def expand_stage(plan, selections, conditions, packets, views, source_step, observation_id):
    """Expand the agent's action batch into real execution waypoints."""
    result={'semantic_plan':deepcopy(plan),'plan':deepcopy(plan),
        'source_step':source_step,'observation_id':observation_id,
        'selections':[],'conditions':[],'frozen_packets':[],'cloud_views':[],
        'locations':[],'execution_receipts':[]}
    execution_steps=[];prefix_selections=[]
    for index,step in enumerate(plan['steps']):
        selection=selections[index]
        prefix_selections.append(selection)
        points=selection['waypoints'] if step['primitive']=='move' and selection is not None else [selection]
        for point_index,point in enumerate(points):
            execution_steps.append(deepcopy(step))
            result['selections'].append(deepcopy(point))
            result['conditions'].append(deepcopy(conditions[index]) if point_index==0 else
                {'kind':'previous_motion_converged'})
            result['frozen_packets'].append(deepcopy(packets[index]))
            result['cloud_views'].append(views[index])
            result['locations'].append({'semantic_step_index':index,'waypoint_index':point_index,
                'grounding_source_step':source_step})
    result['plan']['steps']=execution_steps
    result['needs_observation']=any(selection is None for selection in prefix_selections)
    return result

