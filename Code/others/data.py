import numpy as np
from sklearn import cluster, svm

import aggregators
import assignments
import iops
import pipe
from . import base


step_boundary = pipe.Step(name='boundary', step=iops.IoPComposer(iop_clazz=iops.IoPDistance,
                                                                 iop_kwargs={
                                                                     'how': iops.IoPDistanceType.BOUNDARY,
                                                                     'reverse_how': iops.ReverseType.SUBTRACT_BY_MAX,
                                                                     'inner_kwargs': {
                                                                         # 'distance_metric_exp': 2,
                                                                         'inner': svm.LinearSVC,
                                                                         'inner_kwargs': {'dual': 'auto'},
                                                                         'n_jobs': -1,
                                                                     }
                                                                 },
                                                                 ),
                          post_aggregation=pipe.AggregationFuncPair(
                              aggregation_func=lambda a: np.where((a[..., 0] >= 0.81), 1, 0),
                              aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(args=(a[0],))
                          ),
                          steps_to_aggregate=[-1])

step_cluster = pipe.Step(name='clustering', step=iops.IoPComposer(iop_clazz=iops.IoPDistance,
                                                                  iop_kwargs={
                                                                      'how': iops.IoPDistanceType.CLUSTERING,
                                                                      'reverse_how': iops.ReverseType.SUBTRACT_BY_MAX,
                                                                      'inner_kwargs': {
                                                                          'direction': iops.Direction.FROM_BOTH,
                                                                          'distance_metric_exp': 2,
                                                                          'clustering_clazz': cluster.KMeans,
                                                                          'n_jobs': -1,
                                                                      }
                                                                  },
                                                                  ),
                         post_aggregation=pipe.AggregationFuncPair(
                             aggregation_func=lambda a: np.where(((a[..., 0] <= 0.175) & (a[..., 1] >= 0.82)), 1, 0),
                             aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(args=(a[0],))
                         ),
                         steps_to_aggregate=[-1])

step_neigh = pipe.Step('neigh', step=iops.IoPComposer(iop_clazz=iops.IoPNeighbor,
                                                      iop_kwargs={
                                                          'how': iops.IoPNeighborType.K_DISTANCE,
                                                          'reverse_how': iops.ReverseType.DIVIDE_BY_1,
                                                          'inner_kwargs': {
                                                              'direction': iops.Direction.FROM_BOTH,
                                                              'distance_metric_exp': 2,
                                                              'neighbor_kwargs': {
                                                                  'n_neighbors': 50
                                                              },
                                                              'n_jobs': -1,
                                                          }
                                                      },
                                                      grouper_clazz=iops.GrouperClusteringAutomatic,
                                                      grouper_kwargs={'clustering_kwargs': {'n_jobs': -1}},
                                                      ),
                       post_aggregation=pipe.AggregationFuncPair(
                           aggregation_func=lambda a: np.where((((a[..., 0] <= 0.32) & (a[..., 1] <= 0.11)) | (
                                       (a[..., 0] >= 0.047) & (a[..., 0] <= 0.17) & (a[..., 1] <= 0.32))), 1, 0),
                           aggregation_func_arg_func=lambda a: pipe.ArgFuncOutput(args=(a[0],))
                       ),
                       steps_to_aggregate=[-1])

N_AND_NAME = [(3, '03',),  # c(9, '09'), (15, '15'), (21, '21'), (27, '27'),
              (19, '19'),
              (33, '33')
              ]

def get_pipelines():

    # pipelines for the TSUSC paper aka our baseline
    # pipelines = {}
    pipelines = []
    for n, name in N_AND_NAME:
        p_name = f'tsusc_n{name}'
        #pipelines[p_name] = {
        pipelines.append({
            base.KEY_PIPELINE: pipe.ExtPipeline(name=p_name, steps=[
                pipe.Step('hash', step=assignments.Hasher(algo='md5')),
                pipe.Step('modulo', step=assignments.SingleValuedRouter(N=n,
                                                                        algo=assignments.SingleValuedRouterType.MODULO)),
                pipe.Step('assignment', step=assignments.AssignmentRoundRobinSmart(N=n))
            ]),
            base.KEY_STEP_TYPE: base.KEY_GROUP_TSUSC,
            base.KEY_GROUP: base.KEY_GROUP_TSUSC,
            base.KEY_ROUTING: base.KEY_GROUP_TSUSC,
            base.KEY_N: n
        })

    # pipeline with individual IoPs
    for base_name, single_step in [('boundary', step_boundary),
                                   ('clustering', step_cluster), ('neigh', step_neigh)
                                   ]:
        for n, n_str in N_AND_NAME:
            for last_step_name, last_step in [('rr', assignments.AssignmentRoundRobinSmart),
                                              ('sink', assignments.AssignmentSqueezeSink)
                                              ]:
                p_name = f'risk_one_{base_name}_n{n_str}_{last_step_name}'
                # pipelines[p_name] = {
                pipelines.append({
                    base.KEY_PIPELINE: pipe.ExtPipeline(name=p_name,
                                                        steps=[single_step,
                                                               pipe.Step('assignment', last_step(**{'N': n}))]),
                    base.KEY_STEP_TYPE: base_name,
                    base.KEY_GROUP: 'risk_one',
                    base.KEY_ROUTING: last_step_name,
                    base.KEY_N: n
                })

    # pipeline with all IoPs.
    for n, n_str in N_AND_NAME:
        for last_step_name, last_step in [('rr', assignments.AssignmentRoundRobinSmart),
                                          ('sink', assignments.AssignmentSqueezeSink)
                                          ]:
            p_name = f'risk_three_n{n_str}_{last_step_name}'
                # pipelines[p_name] = {
            pipelines.append({
                base.KEY_PIPELINE: pipe.ExtPipeline(name=p_name,
                                                    steps=[step_boundary, step_cluster, step_neigh,
                                                           aggregators.StepAggregateFlattener(name='flat',
                                                                                              to_aggregate=[0, 1, 2]),
                                                           aggregators.StepAggregateCount(name='count',
                                                                                          to_aggregate=[3]),
                                                           pipe.Step('assignment', last_step(**{'N': n}))
                                                           ]),
                base.KEY_STEP_TYPE: base_name,
                base.KEY_GROUP: 'risk_three',
                base.KEY_ROUTING: last_step_name,
                base.KEY_N: n
            })
    return pipelines