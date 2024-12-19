import enum

import pipe


def execute_pipeline(X, y, p: pipe.ExtPipeline):
    return p.fit_transform(X, y)[0]


def get_n_classes(p: pipe.ExtPipeline):
    # if isinstance(p, utils.FinalPipeline) or isinstance(p, pipe.ExtPipeline):
    #     return p.steps[-1].step.n_classes
    # else:
    #     return p.named_steps['last'].n_classes
    return p.steps[-1].step.n_classes


def get_weights(p: pipe.ExtPipeline, N: int):
    last = p.steps[-1].step
    if hasattr(last, 'get_weights'):
        return last.get_weights()
    else:
        return 1/N


class VotingType(enum.Enum):
    HARD = 'hard'
    SOFT = 'soft'
