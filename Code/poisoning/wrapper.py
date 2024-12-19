import typing

import poisoning.selectors
from . import base


TSelector = typing.TypeVar('TSelector', bound=poisoning.selectors.AbstractSelector)
TPerformer = typing.TypeVar('TPerformer', bound=poisoning.performers.AbstractPerformer)
TPerformInfo = typing.TypeVar('TPerformInfo', bound=base.PerformInfoProtocol)
TSelectionInfo = typing.TypeVar('TSelectionInfo', bound=base.AbstractSelectionInfo)


# here I really would like to be able to constrain these generics,
# ensuring that selection info is compatible with selection input. But this is not Rust :)


class Poisoning(typing.Generic[TSelector, TPerformer]): # , TPoisoningInput]):

    def __init__(self,
                 selection_info: TSelectionInfo,  #: TPoisoningInput,
                 perform_info: TPerformInfo,
                 selector: TSelector,
                 performer: TPerformer
                 ):
        self.perform_info = perform_info
        self.selection_info = selection_info
        self.selector = selector
        self.performer = performer

    def fit(self, X, y) -> "Poisoning":
        self.selector.fit(X=X, y=y, selection_info=self.selection_info)
        self.performer.fit(X=X, y=y, specific_args=self.perform_info)
        return self

    def transform(self, X, y):
        selected_idx = self.selector.predict(X=X, y=y, selection_info=self.selection_info)
        X_, y_ = self.performer.transform(X=X, y=y, selected_idx=selected_idx, specific_args=self.perform_info)
        return X_, y_
