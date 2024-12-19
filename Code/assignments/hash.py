import abc
import enum
import hashlib
import math
import sys
import typing
from abc import ABC

import numpy as np

from sklearn import base as sk_base

from . import base


class AbstractRouter(abc.ABC):

    @property
    @abc.abstractmethod
    def n_classes(self) -> int:
        pass


class InnerReducer(abc.ABC):

    @abc.abstractmethod
    def reducer_func(self, X: np.ndarray) -> int:
        pass

    def check_parameters_during_fit(self) -> typing.Optional[ValueError]:
        """
        This can be subclassed to provide error messages during fit.
        :return:
        """
        return None


class InnerHashReducer(InnerReducer):

    def __init__(self, algo: str ):  # , algo_kwargs = None):
        self.algo = algo
        # self.algo_kwargs = algo_kwargs or {}

    def check_parameters_during_fit(self) -> typing.Optional[ValueError]:
        if self.algo not in hashlib.algorithms_available and self.algo != 'siphash':
            return ValueError(f'Unknown algorithm: {self.algo}')
        return None

    def reducer_func(self, X: typing.Union[np.ndarray, typing.Union[int, float]]):
        """
        It receives as input a row, and then hashes this row.
        :param X:
        :return:
        """
        # need to ensure that the input is contiguous (e.g., it is not when
        # producing from poisoning generator).
        # If not, we convert it.
        X_ = X
        if isinstance(X_, np.ndarray):
            if not X_.flags['C_CONTIGUOUS']:
                X_ = np.ascontiguousarray(X_)
        if self.algo in hashlib.algorithms_available:
            hasher = hashlib.new(self.algo, X_)
            return int.from_bytes(hasher.digest(), sys.byteorder)
        else:
            return hash(X_.tobytes())


class Hasher(sk_base.BaseEstimator, sk_base.TransformerMixin):

    def __init__(self, **kwargs) -> None:
        # super().__init__(**kwargs)
        # self.algo = str
        self.reducer: InnerReducer = InnerHashReducer(**kwargs)

    def fit(self, X, y=None) -> "Hasher":
        return self

    def transform(self, X, y=None):

        if len(X.shape) >= 2:
            # We apply the reducer function row-wise and that's it.
            return np.apply_along_axis(self.reducer.reducer_func, 1, X)
        else:
            # if the array is 1-shaped, we have to use vectorize,
            # because apply_along_axis does not work.
            # It is a bit more complex because
            # hashlib is a bit messed up when we receive an int in input
            # so let's skip it so far.
            # return np.vectorize(self.reducer.reducer_func)(X)
            raise ValueError('This transformer accepts only >=2d-shaped arrays.')


class InnerRouterN(InnerReducer, ABC):

    def __init__(self, N: int, **kwargs):
        self.N = N


class InnerRouterModulo(InnerRouterN):

    def reducer_func(self, x: int) -> int:
        return x % self.N


# class InnerRouterFastRange(InnerRouterN):
#
#     def router_func(self, x: int) -> int:
#         # Also here, we need to a bit of magic to ensure that
#         # fastrange works properly due to Python not having fixed-width integers.
#         return ((x & 0xffffffffffffffffffffffffffffffff) * (self.N & 0xffffffffffffffffffffffffffffffff)) \
#                & 0xffffffffffffffff
#         # return (x * self.N) >> 64


class InnerRouterFibonacci(InnerRouterN):
    PHI = 11400714819323198485 & 0xffffffffffffffff

    def __init__(self, N: int, **kwargs):
        super().__init__(N, **kwargs)
        # let's retrieve the number of bits to shift.
        # if N is 32, then the number of bits to shift is 5.
        # To take the top-most bits, we do 64 - number of bits to shift,
        # in this case, 59.
        self.shift_amount = 64 - int(math.log2(self.N))

    def reducer_func(self, x: np.ndarray) -> np.ndarray:
        # this cast is needed because otherwise numpy complains telling that
        # it cannot do bitwise on arrays. Btw, object is the output of hashing.
        x_ = x.astype(object)
        # we receive sin input the number of slots N, e.g., 32
        # we then perform hashing.
        # We just need a lot of cast to uint64.
        # Since Python does not support them natively, we just and with 64 bit mask.
        x_ = x_ & 0xffffffffffffffff
        # here, we first execute >> and then XOR.
        x_ ^= x_ >> self.shift_amount
        x_ = x_ & 0xffffffffffffffff
        # value = x ^ x >> self.shift_amount
        # return (11400714819323198485 * value) >> self.shift_amount
        return ((InnerRouterFibonacci.PHI * x_) & 0xffffffffffffffff) >> self.shift_amount

    def check_parameters_during_fit(self) -> typing.Optional[ValueError]:
        if self.N % 2 != 0:
            return ValueError('N must be a power of 2')
        return None


class SingleValuedRouterType(enum.Enum):
    MODULO = 'MODULO'
    FIBONACCI = 'FIBONACCI'

    def get(self) -> typing.Type[InnerRouterN]:
        if self == SingleValuedRouterType.MODULO:
            return InnerRouterModulo
        return InnerRouterFibonacci


class SingleValuedRouter(sk_base.BaseEstimator, sk_base.TransformerMixin, base.AbstractAssignmentSpread, AbstractRouter):
    
    @staticmethod
    def guarantee_even_fill() -> base.OutputGuarantee:
        return base.OutputGuarantee.DEPEND_ON_DATA
    #
    # @staticmethod
    # def strategy() -> AbstractAssignmentStrategy:
    #     return base.AssignmentStrategySpread()

    @staticmethod
    def strategy() -> base.AssignmentStrategy:
        return base.AssignmentStrategy.DISTRIBUTE

    @property
    def n_classes(self) -> int:
        return self.kwargs['N']

    def __init__(self, algo: SingleValuedRouterType, **kwargs):
        super().__init__(**kwargs)
        self.algo = algo
        self.kwargs = kwargs
        self.router: InnerReducer = None

    def fit(self, X, y=None) -> "SingleValuedRouter":
        self.router = self.algo.get()(**self.kwargs)
        exc = self.router.check_parameters_during_fit()
        if exc is not None:
            raise exc
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray) and len(X.shape) != 1:
            raise ValueError('This transformer accepts only 1d-shaped arrays')

        # We apply the router function row-wise,
        # and we obtain the partition index.
        return np.apply_along_axis(self.router.reducer_func, 0, X)


if __name__ == '__main__':
    from sklearn import pipeline

    p_fibo = pipeline.make_pipeline(Hasher(algo='md5'), SingleValuedRouter(algo=SingleValuedRouterType.FIBONACCI, N=8))
    p_modulo = pipeline.make_pipeline(Hasher(algo='md5'), SingleValuedRouter(algo=SingleValuedRouterType.MODULO, N=8))
    # p_fastrange = pipeline.make_pipeline(Hasher(algo='md5'), SingleValuedRouter(algo=SingleValuedRouterType., N=8))

    data = np.random.rand(50, 20)

    ps = [(p_modulo, 'modulo'), (p_fibo, 'fibonacci'),] #(p_fastrange, 'fastrange')]
    for p, p_name in ps:
        print(p_name)
        print(p.fit_transform(data))
        print()
