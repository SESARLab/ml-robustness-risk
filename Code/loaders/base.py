import abc
import dataclasses
import importlib
import sys
import typing

import mashumaro


def _import(name: str):
    if name not in sys.modules:
        try:
            module = importlib.import_module(name)
            assert name in sys.modules
        except Exception as e:
            print(name)
            raise e
    else:
        module = sys.modules[name]
    return module


def import_if_needed(name: str):
    # what we do is split the name according to '.',
    # for instance: 'p.i'.rpartition('.') -> ['p', '.', 'i']
    parts = name.rpartition('.')

    # if we are importing an enum, this fails
    # because the mother module, i.e., the left-most part,
    # contain the module name AND the enum name, like '_assignments.SingleValuedRouterType',
    # and this import is not going to work. So we use a simple heuristic to detect enum
    # imports: whether the child module (in the case of the enum, the value of the
    # enum itself) is uppercase only. This convention holds here.

    # to_import = parts[0]
    # child = parts[-1]

    if parts[-1].isupper():
        # we do another rpartition.
        second_parts = parts[0].rpartition('.')
        to_import = second_parts[0]
        child = second_parts[-1]

        mother_module = _import(to_import)
        enum_clazz = getattr(mother_module, child)

        # now we have to import the enum value.
        return  getattr(enum_clazz, parts[-1])

    # otherwise: normal import.
    mother_module = _import(parts[0])
    child = parts[-1]

    return getattr(mother_module, child)


def fill_kwargs(kwargs: dict):
    keys_to_skip = set()
    new_kwargs = {}
    for k in kwargs.keys():
        if k not in keys_to_skip:
            elem = kwargs[k]
            # if k.startswith('___') and isinstance(elem, str):
            #     # let's see how k is made. If it begins with '___' it refers to a code to be evaluated
            #     k_real_name = k.replace('___', '')
            #     val = eval(elem)
            #     new_kwargs[k_real_name] = val
            #     keys_to_skip.add(k)

            if isinstance(elem, str):
                # see if it is a class to instantiate
                if elem.startswith('__') and not elem.startswith('___'):
                    # now, look if the dict also contain the corresponding kwargs
                    # for this function constructor call.
                    # (If the function is called __Class
                    # then we look for a kwargs named Class_kwargs__
                    # NOTE: this is applicable for class constructors but not necessarily.
                    if kwargs.get(f'{k}_kwargs__') is not None:
                        its_kwargs = kwargs[f'{k}_kwargs__']
                        # drop the key from the kwargs. This way we do not read it once again.
                        keys_to_skip.add(f'{k}_kwargs__')
                        # load this kwargs
                        loaded_kwargs = fill_kwargs(its_kwargs)
                        # and instantiate the class
                        # kwargs[k] = load_func(elem, loaded_kwargs)
                        new_kwargs[k] = load_func(elem, loaded_kwargs)
                        # del new_kwargs[f'{k}_kwargs__']
                    else:
                        # if here, then no kwargs to pass to this function.
                        #  kwargs[k] = load_func(elem, func_kwargs={})
                        new_kwargs[k] = load_func(elem, func_kwargs={})
                elif elem.startswith('_') and not elem.startswith('___'):
                    # if it does not start with '__'
                    # but with '_' then it is a "standard" stuff to "just" important (without instantiating it).
                    # kwargs[k] = import_if_needed(elem.removeprefix('_'))
                    new_kwargs[k] = import_if_needed(elem.removeprefix('_'))
                else:
                    # just a plain string with nothing more to be done.
                    new_kwargs[k] = elem
            elif isinstance(elem, dict):
                # if a dict, we might need to fill it.
                # kwargs[k] = fill_kwargs(elem)
                got = fill_kwargs(elem)
                new_kwargs[k] = got
            else:
                new_kwargs[k] = elem
    # return kwargs
    return new_kwargs


def load_func(name, func_kwargs):
    """
    Load a function, and if the prefix starts with '__' the function is also
    called with func_kwargs
    """
    func = import_if_needed(name.removeprefix('__').removeprefix('_'))
    # then, we check if it is a function or a class.
    if name.startswith('__') and not name.startswith('___'):
        # it is something to invoke
        func = func(**fill_kwargs(func_kwargs) if func_kwargs is not None else {})
    else:
        # it is a function (thus nothing to do).
        pass

    return func


A = typing.TypeVar('A')
B = typing.TypeVar('B')


class RawToParsed(abc.ABC, typing.Generic[B]):

    @abc.abstractmethod
    def parse(self) -> B:
        pass


@dataclasses.dataclass
class FuncPair(mashumaro.DataClassDictMixin, RawToParsed[B]):

    func_name: str
    func_kwargs: typing.Optional[dict] = dataclasses.field(default_factory=dict)

    def parse(self) -> B:
        return load_func(self.func_name, self.func_kwargs)


BASE_OUTPUT_DIR_DATASET = 'Datasets'
BASE_OUTPUT_DIR_OUTPUT = 'Output'
