##################################################
# Import Miscellaneous Assets
##################################################
import inspect
# noinspection PyProtectedMember
from typeguard import check_type, typechecked, _CallMemo
import typing


def is_typing(obj, framework):
    try:
        check_type(
            'obj', obj, framework, _CallMemo(
                is_typing, inspect.currentframe()
            )
        )
    except TypeError:
        return False
    else:
        return True


def like_dict(obj, framework, extra_framework_keys=False, extra_obj_keys=False):
    """Checks for agreement between the keys in obj, and framework

    Parameters
    ----------
    obj: Dict
        A dict to be tested against the provided framework
    framework: Dict
        The expected structure and typing that obj should mimic
    extra_framework_keys: Boolean, default=False
        If True, then obj.keys() is allowed to be any subset of framework.keys()
    extra_obj_keys=False: Boolean, default=False
        If True, then obj.keys() is allowed to be any superset of framework.keys()

    Returns
    -------
    Boolean"""
    if not (isinstance(obj, dict) and isinstance(framework, dict)):
        return False

    for framework_key in framework.keys():
        if (framework_key not in obj.keys()) and (extra_framework_keys is False):
            return False

    for obj_key in obj.keys():
        if (obj_key not in framework.keys()) and (extra_obj_keys is False):
            return False

    return True


def like_framework(obj, framework, extra_framework_keys=False, extra_obj_keys=False):
    """Checks whether obj follows the structure and types of framework

    Parameters
    ----------
    obj:
        An object to be tested against the provided framework
    framework:
        The expected structure and typing that obj should mimic
    extra_framework_keys: Boolean, default=False
        If True, and framework (or a child) is of type dict, then obj.keys() is allowed to be any subset of framework.keys()
    extra_obj_keys=False: Boolean, default=False
        If True, and framework (or a child) is of type dict, then obj.keys() is allowed to be any superset of framework.keys()

    Returns
    -------
    Boolean"""
    try:
        #################### framework = standard type ####################
        if isinstance(framework, type):
            if not isinstance(obj, framework):
                return False
        #################### framework = dictionary framework ####################
        elif isinstance(framework, dict):
            if not like_dict(obj, framework, extra_framework_keys=extra_framework_keys, extra_obj_keys=extra_obj_keys):
                return False
            else:
                for framework_key, framework_val in framework.items():
                    if framework_key in obj.keys():
                        if not like_framework(obj[framework_key], framework_val):
                            return False
                    elif extra_framework_keys is False:
                        return False
        #################### framework = something else ####################
        else:
            raise TypeError
    #################### framework = "typing" instance (probably) ####################
    except TypeError:
        try:
            return is_typing(obj, framework)
        except Exception as _ex:
            print('!' * 80)
            print('type(obj): {}     type(framework): {}'.format(type(obj), type(framework)))
            raise _ex

    return True


def is_real_iterable(obj):
    if isinstance(obj, str):
        return False
    return hasattr(obj, '__iter__')


def execute():
    pass


if __name__ == '__main__':
    execute()
