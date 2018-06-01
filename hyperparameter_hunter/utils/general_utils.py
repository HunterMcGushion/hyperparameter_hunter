##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils.boltons_utils import remap

##################################################
# Import Miscellaneous Assets
##################################################
from collections import defaultdict
from datetime import datetime
import string


def deep_restricted_update(default_vals, new_vals):
    """Return an updated dictionary that mirrors `default_vals`, except where the key in `new_vals` matches the path in
    `default_vals`, in which case the `new_vals` value is used

    Parameters
    ----------
    default_vals: Dict, or None
        Dict containing the values to return if an alternative is not found in `new_vals`
    new_vals: Dict
        Dict whose keys are expected to be tuples corresponding to key paths in `default_vals`

    Returns
    -------
    Dict, or None

    Examples
    --------
    >>> deep_restricted_update({'a': 1, 'b': 2}, {('b',): 'foo', ('c',): 'bar'})
    {'a': 1, 'b': 'foo'}
    >>> deep_restricted_update({'a': 1, 'b': {'b1': 2, 'b2': 3}}, {('b', 'b1'): 'foo', ('c', 'c1'): 'bar'})
    {'a': 1, 'b': {'b1': 'foo', 'b2': 3}}
    """
    def _visit_init(path, key, value):
        for _current_key, _current_val in new_vals.items():
            if path + (key,) == _current_key:
                return (key, _current_val)
        return (key, value)

    return remap(default_vals, visit=_visit_init) if default_vals else default_vals


def now_time():
    return datetime.now().time().strftime('%H:%M:%S')


def sec_to_hms(seconds, ms_places=5, as_str=False):
    t_hour, temp_sec = divmod(seconds, 3600)
    t_min, t_sec = divmod(temp_sec, 60)
    t_sec = round(t_sec, ms_places)

    if as_str is True:
        result = ''
        if t_hour != 0:
            result += '{} h, '.format(t_hour)
        if t_min != 0:
            result += '{} m, '.format(t_min)
        if t_sec != 0:
            result += '{} s'.format(t_sec)
        return result
    else:
        return (t_hour, t_min, t_sec)
    # print('Time taken: %i hours %i minutes and %s seconds.' % (t_hour, t_min, round(t_sec, 2)))


def flatten(l):
    return [item for sublist in l for item in sublist]


def type_val(val):
    return type(val), val


def to_standard_string(a_string):
    for to_replace in string.punctuation + ' ':
        a_string = a_string.replace(to_replace, '')

    return a_string.lower()


def standard_equality(string_1, string_2):
    assert (isinstance(string_1, str) and isinstance(string_2, str))

    return to_standard_string(string_1) == to_standard_string(string_2)


def to_even(value, append_char=' '):
    try:
        if len(value) % 2 != 0:
            return value + append_char
    except TypeError:
        if value % 2 != 0:
            return value + 1
    return value


def composed(*decorators):
    def _deco(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f
    return _deco


##################################################
# Recursive Dictionary for Deep Setting
##################################################
recursive_dict = lambda: defaultdict(recursive_dict)


def deep_dict_set(rec_dict, keys, value):
    for key in keys[:-1]:
        rec_dict = rec_dict[key]

    rec_dict[keys[-1]] = value
    return rec_dict


def execute():
    pass


if __name__ == '__main__':
    execute()
