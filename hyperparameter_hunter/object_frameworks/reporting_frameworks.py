##################################################
# Import Miscellaneous Assets
##################################################
from typing import Dict, List, Set, Tuple, TypeVar, Union


def union_of_type_iterables(expected_type):
    return Union[
        List[expected_type],
        Tuple[expected_type, ...],
    ]


##################################################
# AdvancedDisplayLayout
##################################################
class AdvancedDisplayLayoutFrameworks(object):
    ##################################################
    # column_name
    ##################################################
    column_name = Union[
        str
    ]

    ##################################################
    # sub_column_names
    ##################################################
    sub_column_name_entry = Union[
        str,  # Single string - expect actual aggregated attribute name
        union_of_type_iterables(str),  # Iterables of str: [0]=attribute name, (if) [1]=pretty-print name, ignore rest
        # TODO: Consider adding Dict[str, str]
    ]

    sub_column_names = Union[
        union_of_type_iterables(sub_column_name_entry),  # Iterables of "sub_column_name_entry"
        None,  # None - No sub-columns specified
    ]

    ##################################################
    # sub_column_min_sizes
    ##################################################
    sub_column_min_sizes = Union[
        union_of_type_iterables(int),  # Iterables of ints, specifying min size for individual sub-columns
        int,  # Same min size for all sub-columns
        None,  # Default min size for all sub-columns
    ]

    ##################################################
    # Column Entries
    ##################################################
    # FLAG: Below explicit dict declaration breaks "typing" - Need to not rely on typing after this, or don't use actual dict
    # FLAG: Below explicit dict declaration breaks "typing" - Need to not rely on typing after this, or don't use actual dict
    advanced_column_entry = {
        "column_name": column_name,
        "sub_column_names": sub_column_names,
        "sub_column_min_sizes": sub_column_min_sizes
    }
    #
    # simple_column_entry = Union[
    #     str
    # ]
    #
    # ##################################################
    # # Unified Framework
    # ##################################################
    # framework = Union[  # framework = TypeVar[
    #     # FLAG: Check to see if combinations of "advanced_column_entry" and "simple_column_entry" are allowed
    #     # FLAG: Probably need to require all column entries be either "advanced" or "simple" - Try commented "TypeVar" above
    #     union_of_type_iterables(advanced_column_entry),
    #     union_of_type_iterables(simple_column_entry),
    #     # FLAG: Check to see if combinations of "advanced_column_entry" and "simple_column_entry" are allowed
    #     # FLAG: Probably need to require all column entries be either "advanced" or "simple" - Try commented "TypeVar" above
    # ]


def execute():
    pass


if __name__ == '__main__':
    execute()
