# Contributing Guidelines

Hey, you! I want you to contribute! If you think HyperparameterHunter doesn't need or want your 
contribution, let me assure you that is not the case. We will gladly accept any form of 
contribution you can offer, no matter how seemingly insignificant it might seem. No contribution 
is insignificant, and all your help is greatly appreciated. I look forward to working with you to 
make HyperparameterHunter the revolutionary hyperparameter optimization library I know it can be. 
But it can't get there without **your** help! 

#### Table of Contents
[Getting Started](#getting-started)

[How Can I Contribute?](#how-can-i-contribute)

* [Test Other Libraries](#test-other-libraries)
* [Report Issues](#report-issues)
* [Triage and Label Issues](#triage-and-label-issues)
* [Your First Code Contribution](#your-first-code-contribution)

[Python Style Guide](#python-style-guide)

* [Code](#code)
* [Imports](#imports)
* [Comments](#comments)

[Docstring Style Guide](#docstring-style-guide)

* [Documenting Classes](#documenting-classes)
* [Documenting Modules](#documenting-modules)

## Getting Started

In order to develop HyperparameterHunter, follow these steps to setup your environment:

1. Fork [HyperparameterHunter][hh_home] on GitHub
2. Clone the Git repo:
    ```
    $ git clone https://github.com/$USERNAME/hyperparameter_hunter
    $ cd hyperparameter_hunter
    ```
3. Setup the virtual environment with dependencies and development tools
    ```
    $ make dev
    $ source env/bin/activate
    ```
4. Format your code (using [Black][black] and [isort][isort]), then run linter, and unit tests
    ```
    $ make format
    $ make lint test
    ```

## How Can I Contribute?

### Test Other Libraries
If you've noticed that your favorite machine learning library isn't listed anywhere in the README, 
or it does'nt have any examples in the 'examples' directory, test it out! If it works, awesome! 
Let us know, and maybe write an example for it, so we can include it in our list of tested 
libraries. If it doesn't work, also let us know, and try to figure out what might be necessary to 
get that one special library to work with HyperparameterHunter.

### Report Issues
Head over to the repository's [issue tracker][hh_issues], and start reporting bugs and suggesting 
enhancements! **Please search for closed and open issues before opening new ones!** All new issues 
should comply with the following guidelines:

*  Absolutely, definitely not a duplicate issue
*  Appropriately labeled
*  For feature requests, provide example use cases and explain *why* you need the feature
*  For bug reports, provide information about your configuration (OS, versions, etc.) and a script 
to reproduce the issue
    * Script should be a [minimal, complete, and verifiable example][mvce], and runnable as-is    

### Triage and Label Issues
Organization of issues is critical. Therefore, any help assigning priorities and other labels to 
issues is a valuable contribution. Labeling duplicated issues is also very helpful to ensure a 
neat and tidy issue tracker that helps us get the important stuff done. This organization of issues 
also includes showing your support for issues that you feel are important in order to draw more 
attention to them.

### Your First Code Contribution
If you want to ease into your first code contribution, these are some good places to start:

* Unit Tests
    * More tests to ensure HyperparameterHunter is running smoothly is always a good thing
    * Use Python's [Coverage][py_coverage] tool to determine which modules need tests
        * `coverage run --source=hyperparameter_hunter setup.py test`
        * Might help to add the following flag: `--omit=hyperparameter_hunter/env/*`
* Documentation
    * Look for [issues][hh_issues] with a "Documentation" label
    * Use [DocstrCoverage][docstr_coverage] to quickly find code that needs documentation
        * `docstr-coverage hyperparameter_hunter --skipmagic --skipclassdef`
        * Might help to add the following flag: `--exclude='env/*|HyperparameterHunterAssets/*'`
* "Easy" Issues
    * Look for [issues][hh_issues] that have an "Easy" label, or for issues that sound easy to you
    * Also, check [this board][hh_contrib] for our todo list, including more long-term contributions

## Python Style Guide

### Code
HyperparameterHunter uses [Black][black] code formatting, and compliance is verified for all 
commits. One notable modification to the Black defaults, is `line-length`, which is set to a 
maximum of 100. Aside from Black's style conventions, what follows are a few additional stylistic 
choices for HyperparameterHunter:
 
Naming should be as descriptive as necessary. This can result in what some might consider to be 
overly-verbose variable names, but they are formatted this way to convey the role of the variable 
as clearly as possible. Classes should be named following CamelCase conventions, with the first 
letter capitalized. All other variables should be named following snake case conventions, using 
all lowercase letters, with words separated by underscores.

String formatting should be done using `'{}'.format()` syntax, or using f-strings. Do not use 
"old style" string formatting (% operator).

### Imports
Imports should be formatted thusly: 

```python
# <special imports (like __future__) go here if necessary>

##################################################
# Import Own Assets
##################################################
# <hyperparameter_hunter imports go here>
# ...

##################################################
# Import Miscellaneous Assets
##################################################
# <miscellaneous imports go here>
# ...

##################################################
# Import Learning Assets
##################################################
# <imports from machine learning libraries go here>
# ...
```

Imports within each of the three sections should be ordered alphabetically (A-Z, preceded by 
punctuation marks and numbers). The alphabetical ordering should be done with respect to the 
module name, then to the assets being imported. When ordering imports, disregard the import 
statement style: `import ...`, or `from ... import ...`. Each of the three sections is separated 
by a single newline, and each of the section border lines consists of 50 consecutive octothorpe 
characters.

### Comments
Normal comments should start with a single octothorpe character, and a single space, followed by 
the comment content. Capitalization should follow standard sentence capitalization rules, without 
ending punctuation. Variable names are exempt from the capitalization rules.

Todo, and flag comments should be formatted as follows:

```python
# TODO: <Comment content>
# FLAG: <Comment content>
```

The three-line title comments described in the above [Import Style Guide](#imports) can also be 
used to mark significant separations in code. As noted above, they should span three lines, in 
which the first and third lines consist solely of 50 octothorpe characters, and the second line 
starts with a single octothorpe and a space, followed by the comment content. The comment content 
in the second line should be capitalized according to APA title capitalization guidelines. 

Single-line subtitle comments can also be used to mark other separations in code for clarity. They 
should span a single line and start with 20 octothorpe characters, followed by a space, followed 
by the comment content, then ending with a space, followed by 20 octothorpes. Just like the 
three-line title comments described above, comment content should be capitalized according to APA 
title capitalization rules (except for variable names). They are formatted thusly: 

```python
#################### <Comment Content> ####################
```

For examples of both the three-line title comments, and the single-line subtitle comments, 
see ```:mod:`hyperparameter_hunter.environment` ```

## Docstring Style Guide
All code contributions **must** be documented in accordance with these guidelines. We use 
NumPy-style docstrings. For examples on proper docstring-formatting, refer to almost any docstring 
in the repository, or to these [example docstrings][np_doc_examples], or [this guide][np_doc_guide].

All documented parameters, return values, yield values, etc. should include a description of the 
value, in addition to its name and type. These values should also be documented in the order of 
their appearance in the code.

Wherever appropriate, include cross-references to modules, classes, methods, functions, and 
attributes. There are plenty of examples of this in the repository, but the general form is 
```:role:`identifier` ```, in which `identifier` is a period-separated path to the object being 
referenced, and `role` is one of the following columns that correspond to the value below it:

|  mod   | class |  meth  |   func   |   attr    |
|--------|-------|--------|----------|-----------|
| module | class | method | function | attribute |

For additional information, see [this guide][sphinx_refs]

### Documenting Classes
When documenting classes, barring extenuating circumstances, the class should be documented in its 
```:meth:`__init__` ```, rather than below the class definition itself.

### Documenting Modules
Files should include module docstrings at the top of the file that include a description of the 
module's purpose/tasks. Usually, module docstrings should include a "Related" section, identifying 
other modules to which the module is related. If appropriate, a "Notes" section may be included as 
well to draw attention important information.

See the module docstring of ```:mod:`hyperparameter_hunter.experiments` ``` for an example.

[hh_contrib]: https://github.com/HunterMcGushion/hyperparameter_hunter/projects/2
[hh_home]: https://github.com/HunterMcGushion/hyperparameter_hunter
[hh_issues]: https://github.com/HunterMcGushion/hyperparameter_hunter/issues
[docstr_coverage]: https://github.com/HunterMcGushion/docstr_coverage

[black]: https://github.com/ambv/black 
[isort]: https://pypi.org/project/isort/
[py_coverage]: https://coverage.readthedocs.io/en/coverage-4.5.1a/

[sphinx_refs]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects
[np_doc_examples]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
[np_doc_guide]: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
[mvce]: https://stackoverflow.com/help/mcve
