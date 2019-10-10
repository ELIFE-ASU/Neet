# Contributing to Neet

Thanks for taking the time to make Neet better! :tada: :smile:

> **Note**: These contributing guidelines are adapted from the [Atom Project's Contributing Guidelines](https://github.com/atom/atom/blob/master/CONTRIBUTING.md).

#### Table of Contents

[Code of Conduct](#code-of-conduct)

[How can I contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Suggesting Features](#suggesting-features)
  * [Your First Code Contribution](#your-first-code-contribution)
  * [Pull Requests](#pull-requests)

[Style Guides](#style-guides)
  * [Git Commit Messages](#git-commit-messages)
  * [Python Style Guide](#python-style-guide)
  * [Documentation Style Guide](#documentation-style-guide)
  
[Additional Notes](#additional-notes)

## Code of Conduct

This project, and everyone participating in it, is governed by the [Neet Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior to [emergence@asu.edu](mailto:emergence@asu.edu).

## How can I contribute?

### Reporting Bugs

This section guides you through submitting a bug report for Neet. Following these guidelines helps maintainers and
the community understand your report, reproduce the behavior, and find related reports.

Before creating bug reports, perform a [cursory search](https://github.com/search?utf8=%E2%9C%93&q=+is%3Aissue+repo%3AELIFE-ASU%2FNeet&type=)
to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the
existing issue instead of opening a new one. If you find a **closed** issue that seems like it is the same thing
that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### How do I submit a bug report?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). After you are sure the bug is either
new, or needs to be readdressed, create an issue and provide the following information by filling in
[the template](https://github.com/elife-asu/neet/blob/master/.github/ISSUE_TEMPLATE/bug_report.md).

Explain the problem and include addition details to help maintainers reproduce the problem:

  * **Use a clear and descriptive title** for the issue to identify the problem.
  * **Describe the exact steps which reproduce the problem** in as many details as possible. For example, include a
    minimal script to reproduce the bug.
  * **Describe the behavior you observed from the script** and point out exactly what the problem is with that
    behavior. For example, include any output you observe and explain what is wrong with it.
  * **Explain what behavior you expected to see instead and why.**
  
Provide more context by answering these questions:

  * **Did the problem start happening recently** (e.g. after updating to a new version of Neet) or was this always
    a problem?
  * If this problem started recently, **can you reproduce the problem in an older version of Neet?** What is the
    most recent version of Neet which does not have this bug?
  * **Can you reliably reproduce the issue?** If not, provide details about how often the problem happens and under
    which conditions it typically occurs.
  * If the problem is related to working with external resources (e.g. data files, network connections, etc...),
    **does the problem happen for all resources, or only some?** For example, is there a particular data file that
    seems to cause problems, or are all data file an issue?
    
Include details about your configuration

  * **Which version of Neet are you using?**
  * **What's the name and version of the Operating System you are using?**

### Suggesting Features

This section guides you through submitting an feature suggestion for Neet, including completely new features and
minor improvements to existing functionality. Following these guidelines helps maintainers and the community
understand your suggestion, find related suggestions, and prioritize feature development.

Before creating a feature request, perform a [cursory search](https://github.com/search?utf8=%E2%9C%93&q=+is%3Aissue+repo%3Aelife-asu%2Fneet+label%3A%22feature+request%22+&type=) to see if it has
already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

#### How do I submit a feature request?

Feature requests are tracked [GitHub issues](https://guides.github.com/features/issues/). After you are sure the
request is not a duplicate, create an issue and provide the following information by filling in
[the template](https://github.com/ELIFE-ASU/Neet/blob/master/.github/ISSUE_TEMPLATE/feature_request.md).

  * **Use a clear and descriptive title** for the issue to identify the suggestion.
  * **Provide a description of the feature** in as much detail as possible.
  * **Propose an API for the feature** to demonstrate how that feature fits in with the rest of Neet.
  * **Give and example usage** for the proposed API. Of course, the output is not necessary.
  * **Reference any resources** on which the feature is based. The references should any mathematical details
    necessary for implementing the feature, e.g. defining equations.

### Your First Code Contribution

Your contributions are more than welcome!  before you get started It's also advisable that you read through the
[API documentation](https://neet.readthedocs.io/en/latest/api.html) to make sure that you fully understand how
the various components of Neet interact.

For external contributions, we use [GitHub forks](https://guides.github.com/activities/forking/) and
[pull requests](https://guides.github.com/activities/forking/#making-a-pull-request) workflow. To get started
with contributing code, you first need to fork Neet to one of your accounts. As you begin development, have
several recommendations that will make your life easier.

 * **Do not work directly on master.** Create a branch for whatever feature or bug you are currently working on.
 * **Create a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/)** after
   you first push to your fork. This will ensure that the rest of the Neet community knows that you are working
   on a given feature or bug.
 * **Fetch changes from [ELIFE-ASU/Neet](https://github.com/ELIFE-ASU/Neet)'s master branch often** and merge
   them into your working branch. This will reduce the number and severity of merge conflicts that you will have
   to deal with. [How do I fetch changes from ELIFE-ASU/Neet?](#how-do-i-fetch-changes-from-elife-asuneet)

### Pull Requests

The Fork-Pull Request process described here has several goals:

  * Maintain Neet's quality
  * Quickly fix problems with Neet that are important to users
  * Enage the community in working to make Neet as near to perfect as possible
  * Enable a sustainable system for Neet's maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

  1. **Use a clear and descriptive title** for your pull request.
  2. Follow all instructions in the [pull request template](https://github.com/ELIFE-ASU/Neet/blob/master/.github/pull_request_template.md).
  3. Follow the [styleguides](#styleguides)
  4. After you submit your pull request, verify that all
     [status checks](https://help.github.com/articles/about-status-checks/) are passing.
     <details>
       <summary>What if the status checks are failing?</summary>
       If a status check is failing, it is your responsibility to fix any problems. Of course the maintainers
       are here to help, so please post a comment on the pull request if you need any support from us. If you
       believe that the failure is unrelated to your change, please leave a comment on the pull request
       explaining why you believe that to be the case. A maintainer will re-run the status checks for you. If
       we conclude that the failure was a false positive, then we will open an issue to track that problem with
       our own status check suite.
     </details>

## Style Guides

### Git Commit Messages

* Use the present tense ("Add c-sensitivity" not "Added c-sensitivity")
* Use the imperative mood ("Add fix_canalizing parameter..." not "Adds fix_canalizing parameter...")
* Limit the first line to 72 characters or less
* Reference isses and pull requests liberally after the first line
* When only changing documentation, include `[ci skip]` in the commit title
* Consider starting the commit message with an applicable emoji:
    - :art: `:art:` when improving the format/structure of the code
    - :racehorse: `:racehorse:` when improving performance
    - :memo: `:memo:` when writing documentation
    - :penguin: `:penguin:` when fixing something on Linux
    - :apple: `:apple:` when fixing something on maxOS
    - :checkered_flag: `:checkered_flag:` when fixing something on Window
    - :bug: `:bug:` when fixing a bug
    - :fire: `:fire:` when removing code or file
    - :green_heart: `:green_heart:` when fixing the CI build
    - :heavy_check_mark: `:heavy_check_mark:` when adding tests
    - :arrow_up: `:arrow_up:` when upgrading dependencies
    - :arrow_down: `:arrow_down:` when downgrading dependencies
    - :shirt: `:shirt:` when dealing with linter warnings

### Python Style Guide

All Python code must adhere to the [PEP 8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).

To enforce this, we use both [autopep8](https://pypi.org/project/autopep8/) and
[flake8](https://pypi.org/project/flake8/). We recommend running these **before** each commit (consider setting
up a [pre-commit hook](https://git-scm.com/docs/githooks#_pre_commit)). To run them manually:

```shell
$ autopep8 --in-place --recursive neet test
$ flake8 neet test
```

### Documentation Style Guide

* Use [Sphinx](https://www.sphinx-doc.org/en/master/)
    - Use [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) whenever possible
    - Use [doctest](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) in lieu of simple code blocks
* Use [reStructuredText](http://docutils.sourceforge.net/rst.html)
* Use [Markdown](https://daringfireball.net/projects/markdown) for file unrelated to live documentation
* Each module, class, function and method should be documented using a Python docstring.
* Be very **liberal** with examples in documentation
* Use [math](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#math) whenever applicable.

#### Modules

All modules should have a docstring which roughly adheres to the following template::

```python
"""
.. currentmodule:: neet.boolean.logicnetwork

.. testsetup:: logicnetwork

    from neet.boolean.examples import myeloid
    
If the module contains more than computational unit (classes, functions, etc...) then
you should describe the relationship between the components and provide a list of them
using autosummary:

.. autosummary::
    :nosignatures:
    
    LogicNetwork
    myfunc
    
If there are multiple classes, it is ideal to provide an inheritance diagram (if applicable):

.. inheritance-diagram:: neet.boolean.logicnetwork
    :parts: 1
"""
```

#### Classes

All classes should have a docstring which roughly adheres to the following template:

```python
class LogicNetwork(BooleanNetwork):
    """
    A short, one-sentence description. Follow up with as much detail as you reasonbly can
    about the inheritance structure and methods provided by the class. This is a good
    place to include an inheritance diagram and list of methods/properties/attributes of
    the class:
    
    .. inheritance-diagram:: LogicNetwork
        :parts: 1
    
    .. autosummary::
        :nosignatures:
        
        update
        
    Move on to describe initialization. The ``__init__`` method does not need a docstring since
    you'll be describing what it does here. Include examples of initialization:
    
    .. rubric:: Examples
    
    .. doctest:: logicnetwork
    
        >>> LogicNetwork([((0, 1), {'00', '11'}), ((0,1), {'01','10})])
        <neet.boolean.logicnetwork.LogicNetwork object at 0x...>
        
    Also, don't be afraid to use inline :math:`E = mc^2` or block math:
    
    .. math::
    
        s(f, x) = \\sum_{j = 1}^N f(x) \\oplus f(x \\oplus e_j)
        
    Finish up with the parameters and error conditions
    
    :param table: the first argument
    :type table: list, numpy.ndarray
    :raises IndexError: under some condition
    """
    
    def __init__(self, table):
        pass
"""
```

#### Functions and Methods

With the exception of `__init__` (see (Classes)[#classes]), all functions and methods should have
a docstring which roughly adhere to the following template:

```python
def update(self, state):
    """
    A short, one-sentence description. Follow up with as many details as you can, within reason.
    
    .. rubric:: Examples
    
    .. doctest:: logicnetwork
    
        >>> myeloid.update([0,0,1,0,0,1,0,0,1,0,0])
        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    
    :param x: a description of the first argument
    :type x: list, numpy.ndarray
    :return: a description of the return value and type (:class:`numpy.ndarray`)
    :raises ValueError: under some condition
    
    .. seealso:: :meth:`LogicNetwork.example_method`
    """
    pass
```

#### Using [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)

All of the API documentation lives within
[docs/source/api](https://github.com/ELIFE-ASU/Neet/tree/master/docs/source/api). Each module gets it's own
RST file, with the files layed out on disk similar to the structure in the source directory. While we are
using autodoc to extract the docstrings from the python source, we don't do that automatically. Instead, we
prefer to be able to specify the order of the various methods, functions, properties, etc... on the page. This
means that you'll have to add a directive to the corresponding documentation source if you add a new entity to
Neet. For example, the [docs/source/api/boolean/network.rst](https://github.com/ELIFE-ASU/Neet/blob/master/docs/source/api/boolean/network.rst) is formatted as

```rst
BooleanNetwork
--------------

.. automodule:: neet.boolean.network
   :synopsis: Generic Boolean Networks

.. autoclass:: neet.boolean.BooleanNetwork

   .. automethod:: subspace

   .. automethod:: distance

   .. automethod:: hamming_neighbors
```

If you add a new attribute or method to
[`neet.boolean.network.BooleanNetwork`](https://github.com/ELIFE-ASU/Neet/blob/ff23e738ee5cb42172a4d0e4b78f18f6c04131ba/neet/boolean/network.py#L14),
you will need to add a corresponding `.. autoattribute::` or `.. automethod::` directive to the above
file.

## Additional Notes

### How do I fetch changes from ELIFE-ASU/Neet?

After you have cloned your fork, add the [ELIFE-ASU/Neet](https://github.com/ELIFE-ASU/neet) as a remote:
```shell
$ git add remote elife https://github.com/ELIFE-ASU/Neet
```
To fetch changes from ELIFE-ASU/Neet's master branch:
```shell
$ git fetch elife master
```
This will get all of the changes from the main repository's master branch, but it will not merge any of those
changes into your local working branchs. To do that, use `merge`:
```shell
$ git checkout master
$ git merge elife/master
...
```
You can then merge the changes into your feature branch (say `csensitivity`)
```shell
$ git checkout csensitivity
$ git merge master
```
and then deal with any merge conflicts as usual.
