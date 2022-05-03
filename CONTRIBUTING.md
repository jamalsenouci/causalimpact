# Contributing

Welcome to `causalimpact` contributor\'s guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but [other kinds of
contributions](https://opensource.guide/how-to-contribute) are also
appreciated.

If you are new to using [git](https://git-scm.com) or have never
collaborated in a project previously, please have a look at
[contribution-guide.org](https://www.contribution-guide.org/). Other
resources are also listed in the excellent [guide created by
FreeCodeCamp](https://github.com/FreeCodeCamp/how-to-contribute-to-open-source)[^1].

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, [Python
Software Foundation\'s Code of
Conduct](https://www.python.org/psf/conduct/) is a good reference in
terms of behavior guidelines.

## Issue Reports

If you experience bugs or general issues with `causalimpact`, please
have a look on the [issue
tracker](https://github.com/jamalsenouci/causalimpact/issues). If you
don\'t see anything useful there, please feel free to fire an issue
report.

> Please don\'t forget to include the closed issues in your search.
> Sometimes a solution was already reported, and the problem is considered
> **solved**.

New issue reports should include information about your programming
environment (e.g., operating system, Python version) and steps to
reproduce the problem. Please try also to simplify the reproduction
steps to a very minimal example that still illustrates the problem you
are facing. By removing other factors, you help us to identify the root
cause of the issue.

## Documentation Improvements

You can help improve `causalimpact` docs by making them more readable
and coherent, or by adding missing information and correcting mistakes.

`causalimpact` documentation uses as its main
documentation compiler. This means that the docs are kept in the same
repository as the project code, and that any documentation update is
done in the same way as a code contribution.

Documentation is written in the markdown language conforming to the
[CommonMark](https://commonmark.org/) spec

### Using the Github web editor

Please notice that the [GitHub web
interface](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files)
provides a quick way of propose changes in `causalimpact`\'s files.
While this mechanism can be tricky for normal code contributions, it
works perfectly fine for contributing to the docs, and can be quite
handy.

If you are interested in trying this method out, please navigate to the
`docs` folder in the source
[repository](https://github.com/jamalsenouci/causalimpact), find which
file you would like to propose changes and click in the little pencil
icon at the top, to open [GitHub\'s code
editor](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files).
Once you finish editing the file, please write a message in the form at
the bottom of the page describing which changes have you made and what
are the motivations behind them and submit your proposal.

### Using github.dev

[github.dev](https://github.com/github/dev) also provides a convenient way to spin up a vscode editor in your browser for small changes.

### Working locally

When working on documentation changes in your local machine, you can
preview them using your IDE's markdown preview

Example: [vscode guide](https://code.visualstudio.com/docs/languages/markdown)

## Code Contributions

### Internals

The package exports the CausalImpact class which encapsulates the full range of functionality exposed to the user. This class is defined in src/causalimpact/analysis.py which is responsible for orchestrating the causalimpact workflow.

The causal impact workflow is fairly linear and can be broadly represented as

1. check user provided inputs (happens in analysis.py)
2. fit the model (happens in model.py)
3. make the predictions (happens in inferences.py)
4. format, visualise and summarise the output (happens back in analysis.py)

The model fitting is handled by statsmodels.tsa.structural.UnobservedComponents.
The plotting is handled using matplotlib

### Submit an issue

Before you work on any non-trivial code contribution it\'s best to first
create a report in the [issue
tracker](https://github.com/jamalsenouci/causalimpact/issues) to start
a discussion on the subject. This often provides additional
considerations and avoids unnecessary work.

### Create an environment

Before you start coding, we recommend creating an isolated
to avoid any problems with your installed Python packages.
We recommend using vscode's [devcontainers](https://code.visualstudio.com/docs/remote/containers)

### Clone the repository

1.  Create an user account on GitHub if you do not already have one.

2.  Fork the project
    [repository](https://github.com/jamalsenouci/causalimpact): click
    on the _Fork_ button near the top of the page. This creates a copy
    of the code under your account on GitHub.

3.  Clone this copy to your local disk:

        git clone git@github.com:YourLogin/causalimpact.git
        cd causalimpact

4.  You should run:

        pip install -e .

    to be able to import the package under development in the Python REPL.

5.  Install `pre-commit`:

        pip install pre-commit
        pre-commit install

    `causalimpact` comes with a lot of hooks configured to automatically
    help the developer to check the code being written.

### Implement your changes

1.  Create a branch to hold your changes:

        git checkout -b my-feature

    and start making changes. Never work on the main branch!

2.  Start your work on this branch. Don\'t forget to add
    [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings)
    to new functions, modules and classes, especially if they are part
    of public APIs.

3.  Add yourself to the list of contributors in `AUTHORS.md`.

4.  When you're done editing, do:

        git add <MODIFIED FILES>
        git commit

    to record your changes in [git](https://git-scm.com).

    Please make sure to see the validation messages from `pre-commit`\_
    and fix any eventual issues. This should automatically use
    [flake8](https://flake8.pycqa.org/en/stable/)/[black](https://pypi.org/project/black/)
    to check/fix the code style in a way that is compatible with the
    project.

    > **Important** > \
    > Don\'t forget to add unit tests and documentation in case your
    > contribution adds an additional feature and is not just a bugfix.

    Writing a [descriptive commit
    message](https://chris.beams.io/posts/git-commit) is highly
    recommended.

5.  Please check that your changes don\'t break any unit tests with:

        tox

    (after having installed `tox`\_ with `pip install tox` or `pipx`).

    You can also use `tox`\_ to run several other pre-configured tasks
    in the repository. Try `tox -av` to see a list of the available
    checks.

### Submit your contribution

1.  If everything works fine, push your local branch to GitHub with:

        git push -u origin my-feature

2.  Go to the web page of your fork and click \"Create pull request\" to
    send your changes for review.

    Find more detailed information in [creating a
    PR](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).
    You might also want to open the PR as a draft first and mark it as
    ready for review after the feedbacks from the continuous
    integration (CI) system or any required fixes.
    :::

### Troubleshooting

The following tips can be used when facing problems to build or test the
package:

1.  Make sure to fetch all the tags from the upstream
    [repository](https://github.com/jamalsenouci/causalimpact). The
    command `git describe --abbrev=0 --tags` should return the version
    you are expecting. If you are trying to run CI scripts in a fork
    repository, make sure to push all the tags. You can also try to
    remove all the egg files or the complete egg folder, i.e., `.eggs`,
    as well as the `*.egg-info` folders in the `src` folder or
    potentially in the root of your project.

2.  Sometimes `tox`\_ misses out when new dependencies are added,
    especially to `setup.cfg`. If you find
    any problems with missing dependencies when running a command with
    `tox`\_, try to recreate the `tox` environment using the `-r` flag.
    For example, instead of:

        tox -e build

    Try running:

        tox -r -e build

3.  Make sure to have a reliable `tox`\_ installation that uses the
    correct Python version (e.g., 3.7+). When in doubt you can run:

        tox --version
        # OR
        which tox

    If you have trouble and are seeing weird errors upon running
    `tox`\_, you can also try to create a dedicated [virtual
    environment](https://realpython.com/python-virtual-environments-a-primer/)
    with a `tox`\_ binary freshly installed. For example:

        virtualenv .venv
        source .venv/bin/activate
        .venv/bin/pip install tox
        .venv/bin/tox -e all

4.  [Pytest can drop
    you](https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest)
    in an interactive session in the case an error occurs. In order to
    do that you need to pass a `--pdb` option (for example by running
    `tox -- -k <NAME OF THE FALLING TEST> --pdb`). You can also setup
    breakpoints manually instead of using the `--pdb` option.

## Maintainer tasks

### Releases

If you are part of the group of maintainers and have correct user
permissions on [PyPI](https://pypi.org/), the following steps can be
used to release a new version for `causalimpact`:

1.  Make sure all unit tests are successful locally and on CI.
2.  Run `cz bump --changelog` to generate a new tag and an updated changelog.md file
3.  Push the new tag to the upstream
    [repository](https://github.com/jamalsenouci/causalimpact), e.g.,
    `git push upstream v1.2.3`
4.  The github action should detect the new tag and publish to pypi
