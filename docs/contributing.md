# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/OuyangWenyu/torchhydro/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with `bug` and
`help wanted` is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
`enhancement` and `help wanted` is open to whoever wants to implement it.

### Write Documentation

torchhydro could always use more documentation,
whether as part of the official torchhydro docs,
in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/OuyangWenyu/torchhydro/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to implement.
-   Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up torchhydro for local development.

1.  Fork the torchhydro repo on GitHub.

2.  Clone your fork locally (with the sister packages in sibling directories):

    ```shell
    $ git clone git@github.com:your_name_here/torchhydro.git
    $ git clone git@github.com:iHeadWater/hydrodataset.git
    $ git clone git@github.com:iHeadWater/hydrodatasource.git
    $ git clone git@github.com:iHeadWater/hydroutils.git
    ```

3.  Install your local copy using `uv` (recommended; Python >= 3.10):

    ```shell
    $ cd torchhydro
    $ uv sync --all-extras
    ```

    > The sister packages `hydrodataset`, `hydrodatasource`, and `hydroutils` are configured as editable local dependencies in `pyproject.toml`. They must live in sibling directories of the clone above.

4.  Create a branch for local development:

    ```shell
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5.  When you're done making changes, check that your changes pass the linters and the tests:

    ```shell
    $ uv run flake8 torchhydro tests
    $ uv run black torchhydro tests
    $ uv run isort torchhydro tests
    $ uv run mypy torchhydro
    $ uv run pytest tests
    ```

6.  Commit your changes and push your branch to GitHub:

    ```shell
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

7.  Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the relevant sections of `README.md`.
3.  The pull request should work for Python >= 3.10. Check the CI results and make sure that the tests pass.
