# Contributing to Photoholmes

PhotoHolmes is an open-source _python_ library, and welcomes any contribution users and developers wish to make. Through the library's [github](https://github.com/photoholmes/photoholmes), we are open to recieving any issues, bugs, suggestions, and pull requests.

## Develop and Pull-Request to the hub

### 1. Environment setup

The python requirements are `python >= 3.10`. Create a virtual enviroment, either with conda or with pip. 
Activate the enviroment and install the library and required packages.

```
pip install -e .[dev]
```

### 2. Pre-commit hooks


You must also install pre-commit hooks. Pre-commit runs check before a commit to ensure the code quality is being preserved. To install the git hooks, run:
```bash
pre-commit install
```

### 3. Make changes

Remember to follow the guidelines detailed in the documentation. For this, you can read the corresponding `README.md` file available at the section you wish to contribute in. Among other rules, most objects should inherit from an abstract base class. Additionally, we ask the developer to add typing, docstrings, and follow clean-coding practices. Finally, make sure the corresponding documentation is up to date with your changes, and include your contributions in the `factory` and `Registry` correspondingly.

### 4. Submitting Pull Requests

Once you've followed the above steps and are satisfied with your changes:

1. Commit changes to a local branch. Push the branch to the remote repository.
2. Go to the hub repository and create a pull request of your branch towards `develop`.
3. Make sure the merge request pipeline passes the linter and the test.

The PhotoHolmes team is grateful for your contribution!

## License

By contributing to the existing Photoholmes library, you accept that your code will be licensed under the [Apache-2.0 License](LICENSE). You may write to the PhotoHolmes team if this is inconvenient.
