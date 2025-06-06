# Extended Dataset Profile Service

This repository contains the Extended Dataset Profile Service,
which generates machine and human readable meta information about multi
modal datasets.

# User Info

The Extended Dataset Profile Service is shipped as a docker image. The Docker images are found here:
https://hub.docker.com/r/beebucket/edps
Docker images are tagged using the exact Git tag from which they are built.

You can also [create the docker image locally by yourself](#scripts).

The extended data set profile service currently contains four entry points:
- [Job REST API](#job-api): Gets started once. Exposes a REST API and manages queue of jobs.
- [Pontus-X CLI](#pontus-x-cli): Gets invoked once per asset by Pontus-X. For more details check this article on [compute to data.](https://docs.pontus-x.eu/docs/use-cases/compute)
- [Python API](#python-api)
- [Command Line Interface](#command-line-interface)

# Developer Info

## Library Installation

You can install this package in your project with the UV package manager,
but any python package manager should work. We advice to always specify a
tag, so your builds stay reproducible:

```bash
# With UV package manager
uv add <URL> --tag <TAG>

# or with poetry
poetry install <URL>

# or with conda
conda install <URL>

# or with PIP
pip install <URL>
```

## Development setup

The project's dependencies are managed via the uv package manager. Even though, the pyproject.toml is
compliant to other package managers as well, but you might need to add a reference to the pytorch
package sources when installing. We advise using the uv package manager for increased speed and
better support of the pyproject.toml.

[Here you can find information on how to install uv.](https://docs.astral.sh/uv/getting-started/installation).
I strongly advise NOT to install ANYTHING with `pip install uv` in a hosts main python installation.

After you installed uv, you can install this repository as follows. The command will create a ".venv" directory
inside your repository:

```sh
git clone <URL>
cd extended_dataset_profile_service
uv sync --all-extras
```

For linting we use pre-commit. You can enable checking all commits by running this command inside your repository
directory:
```bash
uv run pre-commit install
```

We utilize pytest for the unit tests. You can run the test either through your IDE's test section, or by calling:
```sh
uv run pytest .
```

You can instead install the package directly from the repository, which will result in installing the package and
its entrypoints into your environment:
```sh
# Using UV
uv pip install <URL>

# Or using pip
pip install <URL>
```

## Release Process

To create a release, just push a tag to the repository. The pipeline will then run checks
and create a draft for a release. You can then review it and decide to publish or dump it.

## Creating Your Own Service/Algorithm

The EDPS platform offers two flexible approaches for integrating your own custom algorithms or ontology mappers:

### 1. Fork and Extend the EDPS Repository

You can fork this repository to build upon the existing algorithms or to introduce support for new asset types. This method allows you to directly enhance and tailor the service to your specific needs.

If your implementation could be valuable to others, we warmly encourage you to submit a **Pull Request**. Contributing back to the main EDPS repository helps strengthen the platform for the entire community.

### 2. Develop a Custom Ontology Mapper

Alternatively, you can create a standalone ontology mapper by leveraging the libraries and tools provided by EDPS. This approach is ideal for those who prefer to maintain separate services while still benefiting from the EDPS ecosystem.

For a practical example, please refer to the following example [repository](https://github.com/Mission-KI/LP-MDS-Ontology-Mapper).

## Scripts

See [Scripts README](scripts/README.md).

## Docker

See [Docker README](docker/README.md).

# Job API

See [Job API README](src/jobapi/README.md).

# Pontus-X CLI

See [Pontus-X CLI README](src/pontusx/README.md).

# Python API

You can use the EDPS from python by calling its main interface function:

```python
from pathlib import Path

import edps

async def my_function(input_file: Path, zip_output: Path | IO[bytes], user_data: edps.UserProvidedEdpData):
    ...
    await edps.analyse_asset(input_file, zip_output, user_data)
    ...
```

# Command Line Interface

After having installed it, can just invoke the EDPS from your command line like this:

```bash
# If you installed via uv:
uv run edps_cli --help
# If you installed by any other means:
edps_cli --help
```
