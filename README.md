# Solver

## Setup

In case you want to use conda, you can create a new environment with the following command:

```bash
conda env create -f environment.yml
```

In case you want work with pip and virtualenv, a `requirements.txt` file is provided. To create the environment run the following command:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Update environment

In case we need new packages it would be perfect if we always directly add them to the `environment.yml` and `requirements.txt` file accordingly.
