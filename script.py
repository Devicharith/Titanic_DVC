import git
import streamlit as st
import dvc.api
import pandas as pd
import yaml
import json
from contextlib import contextmanager

REPO = git.Repo(".")
MODELS_COMMITS = list(REPO.iter_commits(paths="dvc.lock"))


FIRST_COMMIT = list(REPO.iter_commits())[-1]

@contextmanager
def git_open(path: str, rev: str):
    commit = REPO.commit(rev)
    # Hack to get the full blob data stream: compute diff with initial commit
    diff = commit.diff(FIRST_COMMIT, str(path))[0]
    yield diff.a_blob.data_stream

def _read_train_params(rev: str) -> dict: 
     with git_open("dvc.lock", rev=rev) as file:
        dvc_lock = yaml.safe_load(file)
        return dvc_lock["stages"]["train"]["params"]["params.yaml"]


MODELS_PARAMETERS = {
    commit.hexsha: _read_train_params(rev=commit.hexsha)
    for commit in MODELS_COMMITS
}

def _read_model_evaluation_metrics(rev: str) -> dict:
    with dvc.api.open("scores.json", rev=rev) as file:
        return json.load(file)

MODELS_EVALUATION_METRICS = {
    commit.hexsha: _read_model_evaluation_metrics(rev=commit.hexsha)
    for commit in MODELS_COMMITS
}


experiments = pd.DataFrame([
    {
        "hash": model_commit.hexsha,
        "message": model_commit.message,
        "committed_datetime": str(model_commit.committed_datetime),
        "committer": str(model_commit.committer),
        **MODELS_PARAMETERS[model_commit.hexsha],
        **MODELS_EVALUATION_METRICS[model_commit.hexsha],
    }
    for model_commit in MODELS_COMMITS
])

st.dataframe(experiments)