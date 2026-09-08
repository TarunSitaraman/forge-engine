# Releasing

A release is a pushed tag. Everything else is automated, and nothing about
it requires a credential on your machine.

**The tag goes on `forge-engine`, not on the vault.** They are two
repositories with similar names and the vault is the one you are usually
standing in. A tag pushed there does nothing except sit on the wrong repo
looking like a release happened, because the workflow that reacts to it
lives here. It has happened once already. Before tagging, check both:

```bash
git remote -v | head -1
grep '^version' pyproject.toml
```

The first must name `forge-engine` and the second must match the tag you
are about to write.

## The one-time setup

PyPI needs to be told, once, that this repository's release workflow is
allowed to publish `forge-kb`. This is **Trusted Publishing**: PyPI verifies
the workflow's OIDC identity at upload time, so there is no API token stored
in a GitHub secret, in `~/.pypirc`, or anywhere else. Nothing to leak, and
nothing to rotate.

At <https://pypi.org/manage/project/forge-kb/settings/publishing/>, add a
publisher with exactly these values:

| Field | Value |
|---|---|
| Owner | `TarunSitaraman` |
| Repository name | `forge-engine` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Then, in the repository, create an environment named `pypi`
(Settings, Environments, New environment). It needs no secrets and no
variables; naming it is what lets you require a manual approval before a
publish if you ever want one.

## Cutting a release

1. Write the entry in [`CHANGELOG.md`](../CHANGELOG.md) first, from the
   commits since the last tag. Writing it first is what turns "what did I
   change" into a decision about the version number rather than a summary
   after the fact.
2. Set `version` in `pyproject.toml` to match.
3. Commit both, and let CI go green on `main`.
4. Tag and push:

   ```bash
   cd /path/to/forge-engine
   git remote -v | head -1
   git tag -a v0.2.0 -m "0.2.0"
   git push origin v0.2.0
   ```

   Pushed to the wrong repository, delete it from there and start again:

   ```bash
   git push origin :refs/tags/v0.2.0
   git tag -d v0.2.0
   ```

The `release` workflow then checks that the tag matches the version in
`pyproject.toml` (a mismatch fails the job rather than uploading something
nobody meant to), builds the sdist and wheel, runs `twine check`, installs
the wheel on its own in a clean virtualenv and runs `forge demo` in it, and
only then publishes. It drafts a GitHub release with the artifacts attached;
edit the notes from the changelog entry and publish it.

`workflow_dispatch` runs the same build and checks without publishing, which
is the way to rehearse a release without spending a version number.

## After

Check the two things that are easy to assume:

```bash
pip install --upgrade forge-kb && forge --version && forge demo
```

and that the README's first command still describes what a new user gets.
A release whose headline feature is not in the published package is the
failure mode this file exists to prevent: 0.1.0 sat on PyPI for five days
telling people to run a command it did not have.
