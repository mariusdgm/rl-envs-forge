# Version 6.0.0 release automation

Prepared from feature-branch commit
`662119dcca674eb05f5249234d5dd3ed6d7146b5`, starting with a clean working tree.
This report describes changes prepared during this task and the behavior that
will run **after a future merge**. No agent commit, push, merge, tag, upload or
GitHub release was executed. During the task HEAD advanced externally to
`0451249858b800443cfb7b848997e8c09b177cf5` (`version bump`), committing most
of the prepared changes. Final working-tree changes preserve the original
`test` required-check name in the workflow/tests and add this report.

## Comparison with v5.23.0

`git show v5.23.0:.github/workflows/citd.yml` shows the original `test -> tag ->
deploy` sequence: tests ran on pushes/PRs, the main-branch job created the
version tag, and Poetry published using `PYPI_USERNAME`/`PYPI_PASSWORD`.
The feature branch had removed both release jobs and retained validation only.

The updated [workflow](../.github/workflows/citd.yml) restores those `tag` and
`deploy` jobs and the original automatic main-branch release model. It retains
the Linux/Windows Python 3.10/3.11/3.12 matrix, slow tests, strict Sphinx build,
citation validation, archive inspection and independent artifact-install tests.
There is no new manual approval, release-dispatch workflow, publishing service,
credential naming scheme or GitHub Release creation step.

## Metadata and changed files

- [pyproject.toml](../pyproject.toml): version `6.0.0`; all dependency constraints,
  extras, build-system requirements and Python support bounds are unchanged.
- [CITATION.cff](../CITATION.cff): adds software version `6.0.0`, preserving the
  author/project metadata. No DOI, archival identifier or release date is invented.
- [CHANGELOG.md](../CHANGELOG.md): moves the prepared changes under `6.0.0`,
  dated `2026.10.26`, retains an Unreleased section, and describes the restored
  automatic release behavior instead of validation-only CI.
- [.github/workflows/citd.yml](../.github/workflows/citd.yml): restored release
  jobs, aggregate validation under the original `test` check name, trusted-main guards, scoped permissions, queued
  concurrency and transfer/verification of the validated artifacts.
- [tests/release_automation_checks.py](../tests/release_automation_checks.py):
  opt-in, offline tests of the actual workflow scripts and configuration. It
  requires an existing tooling interpreter with PyYAML; it adds no library
  dependencies. Git/API writes are mocked and credentials are not read.
- This report records the behavior, verification and remaining operational checks.

`poetry.lock` is intentionally unchanged. Poetry's lock hash covers dependency
configuration, not the root project version; `poetry check --lock` passes.
Its normalized content matches the starting commit. No environment implementation
or dependency file content was changed beyond the root project version above.

## Exact post-merge behavior

1. A merge into `mariusdgm/rl-envs-forge`'s `main` produces a `push` workflow for
   that event's commit SHA. All six compatibility jobs, the Linux slow-test job,
   and the Linux release-check job run. The release-check job builds a wheel and
   source distribution, checks their contents, and tests fresh installations.
2. After those checks pass, the job records the version, event commit SHA and
   SHA-256 hashes of the two distributions in `release-manifest.json`. It uploads
   that exact set as an immutable Actions artifact and exposes its artifact ID.
   An unexpected or stale file in `dist` fails validation rather than being uploaded.
3. The aggregate `test` job preserves the original required-check name and
   requires **every** validation dependency to succeed.
   A failed, canceled or skipped dependency fails this gate. `tag` and `deploy`
   depend on the successful gate.
4. The `tag` job runs only for a non-deletion `push` to this repository's
   `refs/heads/main`. It checks out the event SHA with full history and without
   persisted credentials. For the prepared version it creates the lightweight
   `refs/tags/v6.0.0` at **that exact event SHA**, using GitHub's Git-reference API
   with the job's `GITHUB_TOKEN`. It does not tag a later moving main-branch tip.
5. If `v6.0.0` already targets the same commit, the job permits a publishing retry
   without rewriting the tag. If it targets another commit, the job fails with
   an instruction to increment the version. It never moves or force-updates a tag.
   Consequently, a later main commit must bump the version to create another release,
   matching the original workflow's version-collision policy.
6. `deploy` downloads the **same run's validated artifact by ID**, checks the
   download digest, then independently checks the manifest version, event SHA,
   exact filenames and individual SHA-256 hashes. It does not rebuild the package.
7. Only the final publish step receives the existing `PYPI_USERNAME` and
   `PYPI_PASSWORD` secrets, through Poetry's HTTP-basic environment variables.
   It requires both values to be configured and runs
   `poetry publish --no-interaction --skip-existing` with Poetry 1.6.1.
   It uploads the wheel/sdist, not the JSON manifest. Existing PyPI filenames are
   skipped, allowing same-commit retries to complete a partial upload without
   creating duplicate files. Existing PyPI file contents are not downloaded or
   hash-compared by this workflow. See [Poetry's publish options](https://python-poetry.org/docs/cli/#publish).
8. A release failure leaves the validation result visible; a publish failure
   after tag creation leaves that tag in place. Re-running the failed workflow
   for the same commit is supported. A fresh full rerun rebuilds/revalidates its
   own artifacts; a deploy-only retry reuses the original validated artifact
   while it remains retained by Actions.

The published filenames are:

```text
rl_envs_forge-6.0.0-py3-none-any.whl
rl_envs_forge-6.0.0.tar.gz
```

The workflow serializes runs per workflow/event/ref and does not cancel a running
publication. `queue: max` retains up to 100 pending runs instead of replacing the
previous pending release. This uses GitHub's documented concurrency queue; ordering
follows entry into the queue, and runs beyond its capacity can be canceled.
[GitHub concurrency reference](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

## Triggers and credential boundary

| Event | Validation | Tagging / publishing |
| --- | --- | --- |
| Pull request, including forks | Yes | Never |
| Push to feature/nested branch | Yes | Never |
| Push to this repository's main | Yes | After every required job succeeds |
| Main push in a fork repository | Yes | Never |
| Manual `workflow_dispatch`, including main | Yes | Never |
| Tag push | Not triggered | Never |
| Main deletion | Validation may fail on the deleted ref | Never |

`push.branches: ['**']` includes nested branch names and excludes tag-triggered
runs. No `pull_request_target` or untrusted `workflow_run` publishing path was
introduced. The default token permission is `contents: read`; only `tag` has
`contents: write`. `deploy` has read-only contents/artifact access. All checkout
steps disable credential persistence. Publishing credentials occur only in the
final step of the trusted-main-only deploy job, not tests, builds or PR steps.
These filters and job permission scopes follow the
[GitHub workflow syntax reference](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).

## Local verification

- Nine offline workflow tests passed. They exercise 16 tag/deploy guard contexts,
  nine dependency-failure cases, new/same-commit/conflicting tags, failed tag
  lookups, invalid versions, exact artifact wiring, corrupted files, wrong
  version/SHA and unexpected files. Tag/API commands are mocks; no network
  publishing or real tag creation is involved.
- YAML parsing, boolean handling and compilation of every embedded Python block
  passed using existing tooling (Python 3.11.5 with PyYAML). Guards were evaluated
  for PRs, manual runs, feature/tag refs, fork repositories and deletion events.
- Non-slow project suite on the existing Python 3.10.6 Poetry environment:
  **1,150 passed, four slow tests deselected**, 46 existing warnings, 32.30 seconds.
  The slow suite remains an explicit CI requirement; it was not rerun for these
  workflow/metadata-only changes.
- `poetry check --lock` passed and `poetry version --short` reports `6.0.0`.
  Dependency tables and the lockfile were compared with the starting commit;
  no environment implementation has a diff.
- Both 6.0.0 artifacts built successfully with existing Poetry 1.6.1. Archive
  inspection verifies wheel source against the checkout, required source files
  in the sdist, artifact version `6.0.0` and unchanged dependency metadata.
- The workflow's manifest creation and verification blocks also passed against
  those real wheel/sdist files in an isolated local directory, using an explicitly
  simulated event SHA (`ffffffffffffffffffffffffffffffffffffffff`). No local
  dirty-tree build is represented as an already committed/merged release.
- Strict Sphinx (`-E -b html -W --keep-going`) passed with the existing environment.
- `git diff --check` passed. Local Git line-ending conversion notices are not
  whitespace errors. A build initially reported the sandbox's repository ownership
  mismatch; the repeat build used process-scoped `safe.directory` configuration,
  without changing global Git settings.

Local verification commands include:

```powershell
& 'C:/Users/Chainsword/anaconda3/python.exe' -B tests/release_automation_checks.py
poetry check --lock
poetry version --short
$env:SDL_VIDEODRIVER = 'dummy'
$env:SDL_AUDIODRIVER = 'dummy'
$env:MPLBACKEND = 'Agg'
poetry run python -B -m pytest tests -m 'not slow' -q --tb=short -p no:cacheprovider -o addopts=
poetry run python -B -m sphinx -E -b html -W --keep-going docs docs/_build/release_600/html
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = 'D:/Work/repos/RL/rl-envs-forge'
poetry build
git -c safe.directory=D:/Work/repos/RL/rl-envs-forge diff --check
```

Generated archives and inspection results are under the ignored
`docs/_build/release_600/` directory. Only the newly built 6.0.0 wheel and sdist
are supplied to archive inspection; older local `dist` artifacts are preserved.

## Operational checks before merge

The workflow supplies PR checks and preserves the original required-check name
`test`, now aggregating all validation jobs. **GitHub branch protection/rulesets
must require `test` before merging into main**, preferably with branches up to
date. An existing rule requiring the original `test` check can continue to use
that name. Workflow YAML cannot
itself prevent a maintainer or bypass permission from merging. Repository settings
were not inspected or changed during this task.

The repository must permit the tag job's `contents: write` token to create version
tags, and the existing PyPI secrets must be valid publishing credentials. These
secret values were not read. Tag-protection rules and secret/index permissions
remain operational settings, not locally verified facts. No remote tag inventory
or live PyPI publication was queried or attempted.

The hosted workflow, its six-platform matrix, network installs and credentialed
release steps have not been executed here. The external actionlint download was
declined. The installed VS Code GitHub Actions language server was blocked by
sandbox access, and the request to run it outside the sandbox was also declined.
Neither validator is claimed to have passed. Local parser/script/policy checks
passed; the first hosted run remains the final integration check for Actions.
