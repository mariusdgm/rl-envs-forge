"""Offline workflow checks; run with a tooling Python containing PyYAML.

python tests/release_automation_checks.py

All git/API operations are mocked. No tags, uploads or credentials are used.
This opt-in tooling check adds no library or pytest dependencies.
"""


def main():
    import contextlib
    import copy
    import io
    import json
    import os
    from pathlib import Path
    import re
    import subprocess
    import tempfile
    from types import SimpleNamespace
    import unittest
    from unittest.mock import patch
    import zipfile

    import yaml

    root = Path(__file__).resolve().parents[1]

    class ActionsLoader(yaml.SafeLoader):
        pass

    # YAML 1.2 booleans: PyYAML's default YAML 1.1 interprets the key 'on' as True.
    ActionsLoader.yaml_implicit_resolvers = {
        key: [(tag, regex) for tag, regex in values if tag != 'tag:yaml.org,2002:bool']
        for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    ActionsLoader.add_implicit_resolver('tag:yaml.org,2002:bool',
                                      re.compile(r'^(?:true|false)$'), list('tf'))
    workflow = yaml.load((root / '.github/workflows/citd.yml').read_text(), Loader=ActionsLoader)
    jobs = workflow['jobs']

    def step(job, name):
        return next(item for item in jobs[job]['steps'] if item.get('name') == name)

    def python_block(job, name):
        lines = step(job, name)['run'].splitlines()
        assert lines[0] == "python - <<'PY'" and lines[-1] == 'PY'
        return compile('\n'.join(lines[1:-1]), f'<workflow:{name}>', 'exec')

    blocks = {
        'manifest': python_block('release-checks', 'Record validated release artifacts'),
        'validation': python_block('validation', 'Require every validation job to pass'),
        'tag': python_block('tag', 'Create or reuse version tag'),
        'verify': python_block('deploy', 'Verify release identity and checksums'),
    }
    trusted = dict(GITHUB_EVENT_NAME='push', GITHUB_REF='refs/heads/main',
                   GITHUB_REPOSITORY='mariusdgm/rl-envs-forge', GITHUB_SHA='a' * 40,
                   RELEASE_VERSION='6.0.0')
    artifacts = root / 'docs/_build/release_600'
    artifacts.mkdir(parents=True, exist_ok=True)

    class ReleaseChecks(unittest.TestCase):
        def execute(self, name, environment):
            with patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(io.StringIO()):
                exec(blocks[name], {'__name__': '__main__'})

        def test_events_and_validation_dependencies(self):
            self.assertEqual(set(workflow['on']), {'push', 'pull_request', 'workflow_dispatch'})
            self.assertEqual(workflow['on']['push'], {'branches': ['**']})
            self.assertEqual(jobs['compatibility']['strategy']['matrix'],
                             {'os': ['ubuntu-latest', 'windows-latest'],
                              'python': ['3.10', '3.11', '3.12']})
            self.assertEqual(set(jobs['validation']['needs']),
                             {'compatibility', 'slow-tests', 'release-checks'})
            self.assertEqual(jobs['validation']['if'], 'always()')
            for job in ('tag', 'deploy'):
                self.assertIn('validation', jobs[job]['needs'])
                self.assertIn('release-checks', jobs[job]['needs'])
            self.assertIn('tag', jobs['deploy']['needs'])

        def test_guards_allow_only_trusted_main_pushes(self):
            contexts = [
                ('push', 'refs/heads/main', 'mariusdgm/rl-envs-forge', False, True),
                ('pull_request', 'refs/pull/1/merge', 'mariusdgm/rl-envs-forge', False, False),
                ('pull_request', 'refs/heads/main', 'mariusdgm/rl-envs-forge', False, False),
                ('workflow_dispatch', 'refs/heads/main', 'mariusdgm/rl-envs-forge', False, False),
                ('push', 'refs/heads/feature/test', 'mariusdgm/rl-envs-forge', False, False),
                ('push', 'refs/tags/v6.0.0', 'mariusdgm/rl-envs-forge', False, False),
                ('push', 'refs/heads/main', 'fork/rl-envs-forge', False, False),
                ('push', 'refs/heads/main', 'mariusdgm/rl-envs-forge', True, False),
            ]
            for job in ('tag', 'deploy'):
                expression = jobs[job]['if'].replace('&&', ' and ').replace('!', ' not ')
                for event, ref, repo, deleted, expected in contexts:
                    with self.subTest(job=job, event=event, ref=ref, repo=repo, deleted=deleted):
                        github = SimpleNamespace(event_name=event, ref=ref, repository=repo,
                                                 event=SimpleNamespace(deleted=deleted))
                        self.assertEqual(eval(expression, {'__builtins__': {}}, {'github': github}), expected)

        def test_permissions_credentials_and_artifact_wiring(self):
            self.assertEqual(workflow['permissions'], {'contents': 'read'})
            self.assertIs(workflow['concurrency']['cancel-in-progress'], False)
            self.assertEqual(workflow['concurrency']['queue'], 'max')
            self.assertEqual([name for name, job in jobs.items()
                              if job.get('permissions', {}).get('contents') == 'write'], ['tag'])
            secret_steps = []
            for name, job in jobs.items():
                for item in job['steps']:
                    if item.get('uses', '').startswith('actions/checkout@'):
                        self.assertIs(item['with']['persist-credentials'], False)
                    if 'secrets.' in json.dumps(item):
                        secret_steps.append((name, item['name']))
            self.assertEqual(secret_steps, [('deploy', 'Publish validated distributions to PyPI')])
            publish = step('deploy', 'Publish validated distributions to PyPI')
            self.assertIn('--skip-existing', publish['run'])
            self.assertNotIn('--build', publish['run'])
            self.assertEqual(publish['env']['POETRY_HTTP_BASIC_PYPI_PASSWORD'], '${{ secrets.PYPI_PASSWORD }}')
            download = step('deploy', 'Download the distributions validated by this run')
            self.assertEqual(download['with']['artifact-ids'], '${{ needs.release-checks.outputs.artifact-id }}')
            self.assertEqual(download['with']['digest-mismatch'], 'error')

        def test_validation_gate_requires_success_from_every_job(self):
            results = {name: {'result': 'success'} for name in jobs['validation']['needs']}
            self.execute('validation', {'RESULTS': json.dumps(results)})
            for name in results:
                for status in ('failure', 'cancelled', 'skipped'):
                    changed = copy.deepcopy(results)
                    changed[name]['result'] = status
                    with self.subTest(job=name, status=status), self.assertRaises(SystemExit):
                        self.execute('validation', {'RESULTS': json.dumps(changed)})

        def tag_run(self, exists=1, target=None, environment=None):
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, exists if command[0] == 'git' else 0)

            with patch('subprocess.run', side_effect=run), patch('subprocess.check_output', return_value=target or 'a' * 40):
                self.execute('tag', environment or trusted)
            return calls

        def test_missing_tag_creates_exact_version_at_event_commit(self):
            calls = self.tag_run()
            self.assertEqual(calls[-1], ['gh', 'api', '--method', 'POST',
                                        'repos/mariusdgm/rl-envs-forge/git/refs',
                                        '-f', 'ref=refs/tags/v6.0.0', '-f', f'sha={"a" * 40}'])

        def test_existing_same_commit_tag_is_reused_without_writes(self):
            self.assertEqual(self.tag_run(exists=0),
                             [['git', 'show-ref', '--verify', '--quiet', 'refs/tags/v6.0.0']])

        def test_existing_other_commit_tag_or_git_failure_fails_closed(self):
            for exists, target in ((0, 'b' * 40), (128, None)):
                with self.subTest(exists=exists), self.assertRaises(SystemExit):
                    self.tag_run(exists=exists, target=target)

        def test_tag_script_rejects_untrusted_events_and_invalid_versions(self):
            for key, value in (('GITHUB_EVENT_NAME', 'pull_request'),
                               ('GITHUB_REF', 'refs/heads/feature'),
                               ('GITHUB_REPOSITORY', 'fork/rl-envs-forge'),
                               ('RELEASE_VERSION', '6.0.0\ninjected')):
                with self.subTest(key=key), self.assertRaises(SystemExit):
                    self.tag_run(environment=dict(trusted, **{key: value}))

        def artifact_fixture(self, directory):
            (directory / 'pyproject.toml').write_text('[tool.poetry]\nname="rl-envs-forge"\nversion="6.0.0"\n')
            dist = directory / 'dist'
            dist.mkdir()
            with zipfile.ZipFile(dist / 'rl_envs_forge-6.0.0-py3-none-any.whl', 'w') as archive:
                archive.writestr('rl_envs_forge-6.0.0.dist-info/METADATA', 'Name: rl-envs-forge\nVersion: 6.0.0\n')
            (dist / 'rl_envs_forge-6.0.0.tar.gz').write_bytes(b'unit-test-sdist')

        def test_manifest_and_publisher_checksum_verification(self):
            with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
                directory = Path(temporary)
                self.artifact_fixture(directory)
                previous = Path.cwd()
                try:
                    os.chdir(directory)
                    self.execute('manifest', dict(trusted, SOURCE_SHA='a' * 40,
                                                  GITHUB_OUTPUT=str(directory / 'outputs')))
                    self.assertEqual((directory / 'outputs').read_text(), 'version=6.0.0\n')
                    self.execute('verify', trusted)
                    for key, value in (('RELEASE_VERSION', '6.0.1'), ('GITHUB_SHA', 'b' * 40)):
                        with self.subTest(key=key), self.assertRaises(SystemExit):
                            self.execute('verify', dict(trusted, **{key: value}))
                    wheel = directory / 'dist/rl_envs_forge-6.0.0-py3-none-any.whl'
                    original = wheel.read_bytes()
                    wheel.write_bytes(b'tampered')
                    with self.assertRaises(SystemExit):
                        self.execute('verify', trusted)
                    wheel.write_bytes(original)
                    (directory / 'dist/unexpected.whl').write_bytes(b'unexpected')
                    with self.assertRaises(SystemExit):
                        self.execute('verify', trusted)
                finally:
                    os.chdir(previous)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReleaseChecks))
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
