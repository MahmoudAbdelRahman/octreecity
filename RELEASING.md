# Releasing to PyPI

Release only from an up-to-date `main`, after the changes are merged:
`git switch main && git pull`.

1. Bump `version` in `pyproject.toml` (for example 0.1.0 -> 0.1.1), through a
   small pull request like any other change.
2. Build and check:

   ```bash
   rm -rf dist && uv build && uvx twine check dist/*
   ```

3. Practice upload to TestPyPI (account at test.pypi.org, create an API token):

   ```bash
   uv publish --publish-url https://test.pypi.org/legacy/ --token <testpypi-token>
   ```

4. Try it like a user, in an empty environment:

   ```bash
   uv run --isolated --with "octreecity[viz]" --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match python -c "import octreecity; print('ok')"
   ```

5. Real upload (account at pypi.org, separate token):

   ```bash
   uv publish --token <pypi-token>
   ```

6. Tag the release: `git tag v0.1.0 && git push --tags`.

Notes: a version number can be uploaded only once. Images in the README load
on PyPI only when the GitHub repository is public.
