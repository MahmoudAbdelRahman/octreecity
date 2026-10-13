# Contributing to OctreeCity

Thanks for your interest! Bug reports, ideas, documentation fixes and code are
all welcome.

## Questions and ideas

Open an [issue](https://github.com/MahmoudAbdelRahman/octreecity/issues):
choose "Bug report" or "Feature request". For a bug, a few lines of code that
show the problem help a lot.

## Making a change

1. Fork the repository (or, for collaborators, clone it) and set up:

   ```bash
   git clone https://github.com/MahmoudAbdelRahman/octreecity
   cd octreecity
   uv sync
   ```

2. Create a branch for one change, with a short descriptive name:

   ```bash
   git switch -c feature/dem-reader
   ```

   Prefixes: `feature/` new functionality, `fix/` bug fixes, `docs/`
   documentation, `refactor/` restructuring without new behaviour.

3. Make the change, with a test for new behaviour or a fixed bug, and run the tests:

   ```bash
   uv run pytest -q
   ```

4. If you changed the docs, check that they build without broken links:

   ```bash
   uv run --group docs mkdocs build --strict
   ```

5. Commit, push the branch and open a pull request against `main`:

   ```bash
   git push -u origin feature/dem-reader
   gh pr create --fill
   ```

The tests run automatically on Python 3.10 and 3.13. A pull request is merged
once they pass and it has been reviewed. `main` is protected: all changes,
including the maintainer's, go through pull requests.

## Style

- Plain, readable Python that matches the surrounding code; numpy for heavy work.
- Lengths in metres, coordinates in a projected CRS (see the docs).
- Public functions get a short docstring saying what they take and return.
- Keep the package light: heavy or optional dependencies go into an extra
  (like `[viz]`), not the core.
- Generated data (cities, meshes, frames, videos) stays out of git.

## Releases

Releases are made from `main` by the maintainer; the steps are in
[RELEASING.md](RELEASING.md).

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
