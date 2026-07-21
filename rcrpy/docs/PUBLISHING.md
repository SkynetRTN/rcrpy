# Publishing `rcrpy`

The CI workflow builds distributions automatically on every push; 
this file covers the **manual steps** for actually uploading to TestPyPI / PyPI.

## One-time setup

1. **Create accounts** on both:
   - [TestPyPI](https://test.pypi.org/account/register/) — for trial releases
   - [PyPI](https://pypi.org/account/register/) — for real releases
2. **Enable 2FA** on both accounts (PyPI requires this for new uploads).
3. **Create API tokens** — one per account. Save them somewhere safe.
   - **Tip:** You can set the `UV_PUBLISH_TOKEN` environment variable (e.g., in your `.zshrc`) to avoid passing the `--token` flag every time. `uv` will use this token automatically for uploads.

## Per-release checklist

From the **rcrpy** directory:

```bash
# 1. Bump the version in TWO places.
#    - pyproject.toml:  [project] version = "x.y.z"
#    - src/rcrpy/__init__.py:  __version__ = "x.y.z"
#    (These must match.)

# 2. Run the full test suite.
uv run pytest

# 3. Build the wheel + sdist.
uv build
ls dist/   # should show rcrpy-x.y.z-py3-none-any.whl and rcrpy-x.y.z.tar.gz

# 4. Install the BUILT wheel in a CLEAN venv and smoke-test it.
uv venv /tmp/rcrpy-clean
uv pip install dist/rcrpy-x.y.z-py3-none-any.whl --python /tmp/rcrpy-clean
/tmp/rcrpy-clean/bin/python scripts/smoke_test_full_rcr.py
# (On Windows: c:\path\rcrpy-clean\Scripts\python.exe — adjust accordingly.)

# 5. Upload to TestPyPI FIRST.
# If UV_PUBLISH_TOKEN is set, you can omit the --token flag.
uv publish --publish-url https://test.pypi.org/legacy/ --token "pypi-your-testpypi-token" dist/*

# 6. Install FROM TestPyPI in another clean venv to confirm the upload.
uv venv /tmp/rcrpy-from-testpypi
uv pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    rcrpy==x.y.z \
    --python /tmp/rcrpy-from-testpypi
# scipy and numpy come from real PyPI (--extra-index-url), rcrpy from
# TestPyPI (--index-url).

# 7. If TestPyPI looked good: upload to real PyPI.
# If UV_PUBLISH_TOKEN is set, you can omit the --token flag.
uv publish --token "pypi-your-real-pypi-token" dist/*

# 8. Tag the release in git.
git tag -a vx.y.z -m "rcrpy vx.y.z"
git push origin vx.y.z

# 9. Create a GitHub Release attached to that tag.
```

## Documentation (Sphinx + Read the Docs)

The docs live in `rcrpy/docs/` (Sphinx + MyST Markdown, `furo` theme) and are
built and hosted automatically by [Read the Docs](https://rcrpy.readthedocs.io/).
The build config is `.readthedocs.yaml` at the repository root; the Sphinx
config is `rcrpy/docs/conf.py`. The `docs` extra in `pyproject.toml` pins the
build dependencies (`sphinx`, `furo`, `myst-parser`).

### Build & preview locally

From the **rcrpy** directory:

```bash
# Build the HTML site (installs the docs extra into the env on first run).
uv run --extra docs sphinx-build -b html docs docs/_build/html

# Open it.
open docs/_build/html/index.html        # macOS
# xdg-open docs/_build/html/index.html   # Linux

# Stricter pre-release check: fail on any warning (use once warnings are clean).
uv run --extra docs sphinx-build -b html -W --keep-going docs docs/_build/html
```

(From the repo root instead, add `--project rcrpy` and use the full paths:
`uv run --project rcrpy --extra docs sphinx-build -b html rcrpy/docs rcrpy/docs/_build/html`.)

The `docs/_build/` and `docs/generated/` output directories are git-ignored.

### How it deploys

You don't upload docs by hand — Read the Docs rebuilds them on its own:

- **Every push to `main`** rebuilds the `latest` docs (via the RTD webhook).
- **Every release tag** (`vX.Y.Z`, step 8 above) builds a versioned copy, as
  long as the "build on new tag" Automation Rule is enabled in the RTD admin.

So once you push the release tag, the matching docs version publishes
automatically. Watch the **Builds** tab of the
[Read the Docs project](https://app.readthedocs.org/) to confirm it succeeded.

### Adding a page

Drop a new `.md` (or `.rst`) file in `rcrpy/docs/` and add its name (without the
extension) to a `toctree` in `rcrpy/docs/index.md`. The API reference in
`rcrpy/docs/api.md` is generated from the package's public docstrings, so new
public classes/functions show up there automatically.

## Version-bump conventions

`rcrpy` follows [semver](https://semver.org):

| Bump | When |
|---|---|
| 0.1.0 → 0.1.1 | Bug fix, no API change |
| 0.1.0 → 0.2.0 | New feature OR breaking API change (allowed in 0.x) |
| 0.x → 1.0.0 | "Stable API; we're committing to backwards compat" |

We're at **0.1.0 — early beta**. Stay in 0.x until 1-2 real users have
exercised the package and reported back without surfacing critical
issues.

## If something goes wrong

- **Upload rejected with "already exists"**: PyPI / TestPyPI does not
  allow re-uploading the same version. Bump the version and try again.
- **Test install fails with "no matching distribution"**: TestPyPI
  doesn't have scipy/numpy. Use `--extra-index-url https://pypi.org/simple/`
  to pull dependencies from real PyPI.
- **Wheel is missing a file**: check `MANIFEST.in` and the
  `[tool.hatch.build.targets.wheel]` section of `pyproject.toml`. By default
  `hatchling` includes files in `src/rcrpy/`, the LICENSE, and
  the README.
