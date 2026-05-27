# Publishing `rcr2`

This is a checklist for the maintainer (you). The CI workflow builds
distributions automatically on every push; this file covers the **manual
steps** for actually uploading to TestPyPI / PyPI.

## One-time setup

1. **Create accounts** on both:
   - [TestPyPI](https://test.pypi.org/account/register/) — for trial releases
   - [PyPI](https://pypi.org/account/register/) — for real releases
2. **Enable 2FA** on both accounts (PyPI requires this for new uploads).
3. **Create API tokens** — one per account. Save them somewhere safe;
   you'll paste them when prompted by `twine`.

Optionally, create a `~/.pypirc` once so `twine` knows where to look:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
  username = __token__
  password = pypi-AgEI...your-real-PyPI-token

[testpypi]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-AgEI...your-TestPyPI-token
```

## Per-release checklist

From the **repo root**:

```bash
# 1. Bump the version in TWO places.
#    - python/pyproject.toml:  [project] version = "x.y.z"
#    - python/src/rcr2/__init__.py:  __version__ = "x.y.z"
#    (These must match.)

# 2. Run the full test suite.
cd python
python -m pytest -q
cd ..

# 3. Build the wheel + sdist.
python -m build ./python
ls python/dist/   # should show rcr2-x.y.z-py3-none-any.whl and rcr2-x.y.z.tar.gz

# 4. Install the BUILT wheel in a CLEAN venv and smoke-test it.
python -m venv /tmp/rcr2-clean
/tmp/rcr2-clean/bin/python -m pip install python/dist/rcr2-x.y.z-py3-none-any.whl
/tmp/rcr2-clean/bin/python python/smoke_install.py
# (On Windows: c:\path\rcr2-clean\Scripts\python.exe — adjust accordingly.)

# 5. Install twine if you haven't.
python -m pip install --upgrade twine

# 6. Upload to TestPyPI FIRST.
python -m twine upload --repository testpypi python/dist/rcr2-x.y.z*

# 7. Install FROM TestPyPI in another clean venv to confirm the upload.
python -m venv /tmp/rcr2-from-testpypi
/tmp/rcr2-from-testpypi/bin/python -m pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    rcr2==x.y.z
# scipy and numpy come from real PyPI (--extra-index-url), rcr2 from
# TestPyPI (--index-url).

# 8. If TestPyPI looked good: upload to real PyPI.
python -m twine upload python/dist/rcr2-x.y.z*

# 9. Tag the release in git.
git tag -a vx.y.z -m "rcr2 vx.y.z"
git push origin vx.y.z

# 10. Create a GitHub Release attached to that tag.
```

## Version-bump conventions

`rcr2` follows [semver](https://semver.org):

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
  `[tool.setuptools]` sections of `pyproject.toml`. By default
  setuptools includes everything under `src/rcr2/`, the LICENSE, and
  the README; if you add files outside that path they may need an
  explicit include.
