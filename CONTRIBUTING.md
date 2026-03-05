# Contributing to MMPP

Thanks for contributing to MMPP.

## Before opening a PR

1. Create a feature branch.
2. Keep changes scoped and reviewable.
3. Add or update tests for behavior changes.
4. Update docs when API or workflow changes.

## Local quality checks

Run before submitting:

```bash
python -m ruff format mmpp/ tests/ scripts/
python -m ruff check mmpp/ tests/ scripts/
python -m mypy mmpp/
python -m pytest tests/ -q
```

If you use `just`:

```bash
just format
just lint
just test
```

## Coding guidelines

- Prefer typed APIs and explicit function signatures.
- Keep modules cohesive; split large files with mixed responsibilities.
- Reuse existing helpers/services instead of duplicating logic.
- Avoid introducing hidden side effects in data processing paths.

## Security and safety expectations

- Validate user-controlled inputs.
- Avoid unsafe shell invocation patterns.
- Do not log credentials, tokens, or sensitive file paths.
- For changes affecting file I/O, auth, networking, or execution, note risk + mitigation in the PR description.

## Pull request checklist

- [ ] Tests added/updated for new behavior
- [ ] Lint/type/test checks pass locally
- [ ] Backward compatibility considered
- [ ] Documentation updated if needed
- [ ] Security implications reviewed for sensitive changes

## Reporting issues

Please include:
- Python version
- MMPP version
- OS
- Minimal reproducible example
- Error trace/log excerpt

## Release notes

Maintainers should keep `RELEASE_NOTES.md` updated for user-visible changes.
