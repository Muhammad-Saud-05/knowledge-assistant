---
date: 2026-10-02
---

# Git Workflow Cheatsheet

A short reference for everyday Git use, written for this sample corpus.

## Daily loop

1. Check what changed with `git status`.
2. Review the exact changes with `git diff`.
3. Stage only what belongs together with `git add <file>`.
4. Commit with a message that explains the change.

## Writing good commit messages

The subject line should be in the imperative mood, under about 50 characters, and describe what the commit does, such as "Add login rate limiter". If the change needs explaining, leave a blank line and add a short body that says what changed and why. Avoid vague messages like "update" or "fix stuff", because nobody can learn anything from them later.

## Branching

Create a branch for each piece of work so that `main` always stays in a working state.

```bash
git switch -c feature/search-filters
# ...work and commit...
git switch main
git merge feature/search-filters
```

### Fixing mistakes

- **Undo the last commit but keep the changes:** `git reset --soft HEAD~1`
- **Fix the last commit message before pushing:** `git commit --amend`
- **Discard changes to one file:** `git restore <file>`

Rewriting history is safe only for commits that have not been pushed. Once others may have pulled them, prefer `git revert`, which adds a new commit that undoes an old one.

## Keeping secrets out

Never commit passwords, API keys, or `.env` files. Add them to `.gitignore` before the first commit, and commit an `.env.example` with placeholder values instead. If a secret is ever pushed by accident, treat it as leaked and rotate it, because deleting the file does not remove it from history.
