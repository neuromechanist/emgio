# Git & Version Control Standards

## Commit Messages
- **Format:** `<type>: <description>`
- **Length:** <50 characters
- **No emojis** in commits or PR titles
- **Types:**
  - `feat:` New feature
  - `fix:` Bug fix
  - `docs:` Documentation only
  - `refactor:` Code restructuring
  - `test:` Adding tests (real tests only)
  - `chore:` Maintenance tasks

## Branch Strategy
- **Feature branches:** `feature/short-description`
- **Bugfix branches:** `fix/issue-description`
- **No spaces** in branch names, use hyphens
- **Delete after merge**

## Commit Practice
- **Atomic commits** - One logical change per commit
- **Test before commit** - Ensure code works
- **No broken commits** - Each commit should work independently

## Pull Request Process
1. Create issue first (for significant changes)
2. Branch from main (`gh issue develop <n>` when permitted, else `git checkout -b feature/...`)
3. Make atomic commits
4. Push branch
5. Create PR with:
   - Clear title (no issue numbers)
   - Description with "Fixes #123"
   - Test results
   - Screenshots if UI changes

## Git Commands
```bash
# Start feature
git checkout -b feature/new-thing

# Atomic commits
git add -p  # Stage selectively
git commit -m "feat: add user authentication"

# Update branch
git fetch origin
git rebase origin/main

# Push and create PR
git push -u origin feature/new-thing
gh pr create
```

## .gitignore Essentials
```
__pycache__/     # Python
node_modules/    # JavaScript
.env            # Secrets
*.log           # Logs
```
Note: `AGENTS.md`, `CLAUDE.md`, `.rules/`, and `.context/` are tracked in git, not ignored.

---
*Atomic commits, clear messages, clean history.*