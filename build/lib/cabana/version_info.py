import git

def get_version_info():
    """Extract version information from git repository."""
    try:
        # Get git repository
        repo = git.Repo(search_parent_directories=True)
        # Get current commit hash
        commit_hash = repo.head.commit.hexsha[:7]  # Short hash
        # Get current branch name
        branch = repo.active_branch.name
        # Get latest tag if available
        tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
        latest_tag = tags[-1].name if tags else "No tags"

        return {
            "git_hash": commit_hash,
            "git_branch": branch,
            "latest_tag": latest_tag,
            "version": latest_tag
        }
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return {"version": "unknown", "git_info": "Not a git repository"}
