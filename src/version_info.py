import os
import ctypes
from datetime import datetime
import subprocess

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


def update_to_latest(local_repo_path=None, remote_name='origin', branch_name='main'):
    if local_repo_path is None:
        local_repo_path = os.path.dirname(os.getcwd())

    try:
        # if not admin user
        if ctypes.windll.shell32.IsUserAnAdmin() != 1:
            print("Admin permission is required for update. Update skipped...")
            return
    except AttributeError:
        print('Cannot determine the status of user. Attempt to update...')

    try:
        token = "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"
        username = "lxfhfut"
        repository = "Cabana"
        git_url = f"https://{token}@github.com/{username}/{repository}.git"
        subprocess.run(['git', 'remote', 'set-url', remote_name, git_url], check=True)
        subprocess.run(['git', 'fetch', remote_name, branch_name, '--tags'], cwd=local_repo_path, check=True)
        behind_cnt = int(subprocess.check_output(['git', 'rev-list', '--count', 'HEAD..@{u}'],
                                                 cwd=local_repo_path).strip().decode('utf-8'))
        ahead_cnt = int(subprocess.check_output(['git', 'rev-list', '--count', '@{u}..HEAD'],
                                                cwd=local_repo_path).strip().decode('utf-8'))
        if behind_cnt == ahead_cnt and ahead_cnt == 0:
            print("Local repository is up-to-date.")
        elif behind_cnt > 0:
            print(f"Local repository is {behind_cnt} commits behind remote.")
            print("Updating to the latest version...")
            subprocess.run(['git', 'pull'], cwd=local_repo_path, check=True)
            print("Remote changes and tags fetched successfully.")
        else:
            pass
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"Something went wrong while updating. Update skipped...")
        # print(f"Error fetching remote changes and tags: {e.stderr.decode('utf-8').strip()}")


def export_version_info(save_txt_file, fiji_install_dir=r'C:\Program Files\fiji-win64\Fiji.app'):
    try:
        tag = subprocess.check_output(['git', 'describe', '--tags', '--abbrev=0'],
                                      stderr=subprocess.PIPE).strip().decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error getting Git tag: {e.stderr.decode('utf-8').strip()}")
        return

    plugin_dir = os.path.join(fiji_install_dir, "plugins")
    jar_names = [f.name[:-4] for f in os.scandir(plugin_dir) if f.is_file() and f.name.endswith('.jar')]

    try:
        with open(save_txt_file, 'w+') as tf:
            tf.write('Program started at ' + datetime.now().strftime("%H:%M:%S %Y-%m-%d") + "\n\n")

            tf.write(f"************Versions*************\n")
            tf.write(f"Cabana: {tag}\n")
            for jar_name in jar_names:
                substrings = jar_name.split('-')
                split_idx = 2
                for idx, substr in enumerate(substrings):
                    if substr.replace(".", "").isdigit():
                        split_idx = idx
                        break

                name, version = "-".join(substrings[:split_idx]), "-".join(substrings[split_idx:])
                if name.lower() in ["ij_ridge_detect", "anamorf", 'bio-formats_plugins']:
                    tf.write(f"\t{name}: {version}\n")
        print(f"Version information saved to {save_txt_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")
        return


if __name__ == "__main__":
    update_to_latest()
