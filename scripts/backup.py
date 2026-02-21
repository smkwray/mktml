import os
import shutil
import datetime
from pathlib import Path

def create_backup():
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    today_str = datetime.datetime.now().strftime("%d_%m_%y")
    backup_name = f"backup_{today_str}"
    backup_path = root_dir / backup_name
    
    # 1. Check if today's backup already exists
    if backup_path.exists():
        print(f"Backup for today already exists: {backup_path}")
        return

    # 2. Manage Old Backups
    # Find all existing backup folders in the root
    existing_backups = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("backup_")]
    
    if existing_backups:
        print(f"Found {len(existing_backups)} old backups. Moving them to /old/...")
        old_dir = root_dir / "old"
        old_dir.mkdir(exist_ok=True)
        
        for old_bkp in existing_backups:
            try:
                # Move to old directory
                # Note: shutil.move might fail if destination exists, so we handle that if needed,
                # but for now we assume unique names based on date
                dest = old_dir / old_bkp.name
                if dest.exists():
                     print(f"  Warning: {dest} already exists in /old/, skipping move of {old_bkp.name}")
                else:
                    shutil.move(str(old_bkp), str(old_dir))
                    print(f"  Moved {old_bkp.name} to old/")
            except Exception as e:
                print(f"  Error moving {old_bkp.name}: {e}")

    # 3. Create New Backup
    print(f"Creating new backup: {backup_name}...")
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Define what to copy
    # Includes: *.py, *.md, src/, scripts/, config.py
    # Excludes: .git, __pycache__, venv, data, market_data.duckdb, .* (hidden files except specific ones if needed)
    
    files_to_copy = []
    dirs_to_copy = ["src", "scripts"]
    
    # Add root level files
    for item in root_dir.iterdir():
        if item.is_file():
            if item.suffix in ['.py', '.md', '.txt', '.toml', '.lock'] or item.name == 'config.py':
                files_to_copy.append(item)
    
    # Copy Directories
    for dir_name in dirs_to_copy:
        src_dir = root_dir / dir_name
        if src_dir.exists():
            dest_dir = backup_path / dir_name
            shutil.copytree(src_dir, dest_dir, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            print(f"  Copied directory: {dir_name}")

    # Copy Files
    for file_path in files_to_copy:
        shutil.copy2(file_path, backup_path / file_path.name)
        print(f"  Copied file: {file_path.name}")

    print("Backup complete.")

if __name__ == "__main__":
    create_backup()
