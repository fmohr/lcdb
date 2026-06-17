import sys
import os
import inspect

# Detect the script directory
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# List of potential paths to look for the 'lcdb' package
potential_roots = [
    # 1. 3 levels up (if running from /home/cyan/lcdb/publications/2023-neurips/experiments/surf/snellius)
    os.path.abspath(os.path.join(current_dir, "../../..")),
    # 2. 2 levels up (if running from /home/cyan/lcdb/publications/2023-neurips/experiments/surf)
    os.path.abspath(os.path.join(current_dir, "../..")),
    # 3. Local package reference in standard dev workspace
    os.path.abspath(os.path.join(current_dir, "lcdb/publications/2023-neurips")),
    # 4. Current directory
    current_dir
]

# Find the first path that contains the 'lcdb' package folder
project_root = None
for root in potential_roots:
    if os.path.isdir(os.path.join(root, "lcdb")):
        project_root = root
        break

if project_root:
    sys.path.insert(0, project_root)
    print(f"📂 Added project root to sys.path: {project_root}")
else:
    sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../../..")))

try:
    import lcdb.db._pcloud_repository as repo_mod
    print("✅ Successfully imported PCloudRepository!")
    print(f"📍 Imported from file: {repo_mod.__file__}")
    print(f"🔍 authenticate signature: {inspect.signature(repo_mod.PCloudRepository.authenticate)}")
    PCloudRepository = repo_mod.PCloudRepository
except ImportError as e:
    print(f"❌ Error importing PCloudRepository: {e}")
    sys.exit(1)

# Initialize the repository (repo_code=None for private app workspace access)
print("Initializing repository...")
repo = PCloudRepository(repo_code=None)

# Authenticate
print("Starting OAuth authentication flow...")
try:
    repo.authenticate()
    print("\n🎉 Authentication successful!")
    print(f"Cached Token: {repo.token}")
except Exception as e:
    print(f"\n❌ Authentication failed: {e}")
