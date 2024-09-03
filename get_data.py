import os
import subprocess
import shutil
import stat


def remove_readonly(func, path, excinfo):
    # Change the file's permissions and retry
    os.chmod(path, stat.S_IWRITE)
    func(path)
    
    
# Folders and URLs
extract_folder = '_tmp_data'
repository = 'aladdin_cmr_la_data'
repo_url = 'https://github.com/cgalaz01/aladdin_cmr_la_data.git'

expected_folders = {
    'data': os.path.join(extract_folder, repository, 'data'),
    'data_nn': os.path.join(extract_folder, repository, 'data_nn'),
    'atlas_output': os.path.join(extract_folder, repository, '_atlas_output'),
    'atlas_stats_output': os.path.join(extract_folder, repository, '_atlas_stats_output')
}
target_folders = {
    'data': '.',
    'data_nn': '.',
    'atlas_output': os.path.join('src', 'atlas_construction'),
    'atlas_stats_output': os.path.join('src', 'atlas_construction')
}

license_path = os.path.join(extract_folder, repository, 'LICENSE')

# Remove
if os.path.exists(extract_folder):
    shutil.rmtree(extract_folder, onerror=remove_readonly)
    print(f'Removed existing directory: {extract_folder}')
os.makedirs(extract_folder)

# Clone the GitHub repository
print(f'Cloning {repository} into {extract_folder}. Please wait patiently...')
clone_command = [
    'git', 'clone', '--depth', '1', '--single-branch', 
    repo_url, os.path.join(extract_folder, repository)
]
subprocess.run(clone_command, check=True)
print(f'Cloned {repository} into {extract_folder}.')

# Move the folders
for key, source_path in expected_folders.items():
    target_path = target_folders[key]
    target_folder = os.path.join(target_path, os.path.basename(source_path))
    #os.makedirs(target_path, exist_ok=True)
    
    if os.path.exists(target_folder):
        print(f'Target folder \'{target_folder}\' already exists. Skipped moving new data.')
    elif os.path.exists(source_path):
        shutil.move(source_path, target_path)
        shutil.copy(license_path, target_folder)
        print(f'Moved folder {source_path} to {target_path}')
    else:
        print(f'Source folder not found: {source_path}')
        
# Remove temporary folder
shutil.rmtree(extract_folder, onerror=remove_readonly)
print(f'Removed cloned data repository: {extract_folder}')
