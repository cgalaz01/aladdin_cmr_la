import os
import shutil
import stat
import requests
import zipfile


def remove_readonly(func, path, excinfo):
    # Change the file's permissions and retry
    os.chmod(path, stat.S_IWRITE)
    func(path)


# Folders and URLs
extract_folder = '_tmp_data'
repository = 'aladdin_cmr_la_data'
repo_url = 'https://zenodo.org/records/13645121/files/aladdin_cmr_la_data.zip?download=1'

expected_folders = {
    'data': os.path.join(extract_folder, 'data'),
    'data_nn': os.path.join(extract_folder, 'data_nn'),
    'atlas_output': os.path.join(extract_folder, '_atlas_output'),
    'atlas_stats_output': os.path.join(extract_folder, '_atlas_stats_output')
}
target_folders = {
    'data': '.',
    'data_nn': '.',
    'atlas_output': os.path.join('src', 'atlas_construction'),
    'atlas_stats_output': os.path.join('src', 'atlas_construction')
}

license_path = os.path.join(extract_folder, 'LICENSE')

# Remove the existing folder if it exists
if os.path.exists(extract_folder):
    shutil.rmtree(extract_folder, onerror=remove_readonly)
    print(f'Removed existing directory: {extract_folder}')
os.makedirs(extract_folder)

# Download the repository zip file
print(f'Downloading {repository} as a zip file. Please wait patiently...')
zip_path = os.path.join(extract_folder, f'{repository}.zip')
with requests.get(repo_url, stream=True) as r:
    r.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print(f'Downloaded {repository} zip file.')

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
print(f'Extracted {repository} zip file.')

# Adjust the folder name if necessary (depends on the structure inside the zip)
extracted_folder_name = f'{repository}-main'  # Assuming 'main' branch
extracted_path = os.path.join(extract_folder, extracted_folder_name)

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
