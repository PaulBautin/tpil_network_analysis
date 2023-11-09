import os
import shutil

source_folder = '/mnt/d/23-07-05_DTI_metrics'   # Replace this with the actual path to your source folder
destination_folder = '/mnt/d/Marc-Antoine/TBSS' # Replace this with the actual path to your destination folder

print(os.listdir(source_folder))

def extract_files_from_folder(folder_path):
    root = folder_path
    for dir in os.listdir(folder_path):
        for filename in os.listdir(os.path.join(root, dir, 'DTI_metrics')):
            if filename.endswith('md.nii.gz'):
                source_file = os.path.join(root, dir, 'DTI_metrics', filename)
                
                # Replacing 'md' with 'fa' in the destination filename
                destination_filename = filename.replace('md', 'fa')

                # Extracting v1, v2, or v3 from the filename
                version = None
                if 'v1' in filename:
                    version = 'v1'
                elif 'v2' in filename:
                    version = 'v2'
                elif 'v3' in filename:
                    version = 'v3'
                
                # Creating a subfolder based on the version if it doesn't exist
                if version:
                    version_folder = os.path.join(destination_folder, version)
                    os.makedirs(version_folder, exist_ok=True)
                    
                    destination_file = os.path.join(version_folder, f"clbp_{destination_filename}")
                    print(f"Copying file: {source_file} to {destination_file}")
                    shutil.copy(source_file, destination_file)

if __name__ == '__main__':
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Call the function to extract files from the 'clbp' and 'control' folders
    extract_files_from_folder(os.path.join(source_folder, 'clbp'))
    #extract_files_from_folder(os.path.join(source_folder, 'control'))