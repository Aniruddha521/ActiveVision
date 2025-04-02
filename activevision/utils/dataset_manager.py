import os
import pandas as pd
import requests
from tqdm import tqdm
from zipfile import ZipFile

class DatasetManager:
    def __init__(self, data_dir='dataset'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _download_file(self, url, filename):
        if url.startswith("http://"):
            consent = input(f"WARNING: The URL {url} is not secure. Do you want to proceed? (y/n): ")
            if consent.strip().lower() != 'y':
                print("Download cancelled by user.")
                return None

        print(f"Downloading file from {url} ...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Failed to download file: {response.status_code}")
            return None

        total_size_bytes = int(response.headers.get('content-length', 0))
        total_size_mb = total_size_bytes / (1024 * 1024)
        file_path = os.path.join(self.data_dir, filename)
        block_size = 1024  # 1KB blocks
        
        with open(file_path, "wb") as f, tqdm(
            total=total_size_mb, unit='MB', unit_scale=False, desc=filename
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data) / (1024 * 1024))
        print(f"File downloaded and saved to {file_path}.")
        return file_path

    def download_dataset(self, url, dataset_name):
        
        ext = os.path.splitext(url)[1]
        filename = f"{dataset_name}{ext}"
        file_path = self._download_file(url, filename)
        if not file_path:
            return

        if ext.lower() == ".zip":
            print(f"Extracting {dataset_name} from zip file...")
            extraction_path = os.path.join(self.data_dir, dataset_name)
            os.makedirs(extraction_path, exist_ok=True)
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_path)
            os.remove(file_path)
            print(f"{dataset_name} downloaded and extracted to {extraction_path}.")
        else:
            print(f"{dataset_name} downloaded as a single file at {file_path}.")

    def download_fixation(self, url, dataset_name, output_format='csv'):
        
        if output_format not in ('csv', 'json'):
            raise AssertionError(f"Expected output_format to be 'csv' or 'json' but got {output_format}")

        ext = os.path.splitext(url)[1]
        filename = f"{dataset_name}_fixation{ext}"
        file_path = self._download_file(url, filename)
        if not file_path:
            return

        extraction_path = os.path.join(self.data_dir, dataset_name)
        os.makedirs(extraction_path, exist_ok=True)
        if ext.lower() == ".zip":
            print(f"Extracting fixation data for {dataset_name} from zip file...")
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_path)
            os.remove(file_path)
            print(f"Fixation data for {dataset_name} extracted to {extraction_path}.")
            if os.path.isdir(extraction_path):
                files = [os.path.join(extraction_path, f) for f in os.listdir(extraction_path)
                        if f.endswith('.csv') or f.endswith('.json')]
            else:
                files = [extraction_path] if extraction_path.endswith(('.csv', '.json')) else []

            if not files:
                print("No csv or json files found in the fixation data to merge.")
                return 

            dataframes = []
            for file in files:
                try:
                    if file.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_json(file)
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        else:
            dest_path = os.path.join(extraction_path, os.path.basename(file_path))
            os.rename(file_path, dest_path)
            dataframes = []
            if dest_path.endswith('.csv'):
                df = pd.read_csv(dest_path)
            else:
                df = pd.read_json(dest_path)
            dataframes.append(df)
            os.remove(dest_path)

        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            merged_path = os.path.join(self.data_dir, dataset_name, f"{dataset_name}_fixation_merged.{output_format}")
            if output_format.lower() == 'csv':
                merged_df.to_csv(merged_path, index=False)
            else:
                merged_df.to_json(merged_path, orient="records", lines=True)
            print(f"Merged {output_format} saved to {merged_path}.")

    def load_csv(self, filepath):
        try:
            data = pd.read_csv(filepath)
            print(f"Loaded data from {filepath} with shape {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def load_json(self, filepath):
        try:
            data = pd.read_json(filepath)
            print(f"Loaded data from {filepath} with shape {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None
