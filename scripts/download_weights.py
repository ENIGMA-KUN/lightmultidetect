import os
import argparse
import requests
from tqdm import tqdm

# Model weights URLs (replace with actual URLs for your models)
WEIGHTS = {
    "xception_deepfake.pth": "https://example.com/weights/xception_deepfake.pth",
    "efficientnet_deepfake.pth": "https://example.com/weights/efficientnet_deepfake.pth",
    "mesonet_deepfake.pth": "https://example.com/weights/mesonet_deepfake.pth",
    "wav2vec2_deepfake.pth": "https://example.com/weights/wav2vec2_deepfake.pth",
    "rawnet2_deepfake.pth": "https://example.com/weights/rawnet2_deepfake.pth",
    "melspec_deepfake.pth": "https://example.com/weights/melspec_deepfake.pth",
    "c3d_deepfake.pth": "https://example.com/weights/c3d_deepfake.pth",
    "two_stream_deepfake.pth": "https://example.com/weights/two_stream_deepfake.pth",
    "timesformer_deepfake.pth": "https://example.com/weights/timesformer_deepfake.pth",
}


def download_file(url, destination):
    """
    Download a file with progress bar
    """
    if os.path.exists(destination):
        print(f"File {destination} already exists, skipping.")
        return
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    progress_bar = tqdm(
        total=total_size, 
        unit='iB', 
        unit_scale=True, 
        desc=f"Downloading {os.path.basename(destination)}"
    )
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()


def main(weights_dir):
    """
    Download model weights to specified directory
    """
    os.makedirs(weights_dir, exist_ok=True)
    
    print(f"Downloading model weights to {weights_dir}...")
    
    for filename, url in WEIGHTS.items():
        destination = os.path.join(weights_dir, filename)
        try:
            download_file(url, destination)
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
    
    print("\nDownload complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--weights-dir", type=str, default="backend/app/models/weights", 
                         help="Directory to save model weights")
    args = parser.parse_args()
    
    main(args.weights_dir)