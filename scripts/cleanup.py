#!/usr/bin/env python3
"""
Cleanup script for the DeepFake Detection Platform.

This script helps clean up temporary files, remove unnecessary artifacts,
and organize the project structure. It can be run periodically to maintain
a clean workspace.
"""

import os
import shutil
import argparse
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_UPLOADS_DIR = "uploads"
DEFAULT_VISUALIZATIONS_DIR = "visualizations"
DEFAULT_TEMP_DIR = "tmp"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_MAX_AGE_DAYS = 7  # Default max age for files (7 days)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cleanup script for DeepFake Detection Platform')
    
    parser.add_argument('--uploads', type=str, default=DEFAULT_UPLOADS_DIR,
                        help=f'Path to uploads directory (default: {DEFAULT_UPLOADS_DIR})')
    
    parser.add_argument('--visualizations', type=str, default=DEFAULT_VISUALIZATIONS_DIR,
                        help=f'Path to visualizations directory (default: {DEFAULT_VISUALIZATIONS_DIR})')
    
    parser.add_argument('--temp', type=str, default=DEFAULT_TEMP_DIR,
                        help=f'Path to temporary directory (default: {DEFAULT_TEMP_DIR})')
    
    parser.add_argument('--logs', type=str, default=DEFAULT_LOGS_DIR,
                        help=f'Path to logs directory (default: {DEFAULT_LOGS_DIR})')
    
    parser.add_argument('--max-age', type=int, default=DEFAULT_MAX_AGE_DAYS,
                        help=f'Maximum age of files in days (default: {DEFAULT_MAX_AGE_DAYS})')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform a dry run without actually deleting files')
    
    parser.add_argument('--force', action='store_true',
                        help='Force deletion without confirmation')
    
    return parser.parse_args()


def get_file_age_days(file_path):
    """Get the age of a file in days."""
    file_time = os.path.getmtime(file_path)
    file_datetime = datetime.fromtimestamp(file_time)
    age = datetime.now() - file_datetime
    return age.days


def clean_directory(directory, max_age_days, dry_run=False):
    """
    Clean a directory by removing files older than max_age_days.
    
    Args:
        directory (str): Directory path to clean
        max_age_days (int): Maximum age of files in days
        dry_run (bool): If True, don't actually delete files
        
    Returns:
        tuple: (number of files removed, bytes freed)
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return 0, 0
    
    files_removed = 0
    bytes_freed = 0
    
    logger.info(f"Scanning directory: {directory}")
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                age_days = get_file_age_days(file_path)
                
                if age_days > max_age_days:
                    size = os.path.getsize(file_path)
                    if dry_run:
                        logger.info(f"Would remove: {file_path} (Age: {age_days} days, Size: {size / 1024:.2f} KB)")
                    else:
                        os.remove(file_path)
                        logger.info(f"Removed: {file_path} (Age: {age_days} days, Size: {size / 1024:.2f} KB)")
                    
                    files_removed += 1
                    bytes_freed += size
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Remove empty directories
    if not dry_run:
        for root, dirs, files in os.walk(directory, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):  # Check if directory is empty
                    os.rmdir(dir_path)
                    logger.info(f"Removed empty directory: {dir_path}")
    
    return files_removed, bytes_freed


def main():
    """Main function to run cleanup."""
    args = parse_arguments()
    
    total_files_removed = 0
    total_bytes_freed = 0
    
    logger.info("Starting cleanup process...")
    logger.info(f"{'DRY RUN - ' if args.dry_run else ''}Removing files older than {args.max_age} days")
    
    # Ask for confirmation if not dry run and not forced
    if not args.dry_run and not args.force:
        confirm = input(f"This will permanently delete files older than {args.max_age} days. Continue? (y/N): ")
        if confirm.lower() != 'y':
            logger.info("Cleanup aborted by user.")
            return
    
    # Clean uploads directory
    logger.info(f"\nCleaning uploads directory: {args.uploads}")
    files, bytes_ = clean_directory(args.uploads, args.max_age, args.dry_run)
    total_files_removed += files
    total_bytes_freed += bytes_
    
    # Clean visualizations directory
    logger.info(f"\nCleaning visualizations directory: {args.visualizations}")
    files, bytes_ = clean_directory(args.visualizations, args.max_age, args.dry_run)
    total_files_removed += files
    total_bytes_freed += bytes_
    
    # Clean temporary directory
    logger.info(f"\nCleaning temporary directory: {args.temp}")
    files, bytes_ = clean_directory(args.temp, args.max_age, args.dry_run)
    total_files_removed += files
    total_bytes_freed += bytes_
    
    # Clean logs directory
    logger.info(f"\nCleaning logs directory: {args.logs}")
    files, bytes_ = clean_directory(args.logs, args.max_age, args.dry_run)
    total_files_removed += files
    total_bytes_freed += bytes_
    
    # Print summary
    mb_freed = total_bytes_freed / (1024 * 1024)
    logger.info("\nCleanup Summary:")
    logger.info(f"{'Would remove' if args.dry_run else 'Removed'} {total_files_removed} files")
    logger.info(f"{'Would free' if args.dry_run else 'Freed'} {mb_freed:.2f} MB of disk space")
    
    if args.dry_run:
        logger.info("\nThis was a dry run. No files were actually deleted.")
        logger.info("To actually delete files, run without the --dry-run flag.")


if __name__ == "__main__":
    main()