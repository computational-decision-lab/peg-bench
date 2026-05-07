#!/usr/bin/env python3
"""
01_download_lob_data.py

Download Binance BTCUSDT LOB data from Tardis for the specified date range.

执行位置: ☁️ 云端（AutoDL服务器）
数据范围: 2025-10-01 to 2025-10-30 (30 days)

Usage:
    python src/data/01_download_lob_data.py \
        --symbol BTCUSDT \
        --start-date 2025-10-01 \
        --end-date 2025-10-30 \
        --output-dir data/tardis
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import requests
from tqdm import tqdm


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def generate_date_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Generate list of dates between start and end (inclusive)."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def download_file(url: str, output_path: Path, api_key: str = None) -> bool:
    """
    Download file from URL with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save file
        api_key: Optional Tardis API key

    Returns:
        True if successful, False otherwise
    """
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_tardis_lob_data(
    symbol: str,
    date: datetime,
    output_dir: Path,
    api_key: str = None,
) -> bool:
    """
    Download Tardis LOB data for a specific date.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        date: Date to download
        output_dir: Output directory
        api_key: Tardis API key

    Returns:
        True if successful, False otherwise
    """
    # Tardis URL format
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}_{date_str}_incremental_book_L2.csv.gz"

    # Construct URL (adjust based on actual Tardis API)
    # This is a placeholder - actual URL structure may differ
    url = f"https://datasets.tardis.dev/v1/binance-futures/{date.year}/{date.month:02d}/{filename}"

    output_path = output_dir / filename

    # Skip if already exists
    if output_path.exists():
        print(f"File already exists: {output_path}")
        return True

    print(f"Downloading {symbol} data for {date_str}...")
    success = download_file(url, output_path, api_key)

    if success:
        print(f"Successfully downloaded: {output_path}")

        # Decompress if needed
        if output_path.suffix == '.gz':
            import gzip
            import shutil

            decompressed_path = output_path.with_suffix('')
            print(f"Decompressing to {decompressed_path}...")

            with gzip.open(output_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            print(f"Decompressed: {decompressed_path}")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance BTCUSDT LOB data from Tardis"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tardis"),
        help="Output directory (default: data/tardis)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("TARDIS_API_KEY"),
        help="Tardis API key (default: from TARDIS_API_KEY env var)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip download if files already exist"
    )

    args = parser.parse_args()

    # Parse dates
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    print(f"Downloading {args.symbol} LOB data")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Generate date range
    dates = generate_date_range(start_date, end_date)
    print(f"Total days: {len(dates)}")
    print()

    # Download each day
    success_count = 0
    failed_dates = []

    for date in dates:
        if args.offline:
            # Check if file exists
            date_str = date.strftime("%Y-%m-%d")
            filename = f"{args.symbol}_{date_str}_incremental_book_L2.csv"
            if (args.output_dir / filename).exists():
                print(f"Skipping {date_str} (offline mode, file exists)")
                success_count += 1
                continue

        success = download_tardis_lob_data(
            args.symbol,
            date,
            args.output_dir,
            args.api_key
        )

        if success:
            success_count += 1
        else:
            failed_dates.append(date)

        print()

    # Summary
    print("=" * 60)
    print(f"Download complete!")
    print(f"Successful: {success_count}/{len(dates)}")

    if failed_dates:
        print(f"Failed: {len(failed_dates)}")
        print("Failed dates:")
        for date in failed_dates:
            print(f"  - {date.strftime('%Y-%m-%d')}")
    else:
        print("All downloads successful!")

    print("=" * 60)


if __name__ == "__main__":
    main()
