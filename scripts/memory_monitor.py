#!/usr/bin/env python3
"""
Memory monitoring script to prevent laptop shutdown during intensive training.

This script monitors system memory usage and can terminate a training process
if memory usage becomes critical.

Usage:
    python memory_monitor.py --threshold 85 --check-interval 30 --process-name python
"""

import psutil
import time
import argparse
import signal
import sys
import os
from datetime import datetime

def get_memory_usage():
    """Get current memory usage statistics."""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / 1024**3,  # GB
        'available': memory.available / 1024**3,  # GB
        'used': memory.used / 1024**3,  # GB
        'percent': memory.percent
    }

def find_processes_by_name(process_name):
    """Find all processes with the given name."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if process_name.lower() in proc.info['name'].lower():
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def monitor_memory(threshold=85, check_interval=30, process_name='python', dry_run=False):
    """
    Monitor memory usage and optionally terminate processes.
    
    Args:
        threshold: Memory usage percentage threshold (default: 85%)
        check_interval: Time between checks in seconds (default: 30)
        process_name: Name of process to monitor (default: 'python')
        dry_run: If True, only print warnings without terminating processes
    """
    print(f"🔍 Memory Monitor Started")
    print(f"   Threshold: {threshold}%")
    print(f"   Check interval: {check_interval}s")
    print(f"   Target process: {process_name}")
    print(f"   Dry run: {dry_run}")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        while True:
            # Get current memory usage
            mem = get_memory_usage()
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Print current status
            print(f"[{timestamp}] Memory: {mem['used']:.1f}GB/{mem['total']:.1f}GB ({mem['percent']:.1f}%) - Available: {mem['available']:.1f}GB")
            
            # Check if threshold exceeded
            if mem['percent'] > threshold:
                print(f"⚠️  WARNING: Memory usage {mem['percent']:.1f}% exceeds threshold {threshold}%!")
                
                # Find target processes
                processes = find_processes_by_name(process_name)
                if processes:
                    print(f"   Found {len(processes)} process(es) matching '{process_name}':")
                    for proc in processes:
                        try:
                            proc_mem = proc.memory_info().rss / 1024**3
                            print(f"     PID {proc.pid}: {proc_mem:.1f}GB")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    if not dry_run:
                        print(f"🛑 Terminating processes to free memory...")
                        for proc in processes:
                            try:
                                proc.terminate()
                                print(f"   Terminated PID {proc.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                print(f"   Could not terminate PID {proc.pid}")
                        print(f"✅ Processes terminated. Continuing monitoring...")
                    else:
                        print(f"   (Dry run - would terminate processes)")
                else:
                    print(f"   No processes found matching '{process_name}'")
            
            # Wait before next check
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Memory monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Error in memory monitor: {e}")
    finally:
        print(f"   Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    parser = argparse.ArgumentParser(description='Monitor memory usage and prevent system overload')
    parser.add_argument('--threshold', type=float, default=85.0,
                       help='Memory usage threshold percentage (default: 85)')
    parser.add_argument('--check-interval', type=int, default=30,
                       help='Check interval in seconds (default: 30)')
    parser.add_argument('--process-name', type=str, default='python',
                       help='Process name to monitor (default: python)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only print warnings, do not terminate processes')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 < args.threshold < 100:
        print("Error: Threshold must be between 0 and 100")
        sys.exit(1)
    
    if args.check_interval < 1:
        print("Error: Check interval must be at least 1 second")
        sys.exit(1)
    
    # Start monitoring
    monitor_memory(
        threshold=args.threshold,
        check_interval=args.check_interval,
        process_name=args.process_name,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()

