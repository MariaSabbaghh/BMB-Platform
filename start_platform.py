#!/usr/bin/env python3
"""
BMB Platform Startup Script
Starts all three services: Main Platform, Predictive AI, and Computer Vision
"""

import subprocess
import time
import sys
import os
import webbrowser
from threading import Timer

def check_module_structure():
    """Check if modules have the correct structure"""
    issues = []
    
    # Check Predictive AI
    if os.path.exists("pred_AI"):
        if not os.path.exists("pred_AI/app.py"):
            issues.append("pred_AI/app.py not found")
        if not os.path.exists("pred_AI/templates"):
            issues.append("pred_AI/templates/ folder not found")
    else:
        issues.append("pred_AI/ folder not found")
    
    # Check Computer Vision
    if os.path.exists("computer_vision"):
        if not os.path.exists("computer_vision/app.py"):
            issues.append("computer_vision/app.py not found")
        if not os.path.exists("computer_vision/templates"):
            issues.append("computer_vision/templates/ folder not found")
    else:
        issues.append("computer_vision/ folder not found")
    
    return issues

def open_browser_delayed():
    """Open browser after a delay to allow services to start"""
    time.sleep(8)  # Wait 8 seconds for all services to start
    try:
        webbrowser.open('http://localhost:5000')
    except:
        pass

def start_all_services():
    """Start all BMB Platform services"""
    
    print("🌟 BMB Platform Startup")
    print("=" * 60)
    
    # Check module structure
    issues = check_module_structure()
    
    # Check if modules exist
    pred_ai_exists = os.path.exists("pred_AI/app.py")
    cv_exists = os.path.exists("computer_vision/app.py")
    main_exists = os.path.exists("main_platform.py")
    
    if not main_exists:
        print("❌ main_platform.py not found!")
        print("   Please save the main platform code as 'main_platform.py'")
        return
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print()
    
    if not pred_ai_exists:
        print("⚠️  Predictive AI module not ready!")
        print("   Please ensure pred_AI/app.py exists")
    
    if not cv_exists:
        print("⚠️  Computer Vision module not ready!")
        print("   Please ensure computer_vision/app.py exists")
    
    print("📋 Starting services...")
    processes = []
    
    try:
        # Start main platform
        print("📱 Starting Main Platform (Port 5000)...")
        main_process = subprocess.Popen([sys.executable, "main_platform.py"])
        processes.append(("Main Platform", main_process))
        time.sleep(2)
        
        # Start Predictive AI if available
        if pred_ai_exists:
            print("🤖 Starting Predictive AI Module (Port 5001)...")
            pred_process = subprocess.Popen([sys.executable, "app.py"], cwd="pred_AI")
            processes.append(("Predictive AI", pred_process))
            time.sleep(2)
        else:
            print("⏭️  Skipping Predictive AI (not available)")
        
        # Start Computer Vision if available
        if cv_exists:
            print("👁️  Starting Computer Vision Module (Port 5002)...")
            cv_process = subprocess.Popen([sys.executable, "app.py"], cwd="computer_vision")
            processes.append(("Computer Vision", cv_process))
            time.sleep(2)
        else:
            print("⏭️  Skipping Computer Vision (not available)")
        
        # Open browser in a separate thread
        browser_timer = Timer(3.0, open_browser_delayed)
        browser_timer.start()
        
        print("=" * 60)
        print("✅ Platform services started!")
        print("🌐 Main Platform: http://localhost:5000")
        if pred_ai_exists:
            print("🤖 Predictive AI: http://localhost:5001")
        if cv_exists:
            print("👁️  Computer Vision: http://localhost:5002")
        print("=" * 60)
        print("🎯 Browser will open automatically...")
        print("📝 Or manually go to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop all services")
        print("=" * 60)
        
        # Keep all processes running
        while True:
            time.sleep(1)
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n⚠️  {name} service stopped unexpectedly")
                    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")
        for name, process in processes:
            print(f"   Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        print("✅ All services stopped gracefully")

def setup_guide():
    """Show setup guide if modules are missing"""
    print("📚 SETUP GUIDE")
    print("=" * 40)
    print("1. Create folder structure:")
    print("   📁 your-project/")
    print("   ├── main_platform.py")
    print("   ├── start_platform.py")
    print("   ├── templates/welcome.html")
    print("   ├── pred_AI/")
    print("   │   ├── app.py")
    print("   │   └── templates/")
    print("   └── computer_vision/")
    print("       ├── app.py")
    print("       └── templates/")
    print()
    print("2. Copy your existing projects:")
    print("   - Copy BMB_drive contents → pred_AI/")
    print("   - Copy computer_vision project → computer_vision/")
    print()
    print("3. Run: python start_platform.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        setup_guide()
    else:
        start_all_services()