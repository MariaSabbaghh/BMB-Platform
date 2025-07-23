"""
Directory Setup Script for BMB Training System

LOCATION: Save this file in your project root directory (same level as app.py)

RUN: python setup_directories.py

This will create all necessary folders for the training system to work properly.
"""

import os

def setup_project_directories():
    """Create the necessary directory structure for the training system"""
    
    print("🚀 Setting up BMB Training System directories...")
    print("=" * 50)
    
    # Base directories
    directories = [
        # Python modules
        'pages',
        'pages/train',
        'routes',
        
        # Templates
        'templates',
        'templates/train',
        
        # Static assets
        'static',
        'static/css',
        'static/js', 
        'static/images',
        'static/visualizations',
        
        # Data directories
        'cleaned_data',
        'trained_models',
        'uploads'
    ]
    
    # Create directories
    created_count = 0
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
            created_count += 1
        else:
            print(f"📁 Exists: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        'pages/__init__.py',
        'pages/train/__init__.py'
    ]
    
    init_created = 0
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization file\n')
            print(f"✅ Created: {init_file}")
            init_created += 1
        else:
            print(f"📄 Exists: {init_file}")
    
    print("=" * 50)
    print(f"🎉 Setup complete! Created {created_count} directories and {init_created} files.")
    print("\n📋 Next steps:")
    print("1. Copy files to these locations:")
    print("   📄 train.html → templates/train/train.html")
    print("   📄 prediction.html → templates/train/prediction.html")
    print("   📄 train_routes.py → routes/train_routes.py")
    print("   📄 train_supervised.py → pages/train/train_supervised.py")
    print("   📄 train_unsupervised.py → pages/train/train_unsupervised.py")
    print("   📄 prediction.css → static/css/prediction.css")
    print("\n2. Add to your train.css file:")
    print("   📝 Copy the advanced settings CSS styles")
    print("\n3. Test the training system:")
    print("   🧪 Upload data → Train models → View results → Check predictions")
    print("\n💡 Remember to add the prediction.css link to your base.html!")

if __name__ == "__main__":
    setup_project_directories()