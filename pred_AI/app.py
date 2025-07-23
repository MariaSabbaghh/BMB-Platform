from flask import Flask, render_template, redirect, url_for, current_app, request
import os

app = Flask(__name__)
app.secret_key = "BMB_DRIVE"

# Add these lines:
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["CLEANED_DATA_FOLDER"] = os.path.join(os.getcwd(), "cleaned_data")
app.config['CLEANED_DATA_DIR'] = 'cleaned_data'

# Ensure cleaned data directory exists
os.makedirs(app.config['CLEANED_DATA_DIR'], exist_ok=True)
os.makedirs('pages/train', exist_ok=True)  
os.makedirs('trained_models', exist_ok=True)
os.makedirs('projects', exist_ok=True)

# Import all your existing blueprints
from routes.contact_routes import contact_bp
from routes.connection_routes import connection_bp
from routes.catalog_routes import catalog_bp
from routes.train_routes import train_bp
from routes.xai_routes import xai_bp
from routes.analysis_routes import analysis_bp

# FIX: Register all blueprints with CORRECT prefixes - This was the main issue!
app.register_blueprint(contact_bp, url_prefix='/contact')
app.register_blueprint(connection_bp, url_prefix='/connection')
app.register_blueprint(catalog_bp, url_prefix='/catalog')
app.register_blueprint(train_bp, url_prefix='/train')
app.register_blueprint(xai_bp, url_prefix='/xai')
app.register_blueprint(analysis_bp, url_prefix='/xai')  

@app.route("/")
def index():
    """Main home page - shows predictive AI interface directly"""
    return render_template("index.html")

@app.route("/welcome")
def welcome():
    """Welcome page - kept for compatibility"""
    return render_template("welcome.html")

@app.route("/home")
def home():
    """Redirect to main index page"""
    return redirect(url_for('index'))

@app.route("/contact")
def contact():
    return redirect(url_for('contact.contact'))

@app.route("/connection")
def connection():
    return render_template("connection/generic_domain_index.html")

@app.route("/catalog")
def catalog():
    return redirect(url_for('catalog.catalog_index'))

@app.route("/train")
def train():
    """Main train page - shows available files for training or prompts to select from catalog"""
    # Check if a specific file was requested
    requested_file = request.args.get('file') or request.args.get('filename')
    
    if requested_file:
        # Validate that the requested file exists
        cleaned_data_dir = app.config['CLEANED_DATA_DIR']
        filepath = os.path.join(cleaned_data_dir, requested_file)
        
        if os.path.exists(filepath) and requested_file.endswith('.csv'):
            # File exists, redirect to training page with filename
            return redirect(url_for('train_bp.train_page', filename=requested_file))
        else:
            # File doesn't exist, show train page with error
            return redirect(url_for('train_bp.train_page', file_missing=requested_file))
    
    # No file specified, get list of available files
    cleaned_data_dir = app.config['CLEANED_DATA_DIR']
    files = []
    if os.path.exists(cleaned_data_dir):
        files = [f for f in os.listdir(cleaned_data_dir) if f.endswith('.csv')]
   
    if not files:
        # No files available, show train page without filename (will show catalog prompt)
        return redirect(url_for('train_bp.train_page'))
   
    # If only one file, go directly to training
    if len(files) == 1:
        return redirect(url_for('train_bp.train_page', filename=files[0]))
   
    # Multiple files available, still show train page without filename to prompt catalog selection
    return redirect(url_for('train_bp.train_page'))

@app.route("/xai")
def xai():
    return redirect(url_for('xai.xai_page'))

@app.route("/generative")
def generative():
    """Generative AI placeholder"""
    try:
        return render_template("generative.html")
    except:
        return '''
        <div style="text-align: center; padding: 50px; font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: white;">
            <h1>🧠 Generative AI</h1>
            <p style="margin: 20px 0;">Coming soon!</p>
            <a href="/" style="color: white; text-decoration: none; padding: 10px 20px; border: 2px solid white; border-radius: 5px;">← Back to Home</a>
        </div>
        '''

if __name__ == "__main__":
    print("🤖 Starting Predictive AI on port 5001...")
    print("📋 All routes loaded successfully!")
    app.run(debug=True, port=5001)