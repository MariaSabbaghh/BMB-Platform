from flask import Flask, render_template, redirect
import os

app = Flask(__name__)
app.secret_key = "BMB_PLATFORM_UNIFIED"

@app.route("/")
def welcome():
    """Main welcome page with three AI options"""
    return render_template("welcome.html")

@app.route("/predictive")
def redirect_predictive():
    """Redirect to predictive AI module's home page"""
    return redirect("http://localhost:5001/")

@app.route("/generative")
def redirect_generative():
    """Placeholder for generative AI"""
    return '''
    <div style="text-align: center; padding: 50px; font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: white;">
        <div style="max-width: 800px; margin: 0 auto; padding: 40px 20px;">
            <div style="font-size: 120px; margin-bottom: 30px; opacity: 0.9;">🧠</div>
            <div style="display: inline-block; background: rgba(255, 255, 255, 0.2); padding: 8px 20px; border-radius: 25px; font-size: 0.9rem; margin-bottom: 20px; backdrop-filter: blur(10px);">
                🕐 Coming Soon
            </div>
            <h1 style="font-size: 3.5rem; margin-bottom: 20px; font-weight: 700;">Generative AI</h1>
            <p style="font-size: 1.4rem; margin-bottom: 30px; opacity: 0.9; line-height: 1.6;">
                Unlock the power of creativity with advanced AI generation
            </p>
            <p style="font-size: 1.1rem; margin-bottom: 40px; opacity: 0.8; line-height: 1.7;">
                Our Generative AI module will enable you to create high-quality text, images, code, and more. 
                Transform your ideas into intelligent, human-like output with state-of-the-art models.
            </p>
            <a href="/" style="display: inline-flex; align-items: center; gap: 10px; background: rgba(255, 255, 255, 0.2); color: white; text-decoration: none; padding: 15px 30px; border-radius: 50px; font-weight: 600; transition: all 0.3s ease; backdrop-filter: blur(10px); border: 2px solid rgba(255, 255, 255, 0.3);">
                ← Back to Platform
            </a>
        </div>
    </div>
    '''

@app.route("/computer-vision")
def redirect_computer_vision():
    """Redirect to computer vision module"""
    return redirect("http://localhost:5002")

if __name__ == "__main__":
    print("🚀 BMB Platform starting on http://localhost:5000")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("=" * 50)
    print("⚠️  IMPORTANT: This is the main platform only!")
    print("   To use Predictive AI: Start your pred_AI/app.py on port 5001")
    print("   To use Computer Vision: Start your computer_vision/app.py on port 5002")
    print("=" * 50)
    app.run(debug=True, port=5000, use_reloader=False)