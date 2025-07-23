from flask import Flask, render_template
from routes.object_detection import object_detection_bp
app = Flask(__name__)
app.secret_key = "CV_TYPES_DEMO"

# Register the object detection blueprint
app.register_blueprint(object_detection_bp, url_prefix='/object-detection')

@app.route("/")
def index():
    """Main page showing computer vision types"""
    return render_template("index.html")

@app.route("/contact")
def contact():
    """Contact page"""
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True, port=5002)