from flask import Blueprint, request, render_template, redirect, url_for, flash
import smtplib
import os
from email.message import EmailMessage

contact_bp = Blueprint('contact', __name__)

@contact_bp.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        sender_email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")

        recipient_email = "sabbaghmaria0@gmail.com"
        gmail_email = os.getenv("GMAIL_EMAIL")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")

        if not gmail_email or not gmail_password:
            flash("Email configuration missing. Please set GMAIL_EMAIL and GMAIL_APP_PASSWORD environment variables.", "error")
            return redirect(url_for("contact.contact"))

        msg = EmailMessage()
        msg.set_content(f"From: {name} <{sender_email}>\n\n{message}")
        msg["Subject"] = f"New Contact Message: {subject.title()}"
        msg["From"] = gmail_email
        msg["Reply-To"] = sender_email
        msg["To"] = recipient_email

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(gmail_email, gmail_password)
                smtp.send_message(msg)
            flash("Message sent successfully!", "success")
        except Exception as e:
            print(f"Email error: {str(e)}")
            flash(f"Failed to send email: {str(e)}", "error")
        return redirect(url_for("contact.contact"))
    
    return render_template("contact.html")