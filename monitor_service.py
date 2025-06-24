#!/usr/bin/env python3
import subprocess
import requests
import time
import logging
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/log/monitor.log'),
        logging.StreamHandler()
    ]
)

# Configuration
CHECK_INTERVAL = 300  # 5 minutes
MAX_RETRIES = 3
ALERT_EMAIL = os.getenv('ALERT_EMAIL', '')  # Set this in .env file
SMTP_SERVER = os.getenv('SMTP_SERVER', '')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')

# Log configuration on startup
logging.info("Monitoring configuration loaded:")
logging.info(f"SMTP Server: {SMTP_SERVER}")
logging.info(f"SMTP Port: {SMTP_PORT}")
logging.info(f"SMTP User: {SMTP_USER}")
logging.info(f"Alert Email: {ALERT_EMAIL}")

def check_service_status():
    """Check if the coding-agent service is running"""
    try:
        result = subprocess.run(['systemctl', 'is-active', 'coding-agent.service'], 
                              capture_output=True, text=True)
        return result.stdout.strip() == 'active'
    except Exception as e:
        logging.error(f"Error checking service status: {e}")
        return False

def check_port_listening():
    """Check if port 8000 is being listened to"""
    try:
        result = subprocess.run(['netstat', '-tlpn'], capture_output=True, text=True)
        return '127.0.0.1:8000' in result.stdout
    except Exception as e:
        logging.error(f"Error checking port: {e}")
        return False

def check_local_endpoint():
    """Check if the local endpoint is responding"""
    try:
        response = requests.get('http://127.0.0.1:8000/', timeout=5)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Error checking local endpoint: {e}")
        return False

def check_nginx_proxy():
    """Check if nginx is properly proxying requests"""
    try:
        response = requests.get('https://coding-agent.de/', timeout=5)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Error checking nginx proxy: {e}")
        return False

def send_alert(message):
    """Send alert email"""
    if not all([ALERT_EMAIL, SMTP_SERVER, SMTP_USER, SMTP_PASSWORD]):
        logging.warning("Email alert not configured. Skipping alert.")
        return

    try:
        msg = MIMEText(message)
        msg['Subject'] = f'Coding Agent Service Alert - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_EMAIL

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        logging.info("Alert email sent successfully")
    except Exception as e:
        logging.error(f"Error sending alert email: {e}")

def restart_service():
    """Attempt to restart the service"""
    try:
        # Try to restart the service
        try:
            subprocess.run(['sudo', 'systemctl', 'restart', 'coding-agent.service'], check=True)
            logging.info("Service restarted successfully")
            # Add a delay to allow the service to fully start
            time.sleep(10)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to restart service: {str(e)}")
            return False
    except Exception as e:
        logging.error(f"Error restarting service: {e}")
        return False

def main():
    """Main monitoring loop"""
    logging.info("Starting service monitoring")
    
    while True:
        issues = []
        
        # Check service status
        if not check_service_status():
            issues.append("Service is not running")
            if restart_service():
                logging.info("Service restarted successfully")
            else:
                issues.append("Failed to restart service")
        
        # Check port
        if not check_port_listening():
            issues.append("Port 8000 is not being listened to")
        
        # Check local endpoint
        if not check_local_endpoint():
            issues.append("Local endpoint is not responding")
        
        # Check nginx proxy
        if not check_nginx_proxy():
            issues.append("Nginx proxy is not working")
        
        # If there are issues, send alert
        if issues:
            message = "Coding Agent Service Issues Detected:\n\n" + "\n".join(issues)
            logging.error(message)
            send_alert(message)
        else:
            logging.info("All checks passed")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main() 