#!/bin/bash

# Check if backup file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <backup_file>"
    echo "Example: $0 /home/aipipeline/backups/coding_agent_backup_20250526_151004.tar.gz"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="/home/aipipeline/restore_temp"
APP_DIR="/home/aipipeline/coding-agent"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "Starting restore at $(date)"
echo "Backup file: $BACKUP_FILE"

# Create temporary restore directory
echo "Creating temporary restore directory..."
rm -rf "${RESTORE_DIR}"
mkdir -p "${RESTORE_DIR}"

# Extract backup
echo "Extracting backup..."
tar -xzf "${BACKUP_FILE}" -C "${RESTORE_DIR}"

# Stop services
echo "Stopping services..."
sudo systemctl stop nginx coding-agent.service

# Restore application code
echo "Restoring application code..."
rm -rf "${APP_DIR}"/*
cp -r "${RESTORE_DIR}"/* "${APP_DIR}/"

# Restore configuration files
echo "Restoring configuration files..."
sudo cp "${RESTORE_DIR}/config/coding-agent.conf" /etc/nginx/sites-enabled/
sudo cp "${RESTORE_DIR}/config/coding-agent.service" /etc/systemd/system/

# Restore SSL certificates
echo "Restoring SSL certificates..."
sudo mkdir -p /etc/letsencrypt/live/coding-agent.de
sudo cp -r "${RESTORE_DIR}/ssl/coding-agent.de"/* /etc/letsencrypt/live/coding-agent.de/

# Set proper permissions
echo "Setting permissions..."
sudo chown -R aipipeline:aipipeline "${APP_DIR}"
sudo chmod -R 755 "${APP_DIR}"

# Reload systemd and restart services
echo "Reloading systemd and restarting services..."
sudo systemctl daemon-reload
sudo systemctl restart nginx coding-agent.service

# Clean up
echo "Cleaning up..."
rm -rf "${RESTORE_DIR}"

echo "Restore completed at $(date)"
echo "Please verify that the application is running correctly"
echo "You can check the service status with: sudo systemctl status nginx coding-agent.service" 