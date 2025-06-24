#!/bin/bash

# Set backup directory
BACKUP_DIR="/home/aipipeline/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="coding_agent_backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Create backup directory
mkdir -p "${BACKUP_PATH}"

echo "Starting backup at $(date)"

# Backup application code
echo "Backing up application code..."
cp -r /home/aipipeline/coding-agent/* "${BACKUP_PATH}/"

# Backup configuration files
echo "Backing up configuration files..."
mkdir -p "${BACKUP_PATH}/config"
cp /etc/nginx/sites-enabled/coding-agent.conf "${BACKUP_PATH}/config/"
cp /etc/systemd/system/coding-agent.service "${BACKUP_PATH}/config/"

# Backup SSL certificates
echo "Backing up SSL certificates..."
mkdir -p "${BACKUP_PATH}/ssl"
cp -r /etc/letsencrypt/live/coding-agent.de "${BACKUP_PATH}/ssl/"

# Create a tar archive
echo "Creating archive..."
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"

# Remove the temporary directory
rm -rf "${BACKUP_PATH}"

# Keep only the last 5 backups
echo "Cleaning up old backups..."
ls -t "${BACKUP_DIR}"/coding_agent_backup_*.tar.gz | tail -n +6 | xargs -r rm

echo "Backup completed at $(date)"
echo "Backup saved to: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" 