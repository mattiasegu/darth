# Bash script for downloading large files from Google Drive
# Follow the instructions at the following link to get your access token: 
# https://stackoverflow.com/questions/65312867/how-to-download-large-file-from-google-drive-from-terminal-gdown-doesnt-work

ACCESS_TOKEN=$1  # your access token
FILE_ID=$2  # the file id of the file to download
FILE_NAME=$3 # the location where to save the downloaded file, e.g. data/dancetrack/train.zip

curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/$FILE_ID?alt=media -o $FILE_NAME 

