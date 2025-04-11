import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Paper,
  Chip
} from '@mui/material';
import {
  CloudUpload,
  InsertDriveFile,
  Image,
  Audiotrack,
  Videocam,
  Close
} from '@mui/icons-material';

const UploadDropzone = ({ onFilesChange, maxFiles = 10, maxSize = 100 * 1024 * 1024 }) => {
  const [files, setFiles] = useState([]);
  const [rejected, setRejected] = useState([]);

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setFiles(prev => [...prev, ...acceptedFiles]);
    setRejected(prev => [...prev, ...rejectedFiles]);
    onFilesChange([...files, ...acceptedFiles]);
  }, [files, onFilesChange]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles,
    maxSize,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif'],
      'audio/*': ['.mp3', '.wav', '.ogg'],
      'video/*': ['.mp4', '.avi', '.mov']
    }
  });

  const removeFile = (index) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    setFiles(newFiles);
    onFilesChange(newFiles);
  };

  const clearRejected = () => {
    setRejected([]);
  };

  const getFileIcon = (file) => {
    if (file.type.startsWith('image/')) return <Image />;
    if (file.type.startsWith('audio/')) return <Audiotrack />;
    if (file.type.startsWith('video/')) return <Videocam />;
    return <InsertDriveFile />;
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box>
      <Paper
        {...getRootProps()}
        sx={{
          p: 3,
          textAlign: 'center',
          backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'divider',
          cursor: 'pointer',
          '&:hover': {
            backgroundColor: 'action.hover'
          }
        }}
      >
        <input {...getInputProps()} />
        <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          {isDragActive
            ? 'Drop the files here'
            : 'Drag and drop files here, or click to select files'}
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Supported formats: Images, Audio, Video
          <br />
          Max file size: {formatFileSize(maxSize)}
        </Typography>
      </Paper>

      {files.length > 0 && (
        <List>
          {files.map((file, index) => (
            <ListItem
              key={index}
              secondaryAction={
                <IconButton edge="end" onClick={() => removeFile(index)}>
                  <Close />
                </IconButton>
              }
            >
              <ListItemIcon>{getFileIcon(file)}</ListItemIcon>
              <ListItemText
                primary={file.name}
                secondary={formatFileSize(file.size)}
              />
            </ListItem>
          ))}
        </List>
      )}

      {rejected.length > 0 && (
        <Box mt={2}>
          <Typography variant="subtitle2" color="error" gutterBottom>
            Rejected files:
          </Typography>
          <Box display="flex" gap={1} flexWrap="wrap">
            {rejected.map((file, index) => (
              <Chip
                key={index}
                label={`${file.file.name} (${file.errors[0].message})`}
                onDelete={() => {
                  const newRejected = [...rejected];
                  newRejected.splice(index, 1);
                  setRejected(newRejected);
                }}
                color="error"
                variant="outlined"
              />
            ))}
            <Chip
              label="Clear all"
              onClick={clearRejected}
              color="error"
              variant="outlined"
            />
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default UploadDropzone; 