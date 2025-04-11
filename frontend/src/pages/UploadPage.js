import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Paper,
  Alert,
  CircularProgress
} from '@mui/material';
import UploadDropzone from '../components/UploadDropzone';
import DetectionOptions from '../components/DetectionOptions';
import api from '../services/api';

const UploadPage = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [files, setFiles] = useState([]);
  const [options, setOptions] = useState({
    modality: 'image',
    confidenceThreshold: 0.7,
    explainResults: true
  });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const steps = ['Select Files', 'Configure Options', 'Upload and Process'];

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleFilesChange = (newFiles) => {
    setFiles(newFiles);
  };

  const handleOptionsChange = (newOptions) => {
    setOptions(newOptions);
  };

  const handleUpload = async () => {
    try {
      setLoading(true);
      setError(null);

      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('options', JSON.stringify(options));

      const response = await api.uploadFiles(formData);
      navigate(`/results/${response.data.id}`);
    } catch (err) {
      setError(err.response?.data?.message || 'Upload failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleViewResults = () => {
    navigate('/results');
  };

  const isStepComplete = (step) => {
    switch (step) {
      case 0:
        return files.length > 0;
      case 1:
        return true;
      case 2:
        return true;
      default:
        return false;
    }
  };

  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Select Files
            </Typography>
            <UploadDropzone
              onFilesChange={handleFilesChange}
              maxFiles={10}
              maxSize={100 * 1024 * 1024}
            />
          </Box>
        );
      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Configure Detection Options
            </Typography>
            <DetectionOptions
              options={options}
              onChange={handleOptionsChange}
            />
          </Box>
        );
      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Review and Upload
            </Typography>
            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Selected Files:
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {files.map(file => file.name).join(', ')}
              </Typography>
            </Paper>
            <Paper sx={{ p: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Detection Options:
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Modality: {options.modality}
                <br />
                Confidence Threshold: {options.confidenceThreshold}
                <br />
                Generate Explanations: {options.explainResults ? 'Yes' : 'No'}
              </Typography>
            </Paper>
          </Box>
        );
      default:
        return 'Unknown step';
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {getStepContent(activeStep)}

      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
        {activeStep !== 0 && (
          <Button onClick={handleBack} sx={{ mr: 1 }}>
            Back
          </Button>
        )}
        {activeStep === steps.length - 1 ? (
          <Button
            variant="contained"
            onClick={handleUpload}
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Upload and Process'}
          </Button>
        ) : (
          <Button
            variant="contained"
            onClick={handleNext}
            disabled={!isStepComplete(activeStep)}
          >
            Next
          </Button>
        )}
      </Box>
    </Box>
  );
};

export default UploadPage; 