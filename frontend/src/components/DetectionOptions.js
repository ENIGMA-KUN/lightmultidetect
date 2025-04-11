import React from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  FormControlLabel,
  Switch,
  Typography,
  Paper
} from '@mui/material';

const DetectionOptions = ({ options, onChange }) => {
  const handleModalityChange = (event) => {
    onChange({
      ...options,
      modality: event.target.value
    });
  };

  const handleConfidenceChange = (event, newValue) => {
    onChange({
      ...options,
      confidenceThreshold: newValue
    });
  };

  const handleExplainChange = (event) => {
    onChange({
      ...options,
      explainResults: event.target.checked
    });
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Detection Options
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <FormControl fullWidth>
          <InputLabel>Detection Modality</InputLabel>
          <Select
            value={options.modality}
            label="Detection Modality"
            onChange={handleModalityChange}
          >
            <MenuItem value="image">Image Detection</MenuItem>
            <MenuItem value="audio">Audio Detection</MenuItem>
            <MenuItem value="video">Video Detection</MenuItem>
            <MenuItem value="multimodal">Multi-modal Detection</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Box sx={{ mb: 3 }}>
        <Typography gutterBottom>Confidence Threshold</Typography>
        <Slider
          value={options.confidenceThreshold}
          onChange={handleConfidenceChange}
          min={0}
          max={1}
          step={0.05}
          marks={[
            { value: 0, label: '0' },
            { value: 0.5, label: '0.5' },
            { value: 1, label: '1' }
          ]}
          valueLabelDisplay="auto"
        />
      </Box>

      <FormControlLabel
        control={
          <Switch
            checked={options.explainResults}
            onChange={handleExplainChange}
          />
        }
        label="Generate Explanations"
      />
    </Paper>
  );
};

export default DetectionOptions; 