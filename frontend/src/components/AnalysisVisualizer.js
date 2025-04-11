import React, { useState } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Typography,
  Paper,
  Grid,
  Chip,
  Card,
  CardContent,
  CardMedia
} from '@mui/material';
import {
  Image,
  Audiotrack,
  Videocam,
  CheckCircle,
  Error,
  Warning
} from '@mui/icons-material';

const AnalysisVisualizer = ({ result }) => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const getBackgroundColor = (confidence, threshold = 0.7) => {
    if (confidence >= threshold) return 'success.light';
    if (confidence >= threshold * 0.7) return 'warning.light';
    return 'error.light';
  };

  const renderImageResult = (item) => (
    <Card sx={{ mb: 2 }}>
      <CardMedia
        component="img"
        height="200"
        image={item.url}
        alt={item.filename}
      />
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {item.filename}
        </Typography>
        <Grid container spacing={2}>
          {item.detections.map((detection, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Paper
                sx={{
                  p: 2,
                  backgroundColor: getBackgroundColor(detection.confidence)
                }}
              >
                <Typography variant="subtitle1">
                  {detection.label}
                </Typography>
                <Typography variant="body2">
                  Confidence: {(detection.confidence * 100).toFixed(1)}%
                </Typography>
                {detection.explanation && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {detection.explanation}
                  </Typography>
                )}
              </Paper>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );

  const renderAudioResult = (item) => (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Audiotrack sx={{ mr: 1 }} />
          <Typography variant="h6">
            {item.filename}
          </Typography>
        </Box>
        <Grid container spacing={2}>
          {item.detections.map((detection, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Paper
                sx={{
                  p: 2,
                  backgroundColor: getBackgroundColor(detection.confidence)
                }}
              >
                <Typography variant="subtitle1">
                  {detection.label}
                </Typography>
                <Typography variant="body2">
                  Confidence: {(detection.confidence * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  Time: {detection.start_time}s - {detection.end_time}s
                </Typography>
                {detection.explanation && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {detection.explanation}
                  </Typography>
                )}
              </Paper>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );

  const renderVideoResult = (item) => (
    <Card sx={{ mb: 2 }}>
      <CardMedia
        component="video"
        height="300"
        controls
        src={item.url}
      />
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {item.filename}
        </Typography>
        <Grid container spacing={2}>
          {item.detections.map((detection, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Paper
                sx={{
                  p: 2,
                  backgroundColor: getBackgroundColor(detection.confidence)
                }}
              >
                <Typography variant="subtitle1">
                  {detection.label}
                </Typography>
                <Typography variant="body2">
                  Confidence: {(detection.confidence * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  Frame: {detection.frame_number}
                </Typography>
                <Typography variant="body2">
                  Time: {detection.timestamp}s
                </Typography>
                {detection.explanation && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {detection.explanation}
                  </Typography>
                )}
              </Paper>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );

  return (
    <Box>
      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        sx={{ mb: 3 }}
      >
        <Tab icon={<Image />} label="Images" />
        <Tab icon={<Audiotrack />} label="Audio" />
        <Tab icon={<Videocam />} label="Video" />
      </Tabs>

      {activeTab === 0 && result.image_results?.map((item, index) => (
        <div key={index}>{renderImageResult(item)}</div>
      ))}

      {activeTab === 1 && result.audio_results?.map((item, index) => (
        <div key={index}>{renderAudioResult(item)}</div>
      ))}

      {activeTab === 2 && result.video_results?.map((item, index) => (
        <div key={index}>{renderVideoResult(item)}</div>
      ))}
    </Box>
  );
};

export default AnalysisVisualizer; 