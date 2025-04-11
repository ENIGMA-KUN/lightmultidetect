import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Alert,
  Chip,
  Divider
} from '@mui/material';
import {
  ArrowBack,
  Delete,
  Download,
  Image,
  Audiotrack,
  Videocam
} from '@mui/icons-material';
import AnalysisVisualizer from '../components/AnalysisVisualizer';
import api from '../services/api';

const ResultDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  const fetchResult = async (showLoading = true) => {
    try {
      if (showLoading) setLoading(true);
      setError(null);
      const response = await api.getAnalysisDetail(id);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to fetch result details');
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  useEffect(() => {
    fetchResult();
  }, [id]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <Chip label="Completed" color="success" />;
      case 'processing':
        return <Chip label="Processing" color="warning" />;
      case 'failed':
        return <Chip label="Failed" color="error" />;
      default:
        return <Chip label={status} />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success.main';
      case 'processing':
        return 'warning.main';
      case 'failed':
        return 'error.main';
      default:
        return 'text.primary';
    }
  };

  const handleDelete = async () => {
    try {
      await api.deleteAnalysis(id);
      navigate('/results');
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to delete result');
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const calculateStats = () => {
    if (!result) return null;

    const stats = {
      totalFiles: result.files?.length || 0,
      imageFiles: 0,
      audioFiles: 0,
      videoFiles: 0,
      totalDetections: 0,
      averageConfidence: 0
    };

    let totalConfidence = 0;
    let detectionCount = 0;

    result.files?.forEach(file => {
      if (file.type.startsWith('image/')) stats.imageFiles++;
      else if (file.type.startsWith('audio/')) stats.audioFiles++;
      else if (file.type.startsWith('video/')) stats.videoFiles++;

      file.detections?.forEach(detection => {
        totalConfidence += detection.confidence;
        detectionCount++;
      });
    });

    stats.totalDetections = detectionCount;
    stats.averageConfidence = detectionCount > 0
      ? (totalConfidence / detectionCount * 100).toFixed(1)
      : 0;

    return stats;
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!result) {
    return (
      <Alert severity="info" sx={{ mt: 2 }}>
        Result not found
      </Alert>
    );
  }

  const stats = calculateStats();

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/results')}
          sx={{ mr: 2 }}
        >
          Back to Results
        </Button>
        <Typography variant="h4" sx={{ flexGrow: 1 }}>
          Analysis Result #{id}
        </Typography>
        {getStatusIcon(result.status)}
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Details
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Created: {formatTimestamp(result.created_at)}
              <br />
              Status: {result.status}
              <br />
              Modality: {result.modality}
              <br />
              Confidence Threshold: {result.confidence_threshold}
              <br />
              Processing Time: {result.processing_time}s
            </Typography>
          </Paper>

          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Statistics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Total Files
                </Typography>
                <Typography variant="h6">
                  {stats.totalFiles}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Total Detections
                </Typography>
                <Typography variant="h6">
                  {stats.totalDetections}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Avg. Confidence
                </Typography>
                <Typography variant="h6">
                  {stats.averageConfidence}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  File Types
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  {stats.imageFiles > 0 && (
                    <Chip icon={<Image />} label={stats.imageFiles} size="small" />
                  )}
                  {stats.audioFiles > 0 && (
                    <Chip icon={<Audiotrack />} label={stats.audioFiles} size="small" />
                  )}
                  {stats.videoFiles > 0 && (
                    <Chip icon={<Videocam />} label={stats.videoFiles} size="small" />
                  )}
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
              <Typography variant="h6">
                Analysis Results
              </Typography>
              <Box>
                <Button
                  startIcon={<Download />}
                  sx={{ mr: 1 }}
                  onClick={() => window.open(result.download_url)}
                >
                  Download
                </Button>
                <Button
                  startIcon={<Delete />}
                  color="error"
                  onClick={() => setDeleteDialogOpen(true)}
                >
                  Delete
                </Button>
              </Box>
            </Box>

            <AnalysisVisualizer result={result} />
          </Paper>
        </Grid>
      </Grid>

      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Result</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this analysis result? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ResultDetailPage; 