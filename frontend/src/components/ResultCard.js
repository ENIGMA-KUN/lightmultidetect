import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  IconButton,
  Collapse,
  Box,
  Chip,
  Grid
} from '@mui/material';
import {
  ExpandMore,
  ExpandLess,
  Delete,
  Visibility,
  Image,
  Audiotrack,
  Videocam
} from '@mui/icons-material';

const ResultCard = ({ result, onDelete }) => {
  const navigate = useNavigate();
  const [expanded, setExpanded] = useState(false);

  const handleExpandClick = () => {
    setExpanded(!expanded);
  };

  const handleViewDetails = () => {
    navigate(`/results/${result.id}`);
  };

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

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getFileCount = () => {
    return result.files?.length || 0;
  };

  const renderFileTypeCounts = () => {
    const counts = {
      image: 0,
      audio: 0,
      video: 0
    };

    result.files?.forEach(file => {
      if (file.type.startsWith('image/')) counts.image++;
      else if (file.type.startsWith('audio/')) counts.audio++;
      else if (file.type.startsWith('video/')) counts.video++;
    });

    return (
      <Box sx={{ mt: 1 }}>
        {counts.image > 0 && (
          <Chip
            icon={<Image />}
            label={`${counts.image} Images`}
            size="small"
            sx={{ mr: 1 }}
          />
        )}
        {counts.audio > 0 && (
          <Chip
            icon={<Audiotrack />}
            label={`${counts.audio} Audio`}
            size="small"
            sx={{ mr: 1 }}
          />
        )}
        {counts.video > 0 && (
          <Chip
            icon={<Videocam />}
            label={`${counts.video} Video`}
            size="small"
          />
        )}
      </Box>
    );
  };

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs>
            <Typography variant="h6" component="div">
              Analysis #{result.id}
            </Typography>
          </Grid>
          <Grid item>
            {getStatusIcon(result.status)}
          </Grid>
        </Grid>

        <Typography color="text.secondary" sx={{ mb: 1.5 }}>
          Created: {formatTimestamp(result.created_at)}
        </Typography>

        <Typography variant="body2">
          Files: {getFileCount()}
        </Typography>

        {renderFileTypeCounts()}

        <Collapse in={expanded} timeout="auto" unmountOnExit>
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Details:
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Modality: {result.modality}
              <br />
              Confidence Threshold: {result.confidence_threshold}
              <br />
              Processing Time: {result.processing_time}s
            </Typography>
          </Box>
        </Collapse>
      </CardContent>

      <CardActions disableSpacing>
        <IconButton onClick={handleViewDetails}>
          <Visibility />
        </IconButton>
        <IconButton onClick={onDelete}>
          <Delete />
        </IconButton>
        <Box sx={{ flexGrow: 1 }} />
        <IconButton
          onClick={handleExpandClick}
          aria-expanded={expanded}
          aria-label="show more"
        >
          {expanded ? <ExpandLess /> : <ExpandMore />}
        </IconButton>
      </CardActions>
    </Card>
  );
};

export default ResultCard; 