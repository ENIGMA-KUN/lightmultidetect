import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  TextField,
  Button,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Alert
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import ResultCard from '../components/ResultCard';
import api from '../services/api';

const ResultsPage = () => {
  const navigate = useNavigate();
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);

  const fetchResults = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await api.getAnalysisResults({
        search: searchQuery
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to fetch results');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, [searchQuery]);

  const handleSearch = (e) => {
    setSearchQuery(e.target.value);
  };

  const openDeleteDialog = (taskId) => {
    setSelectedResult(taskId);
    setDeleteDialogOpen(true);
  };

  const handleDelete = async () => {
    try {
      await api.deleteAnalysis(selectedResult);
      setResults(results.filter(result => result.id !== selectedResult));
      setDeleteDialogOpen(false);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to delete result');
    }
  };

  const navigateToUpload = () => {
    navigate('/upload');
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Analysis Results</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={navigateToUpload}
        >
          New Analysis
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <TextField
        fullWidth
        variant="outlined"
        placeholder="Search results..."
        value={searchQuery}
        onChange={handleSearch}
        sx={{ mb: 3 }}
      />

      <Grid container spacing={3}>
        {results.map((result) => (
          <Grid item xs={12} key={result.id}>
            <ResultCard
              result={result}
              onDelete={() => openDeleteDialog(result.id)}
            />
          </Grid>
        ))}
      </Grid>

      {results.length === 0 && !loading && (
        <Box sx={{ textAlign: 'center', mt: 4 }}>
          <Typography variant="h6" color="text.secondary">
            No results found
          </Typography>
          <Button
            variant="contained"
            onClick={navigateToUpload}
            sx={{ mt: 2 }}
          >
            Start New Analysis
          </Button>
        </Box>
      )}

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

export default ResultsPage; 