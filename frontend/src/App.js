// frontend/src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AuthProvider, useAuth } from './contexts/AuthContext';

// Import pages
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import Dashboard from './pages/Dashboard';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import ResultDetailPage from './pages/ResultDetailPage';
import NotFoundPage from './pages/NotFoundPage';
import ProfilePage from './pages/ProfilePage';

// Import components
import Layout from './components/Layout';

// Create theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3f51b5',
      light: '#757de8',
      dark: '#002984',
    },
    secondary: {
      main: '#f50057',
      light: '#ff4081',
      dark: '#c51162',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.8rem',
      fontWeight: 500,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.3rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1.1rem',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 8px 16px 0 rgba(0,0,0,0.2)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 4px 12px 0 rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

// Protected route component
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  return children;
};

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <Router>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
            <Route path="/" element={
              <ProtectedRoute>
                <Layout>
                  <Dashboard />
                </Layout>
              </ProtectedRoute>
            } />
            <Route path="/upload" element={
              <ProtectedRoute>
                <Layout>
                  <UploadPage />
                </Layout>
              </ProtectedRoute>
            } />
            <Route path="/results" element={
              <ProtectedRoute>
                <Layout>
                  <ResultsPage />
                </Layout>
              </ProtectedRoute>
            } />
            <Route path="/results/:id" element={
              <ProtectedRoute>
                <Layout>
                  <ResultDetailPage />
                </Layout>
              </ProtectedRoute>
            } />
            <Route path="/profile" element={
              <ProtectedRoute>
                <Layout>
                  <ProfilePage />
                </Layout>
              </ProtectedRoute>
            } />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;


// frontend/src/contexts/AuthContext.js
import React, { createContext, useState, useContext, useEffect } from 'react';
import api from '../services/api';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Check if user is already logged in
    const checkAuth = async () => {
      const token = localStorage.getItem('token');
      
      if (token) {
        try {
          // Set token in API headers
          api.setAuthToken(token);
          
          // Get user data
          const response = await api.get('/users/me');
          
          setUser(response.data);
          setIsAuthenticated(true);
        } catch (err) {
          // Token might be invalid or expired
          localStorage.removeItem('token');
          api.setAuthToken(null);
          setError(err.message);
        }
      }
      
      setLoading(false);
    };

    checkAuth();
  }, []);

  const login = async (username, password) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await api.login(username, password);
      const { access_token } = response.data;
      
      // Save token to local storage
      localStorage.setItem('token', access_token);
      
      // Set token in API headers
      api.setAuthToken(access_token);
      
      // Get user data
      const userResponse = await api.get('/users/me');
      
      setUser(userResponse.data);
      setIsAuthenticated(true);
      
      return userResponse.data;
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const register = async (username, email, password) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await api.register(username, email, password);
      
      // Automatically log in after registration
      await login(username, password);
      
      return response.data;
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    // Remove token from local storage
    localStorage.removeItem('token');
    
    // Remove token from API headers
    api.setAuthToken(null);
    
    // Reset state
    setUser(null);
    setIsAuthenticated(false);
  };

  const value = {
    user,
    isAuthenticated,
    loading,
    error,
    login,
    register,
    logout
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};


// frontend/src/services/api.js
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const axiosInstance = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Intercept responses to handle common errors
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized errors (e.g., token expired)
      localStorage.removeItem('token');
      // Redirect to login if needed
    }
    return Promise.reject(error);
  }
);

const api = {
  setAuthToken: (token) => {
    if (token) {
      axiosInstance.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete axiosInstance.defaults.headers.common['Authorization'];
    }
  },
  
  // Auth endpoints
  login: (username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    return axiosInstance.post('/users/token', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
  },
  
  register: (username, email, password) => {
    return axiosInstance.post('/users/register', {
      username,
      email,
      password,
    });
  },
  
  // User endpoints
  getUser: () => axiosInstance.get('/users/me'),
  
  updateUser: (data) => axiosInstance.put('/users/me', data),
  
  // Detection endpoints
  uploadMedia: (files, options) => {
    const formData = new FormData();
    
    // Add files
    files.forEach((file) => {
      formData.append('files', file);
    });
    
    // Add options
    if (options.modalities) {
      formData.append('modalities', options.modalities);
    }
    
    if (options.confidence_threshold) {
      formData.append('confidence_threshold', options.confidence_threshold);
    }
    
    if (options.explain_results !== undefined) {
      formData.append('explain_results', options.explain_results);
    }
    
    return axiosInstance.post('/detection/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: options.onProgress,
    });
  },
  
  checkStatus: (taskId) => axiosInstance.get(`/detection/status/${taskId}`),
  
  getResult: (taskId) => axiosInstance.get(`/detection/result/${taskId}`),
  
  // Generic methods
  get: (endpoint) => axiosInstance.get(endpoint),
  
  post: (endpoint, data) => axiosInstance.post(endpoint, data),
  
  put: (endpoint, data) => axiosInstance.put(endpoint, data),
  
  delete: (endpoint) => axiosInstance.delete(endpoint),
};

export default api;


// frontend/src/components/Layout.js
import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Box,
  Toolbar,
  IconButton,
  Typography,
  Menu,
  MenuItem,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  Button,
  useMediaQuery,
  Container,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  CloudUpload as UploadIcon,
  Assessment as ResultsIcon,
  Person as ProfileIcon,
  ExitToApp as LogoutIcon,
  ChevronLeft as ChevronLeftIcon,
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';

const drawerWidth = 240;

const Layout = ({ children }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('md'));
  
  const [drawerOpen, setDrawerOpen] = useState(!isSmallScreen);
  const [anchorEl, setAnchorEl] = useState(null);
  
  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };
  
  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleNavigate = (path) => {
    navigate(path);
    if (isSmallScreen) {
      setDrawerOpen(false);
    }
  };
  
  const handleLogout = () => {
    handleMenuClose();
    logout();
    navigate('/login');
  };
  
  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Upload', icon: <UploadIcon />, path: '/upload' },
    { text: 'Results', icon: <ResultsIcon />, path: '/results' },
  ];
  
  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          ...(drawerOpen && {
            marginLeft: drawerWidth,
            width: `calc(100% - ${drawerWidth}px)`,
            transition: theme.transitions.create(['width', 'margin'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
          }),
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            LightMultiDetect
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography variant="body2" sx={{ mr: 2 }}>
              {user?.username}
            </Typography>
            <IconButton
              size="large"
              edge="end"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleProfileMenuOpen}
              color="inherit"
            >
              <Avatar
                alt={user?.username}
                src="/static/images/avatar/1.jpg"
                sx={{ width: 32, height: 32 }}
              />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>
      
      <Menu
        id="menu-appbar"
        anchorEl={anchorEl}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        keepMounted
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => { handleMenuClose(); navigate('/profile'); }}>
          <ListItemIcon>
            <ProfileIcon fontSize="small" />
          </ListItemIcon>
          <Typography variant="inherit">Profile</Typography>
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleLogout}>
          <ListItemIcon>
            <LogoutIcon fontSize="small" />
          </ListItemIcon>
          <Typography variant="inherit">Logout</Typography>
        </MenuItem>
      </Menu>
      
      <Drawer
        variant={isSmallScreen ? 'temporary' : 'persistent'}
        open={drawerOpen}
        onClose={handleDrawerToggle}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            backgroundColor: theme.palette.background.paper,
            borderRight: `1px solid ${theme.palette.divider}`,
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', mt: 2 }}>
          {!isSmallScreen && (
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', px: 1 }}>
              <IconButton onClick={handleDrawerToggle}>
                <ChevronLeftIcon />
              </IconButton>
            </Box>
          )}
          
          <List>
            {menuItems.map((item) => (
              <ListItem
                button
                key={item.text}
                onClick={() => handleNavigate(item.path)}
                selected={location.pathname === item.path}
                sx={{
                  '&.Mui-selected': {
                    backgroundColor: 'rgba(63, 81, 181, 0.12)',
                    '&:hover': {
                      backgroundColor: 'rgba(63, 81, 181, 0.2)',
                    },
                  },
                  my: 0.5,
                  borderRadius: 1,
                  mx: 1,
                }}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            ))}
          </List>
          
          <Divider sx={{ my: 2 }} />
          
          <Box sx={{ px: 2, mb: 2 }}>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: 'block', mb: 1 }}
            >
              Detect deepfakes across multiple media formats with state-of-the-art efficiency.
            </Typography>
            
            <Button
              variant="outlined"
              color="primary"
              fullWidth
              onClick={() => window.open('https://github.com/example/lightmultidetect', '_blank')}
              size="small"
            >
              About Project
            </Button>
          </Box>
        </Box>
      </Drawer>
      
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: '100%',
          minHeight: '100vh',
          backgroundColor: theme.palette.background.default,
          transition: theme.transitions.create('margin', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          ...(drawerOpen && !isSmallScreen && {
            marginLeft: drawerWidth,
            transition: theme.transitions.create('margin', {
              easing: theme.transitions.easing.easeOut,
              duration: theme.transitions.duration.enteringScreen,
            }),
          }),
        }}
      >
        <Toolbar />
        <Container maxWidth="xl">
          {children}
        </Container>
      </Box>
    </Box>
  );
};

export default Layout;


// frontend/src/components/UploadDropzone.js
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  LinearProgress,
  Paper,
  Divider,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Image as ImageIcon,
  AudioFile as AudioIcon,
  VideoFile as VideoIcon,
  InsertDriveFile as FileIcon,
} from '@mui/icons-material';

const UploadDropzone = ({ onFilesChange, maxFiles = 10, maxSize = 100 * 1024 * 1024 }) => {
  const [files, setFiles] = useState([]);
  const [rejectedFiles, setRejectedFiles] = useState([]);

  const onDrop = useCallback((acceptedFiles, rejected) => {
    // Process accepted files
    const newFiles = acceptedFiles.map(file => Object.assign(file, {
      preview: URL.createObjectURL(file),
      progress: 0,
    }));
    
    const updatedFiles = [...files, ...newFiles].slice(0, maxFiles);
    setFiles(updatedFiles);
    onFilesChange(updatedFiles);
    
    // Process rejected files
    setRejectedFiles(rejected);
  }, [files, maxFiles, onFilesChange]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles,
    maxSize,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.bmp'],
      'audio/*': ['.mp3', '.wav', '.ogg', '.flac'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv'],
    },
  });

  const removeFile = (index) => {
    const newFiles = [...files];
    URL.revokeObjectURL(newFiles[index].preview);
    newFiles.splice(index, 1);
    setFiles(newFiles);
    onFilesChange(newFiles);
  };

  const clearRejected = () => {
    setRejectedFiles([]);
  };

  const getFileIcon = (file) => {
    if (file.type.startsWith('image/')) return <ImageIcon color="primary" />;
    if (file.type.startsWith('audio/')) return <AudioIcon color="secondary" />;
    if (file.type.startsWith('video/')) return <VideoIcon color="error" />;
    return <FileIcon color="disabled" />;
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box sx={{ width: '100%', mb: 4 }}>
      <Paper
        {...getRootProps()}
        sx={{
          p: 3,
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'divider',
          borderRadius: 2,
          backgroundColor: isDragActive ? 'rgba(63, 81, 181, 0.08)' : 'background.paper',
          cursor: 'pointer',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            borderColor: 'primary.main',
            backgroundColor: 'rgba(63, 81, 181, 0.04)',
          },
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 200,
        }}
      >
        <input {...getInputProps()} />
        <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" align="center" gutterBottom>
          {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
        </Typography>
        <Typography variant="body2" align="center" color="textSecondary">
          or click to select files
        </Typography>
        <Typography variant="caption" align="center" color="textSecondary" sx={{ mt: 1 }}>
          Supported formats: Images (.jpg, .png), Audio (.mp3, .wav), Video (.mp4, .mov)
        </Typography>
        <Typography variant="caption" align="center" color="textSecondary">
          Max file size: {formatFileSize(maxSize)} • Max files: {maxFiles}
        </Typography>
      </Paper>

      {files.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Selected Files ({files.length}/{maxFiles})
          </Typography>
          <Paper variant="outlined" sx={{ borderRadius: 2 }}>
            <List>
              {files.map((file, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <Divider component="li" />}
                  <ListItem>
                    <ListItemIcon>
                      {getFileIcon(file)}
                    </ListItemIcon>
                    <ListItemText
                      primary={file.name}
                      secondary={`${formatFileSize(file.size)} • ${file.type}`}
                      primaryTypographyProps={{
                        noWrap: true,
                        style: { maxWidth: '70%' }
                      }}
                    />
                    <ListItemSecondaryAction>
                      <IconButton edge="end" onClick={() => removeFile(index)} size="small">
                        <DeleteIcon />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                  {file.progress > 0 && file.progress < 100 && (
                    <Box sx={{ px: 2, pb: 1, width: '100%' }}>
                      <LinearProgress variant="determinate" value={file.progress} />
                    </Box>
                  )}
                </React.Fragment>
              ))}
            </List>
          </Paper>
        </Box>
      )}

      {rejectedFiles.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle1" color="error" gutterBottom>
            Rejected Files ({rejectedFiles.length})
          </Typography>
          <Paper variant="outlined" sx={{ borderRadius: 2, borderColor: 'error.light' }}>
            <List dense>
              {rejectedFiles.map((rejection, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <Divider component="li" />}
                  <ListItem>
                    <ListItemIcon>
                      <FileIcon color="error" />
                    </ListItemIcon>
                    <ListItemText
                      primary={rejection.file.name}
                      secondary={
                        rejection.errors.map(e => e.message).join(', ')
                      }
                    />
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          </Paper>
          <Box sx={{ mt: 1, display: 'flex', justifyContent: 'flex-end' }}>
            <Typography
              variant="caption"
              color="primary"
              sx={{ cursor: 'pointer' }}
              onClick={clearRejected}
            >
              Clear rejected files
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default UploadDropzone;


// frontend/src/components/DetectionOptions.js
import React, { useState } from 'react';
import {
  Paper,
  Typography,
  FormControl,
  FormControlLabel,
  FormGroup,
  Checkbox,
  Slider,
  Switch,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  HelpOutline as HelpIcon,
} from '@mui/icons-material';

const DetectionOptions = ({ options, onChange }) => {
  const [expanded, setExpanded] = useState(true);
  
  const handleModalityChange = (event) => {
    const { name, checked } = event.target;
    onChange({
      ...options,
      modalities: {
        ...options.modalities,
        [name]: checked,
      },
    });
  };
  
  const handleConfidenceChange = (event, newValue) => {
    onChange({
      ...options,
      confidenceThreshold: newValue,
    });
  };
  
  const handleExplainChange = (event) => {
    onChange({
      ...options,
      explainResults: event.target.checked,
    });
  };
  
  return (
    <Paper sx={{ p: 3, borderRadius: 2, mb: 4 }}>
      <Typography variant="h6" gutterBottom>
        Detection Options
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Modalities
          <Tooltip title="Select which types of media to analyze">
            <IconButton size="small" sx={{ ml: 1 }}>
              <HelpIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>
        <FormGroup>
          <FormControlLabel
            control={
              <Checkbox
                checked={options.modalities.image}
                onChange={handleModalityChange}
                name="image"
              />
            }
            label="Image Analysis"
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={options.modalities.audio}
                onChange={handleModalityChange}
                name="audio"
              />
            }
            label="Audio Analysis"
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={options.modalities.video}
                onChange={handleModalityChange}
                name="video"
              />
            }
            label="Video Analysis"
          />
        </FormGroup>
      </Box>
      
      <Divider sx={{ my: 2 }} />
      
      <Accordion expanded={expanded} onChange={() => setExpanded(!expanded)} disableGutters elevation={0}>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          sx={{ px: 0 }}
        >
          <Typography variant="subtitle2">Advanced Options</Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0 }}>
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" gutterBottom>
              Confidence Threshold: {options.confidenceThreshold}
              <Tooltip title="Higher values reduce false positives but may miss some manipulations">
                <IconButton size="small" sx={{ ml: 1 }}>
                  <HelpIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Typography>
            <Slider
              value={options.confidenceThreshold}
              onChange={handleConfidenceChange}
              aria-labelledby="confidence-threshold-slider"
              step={0.05}
              marks
              min={0.5}
              max={0.95}
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
            label={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography variant="body2">Explain Results</Typography>
                <Tooltip title="Generate visual explanations for detected manipulations">
                  <IconButton size="small" sx={{ ml: 1 }}>
                    <HelpIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
        </AccordionDetails>
      </Accordion>
    </Paper>
  );
};

export default DetectionOptions;


// frontend/src/components/ResultCard.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Chip,
  LinearProgress,
  Box,
  Button,
  IconButton,
  Collapse,
  Divider,
  Grid,
  Avatar,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Pending as PendingIcon,
  Delete as DeleteIcon,
  Image as ImageIcon,
  AudioFile as AudioIcon,
  VideoFile as VideoIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const ExpandButton = styled((props) => {
  const { expand, ...other } = props;
  return <IconButton {...other} />;
})(({ theme, expand }) => ({
  transform: !expand ? 'rotate(0deg)' : 'rotate(180deg)',
  marginLeft: 'auto',
  transition: theme.transitions.create('transform', {
    duration: theme.transitions.duration.shortest,
  }),
}));

const ResultCard = ({ result, onDelete }) => {
  const navigate = useNavigate();
  const [expanded, setExpanded] = useState(false);
  
  const handleExpandClick = () => {
    setExpanded(!expanded);
  };
  
  const handleViewDetails = () => {
    navigate(`/results/${result.taskId}`);
  };
  
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'processing':
      case 'pending':
      default:
        return <PendingIcon color="warning" />;
    }
  };
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'processing':
        return 'warning';
      case 'pending':
      default:
        return 'default';
    }
  };
  
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };
  
  const getFileCount = () => {
    if (!result.numFiles) return '0 files';
    
    if (typeof result.numFiles === 'object') {
      const total = 
        (result.numFiles.image || 0) + 
        (result.numFiles.audio || 0) + 
        (result.numFiles.video || 0);
      return `${total} files`;
    }
    
    return `${result.numFiles} files`;
  };
  
  const renderFileTypeCounts = () => {
    if (!result.numFiles || typeof result.numFiles !== 'object') return null;
    
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
        {result.numFiles.image > 0 && (
          <Chip
            size="small"
            icon={<ImageIcon fontSize="small" />}
            label={result.numFiles.image}
            variant="outlined"
          />
        )}
        {result.numFiles.audio > 0 && (
          <Chip
            size="small"
            icon={<AudioIcon fontSize="small" />}
            label={result.numFiles.audio}
            variant="outlined"
          />
        )}
        {result.numFiles.video > 0 && (
          <Chip
            size="small"
            icon={<VideoIcon fontSize="small" />}
            label={result.numFiles.video}
            variant="outlined"
          />
        )}
      </Box>
    );
  };
  
  return (
    <Card
      sx={{
        mb: 2,
        borderRadius: 2,
        boxShadow: '0 4px 12px 0 rgba(0,0,0,0.1)',
        overflow: 'visible',
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 1 }}>
          <Avatar
            sx={{
              bgcolor: 
                result.status === 'completed' ? 'success.main' :
                result.status === 'failed' ? 'error.main' :
                'warning.main',
              width: 40,
              height: 40,
              mr: 2,
            }}
          >
            {getStatusIcon(result.status)}
          </Avatar>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" component="div" gutterBottom>
              Detection Task {result.taskId.substring(0, 8)}...
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
              <Chip
                label={result.status.toUpperCase()}
                color={getStatusColor(result.status)}
                size="small"
              />
              {result.modality && (
                <Chip
                  label={result.modality.toUpperCase()}
                  variant="outlined"
                  size="small"
                />
              )}
              <Chip
                label={getFileCount()}
                variant="outlined"
                size="small"
              />
            </Box>
            {renderFileTypeCounts()}
          </Box>
        </Box>
        
        {result.status === 'processing' && (
          <Box sx={{ width: '100%', mt: 2 }}>
            <LinearProgress />
          </Box>
        )}
      </CardContent>
      
      <Divider />
      
      <CardActions disableSpacing>
        <Button
          size="small"
          onClick={handleViewDetails}
          disabled={result.status === 'pending' || result.status === 'processing'}
        >
          View Details
        </Button>
        <Button
          size="small"
          color="error"
          startIcon={<DeleteIcon />}
          onClick={() => onDelete(result.taskId)}
        >
          Delete
        </Button>
        <ExpandButton
          expand={expanded}
          onClick={handleExpandClick}
          aria-expanded={expanded}
          aria-label="show more"
        >
          <ExpandMoreIcon />
        </ExpandButton>
      </CardActions>
      
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <Divider />
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="text.secondary">
                Created: {formatTimestamp(result.createdAt)}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="text.secondary">
                Completed: {formatTimestamp(result.completedAt)}
              </Typography>
            </Grid>
            {result.message && (
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  Message: {result.message}
                </Typography>
              </Grid>
            )}
            {result.status === 'completed' && result.results && (
              <Grid item xs={12}>
                <Typography variant="body2">
                  Results: {result.results.length} file(s) analyzed
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {result.results.filter(r => r.predicted_label === 'fake').length} detected as fake
                </Typography>
              </Grid>
            )}
          </Grid>
        </CardContent>
      </Collapse>
    </Card>
  );
};

export default ResultCard;


// frontend/src/components/AnalysisVisualizer.js
import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  Divider,
  Grid,
  Card,
  CardMedia,
  CardContent,
  Chip,
  LinearProgress,
} from '@mui/material';
import { styled } from '@mui/material/styles';

const StyledTab = styled(Tab)(({ theme }) => ({
  minWidth: 120,
  '&.Mui-selected': {
    backgroundColor: 'rgba(63, 81, 181, 0.08)',
    borderRadius: theme.shape.borderRadius,
  },
}));

const AnalysisVisualizer = ({ result }) => {
  const [selectedTab, setSelectedTab] = useState(0);
  
  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };
  
  if (!result || !result.results || result.results.length === 0) {
    return (
      <Paper sx={{ p: 3, borderRadius: 2 }}>
        <Typography variant="h6" gutterBottom>
          No analysis results available
        </Typography>
        <Typography variant="body2" color="text.secondary">
          There are no results to display for this detection task.
        </Typography>
      </Paper>
    );
  }
  
  const getBackgroundColor = (confidence, threshold = 0.7) => {
    if (confidence >= threshold) {
      return 'rgba(244, 67, 54, 0.05)'; // Red for high confidence fakes
    }
    return 'rgba(76, 175, 80, 0.05)'; // Green for low confidence or real
  };
  
  const renderImageResult = (item) => (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        borderRadius: 2,
        overflow: 'hidden',
        backgroundColor: item.predicted_label === 'fake' 
          ? getBackgroundColor(item.confidence) 
          : 'transparent',
      }}
    >
      {item.visualization_file ? (
        <CardMedia
          component="img"
          height="200"
          image={`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/detection/visualization/${result.task_id}/${item.visualization_file}`}
          alt={`Analysis of ${item.file_name}`}
        />
      ) : (
        <Box
          sx={{
            height: 200,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'background.paper',
          }}
        >
          <Typography variant="body2" color="text.secondary">
            No visualization available
          </Typography>
        </Box>
      )}
      <CardContent>
        <Typography variant="subtitle2" noWrap gutterBottom>
          {item.file_name}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Chip
            label={item.predicted_label.toUpperCase()}
            color={item.predicted_label === 'fake' ? 'error' : 'success'}
            size="small"
            sx={{ mr: 1 }}
          />
          <Typography variant="body2" color="text.secondary">
            {Math.round(item.confidence * 100)}% confidence
          </Typography>
        </Box>
        <Box sx={{ width: '100%', mt: 1 }}>
          <LinearProgress
            variant="determinate"
            value={item.confidence * 100}
            color={item.predicted_label === 'fake' ? 'error' : 'success'}
          />
        </Box>
        {item.explanation && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Key Features:
            </Typography>
            {item.explanation.frequency_analysis && (
              <Box sx={{ mt: 0.5 }}>
                <Typography variant="caption" display="block">
                  • Frequency patterns: {
                    item.explanation.frequency_analysis.high_freq_energy > 0.25 
                      ? 'Suspicious high frequency artifacts detected' 
                      : 'Normal frequency distribution'
                  }
                </Typography>
              </Box>
            )}
            {item.explanation.important_regions && (
              <Box sx={{ mt: 0.5 }}>
                <Typography variant="caption" display="block">
                  • Detected {item.explanation.important_regions.length} suspicious region(s)
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
  
  const renderAudioResult = (item) => (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        borderRadius: 2,
        overflow: 'hidden',
        backgroundColor: item.predicted_label === 'fake' 
          ? getBackgroundColor(item.confidence) 
          : 'transparent',
      }}
    >
      {item.visualization_file ? (
        <CardMedia
          component="img"
          height="200"
          image={`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/detection/visualization/${result.task_id}/${item.visualization_file}`}
          alt={`Analysis of ${item.file_name}`}
        />
      ) : (
        <Box
          sx={{
            height: 200,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'background.paper',
          }}
        >
          <Typography variant="body2" color="text.secondary">
            No visualization available
          </Typography>
        </Box>
      )}
      <CardContent>
        <Typography variant="subtitle2" noWrap gutterBottom>
          {item.file_name}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Chip
            label={item.predicted_label.toUpperCase()}
            color={item.predicted_label === 'fake' ? 'error' : 'success'}
            size="small"
            sx={{ mr: 1 }}
          />
          <Typography variant="body2" color="text.secondary">
            {Math.round(item.confidence * 100)}% confidence
          </Typography>
        </Box>
        <Box sx={{ width: '100%', mt: 1 }}>
          <LinearProgress
            variant="determinate"
            value={item.confidence * 100}
            color={item.predicted_label === 'fake' ? 'error' : 'success'}
          />
        </Box>
        {item.explanation && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Key Features:
            </Typography>
            {item.explanation.spectral_artifacts && (
              <Box sx={{ mt: 0.5 }}>
                <Typography variant="caption" display="block">
                  • Spectral artifacts: {
                    item.explanation.spectral_artifacts.presence 
                      ? `Detected (strength: ${Math.round(item.explanation.spectral_artifacts.strength * 100)}%)` 
                      : 'None detected'
                  }
                </Typography>
              </Box>
            )}
            {item.explanation.temporal_consistency !== undefined && (
              <Box sx={{ mt: 0.5 }}>
                <Typography variant="caption" display="block">
                  • Temporal consistency: {Math.round(item.explanation.temporal_consistency * 100)}%
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
  
  const renderVideoResult = (item) => (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        borderRadius: 2,
        overflow: 'hidden',
        backgroundColor: item.predicted_label === 'fake' 
          ? getBackgroundColor(item.confidence) 
          : 'transparent',
      }}
    >
      {item.visualization_file ? (
        <CardMedia
          component="img"
          height="200"
          image={`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/detection/visualization/${result.task_id}/${item.visualization_file}`}
          alt={`Analysis of ${item.file_name}`}
        />
      ) : (
        <Box
          sx={{
            height: 200,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'background.paper',
          }}
        >
          <Typography variant="body2" color="text.secondary">
            No visualization available
          </Typography>
        </Box>
      )}
      <CardContent>
        <Typography variant="subtitle2" noWrap gutterBottom>
          {item.file_name}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Chip
            label={item.predicted_label.toUpperCase()}
            color={item.predicted_label === 'fake' ? 'error' : 'success'}
            size="small"
            sx={{ mr: 1 }}
          />
          <Typography variant="body2" color="text.secondary">
            {Math.round(item.confidence * 100)}% confidence
          </Typography>
        </Box>
        <Box sx={{ width: '100%', mt: 1 }}>
          <LinearProgress
            variant="determinate"
            value={item.confidence * 100}
            color={item.predicted_label === 'fake' ? 'error' : 'success'}
          />
        </Box>
        {item.explanation && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Key Features:
            </Typography>
            {item.explanation.facial_landmarks && (
              <Box sx={{ mt: 0.5 }}>
                <Typography variant="caption" display="block">
                  • Facial landmarks: {
                    item.explanation.facial_landmarks.abnormalities 
                      ? 'Abnormalities detected' 
                      : 'No abnormalities'
                  } (stability: {Math.round(item.explanation.facial_landmarks.stability * 100)}%)
                </Typography>
              </Box>
            )}
            {item.explanation.audio_visual_sync !== undefined && (
              <Box sx={{ mt: 0.5 }}>
                <Typography variant="caption" display="block">
                  • Audio-visual sync: {Math.round(item.explanation.audio_visual_sync * 100)}%
                </Typography>
              </Box>
            )}
            {item.explanation.important_frames && (
              <Box sx={{ mt: 0.5 }}>
                <Typography variant="caption" display="block">
                  • Detected {item.explanation.important_frames.length} suspicious frame(s)
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
  
  // Filter results by modality
  const imageResults = result.results.filter(item => item.modality === 'image');
  const audioResults = result.results.filter(item => item.modality === 'audio');
  const videoResults = result.results.filter(item => item.modality === 'video');
  
  // Determine which tabs to show
  const tabs = [];
  if (imageResults.length > 0) tabs.push('Images');
  if (audioResults.length > 0) tabs.push('Audio');
  if (videoResults.length > 0) tabs.push('Videos');
  if (tabs.length === 0) tabs.push('All');
  
  return (
    <Paper sx={{ p: 3, borderRadius: 2 }}>
      <Typography variant="h6" gutterBottom>
        Analysis Results
      </Typography>
      
      <Tabs
        value={selectedTab}
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ mb: 2 }}
      >
        {tabs.map((tab, index) => (
          <StyledTab key={tab} label={`${tab} (${
            tab === 'Images' ? imageResults.length :
            tab === 'Audio' ? audioResults.length :
            tab === 'Videos' ? videoResults.length :
            result.results.length
          })`} />
        ))}
      </Tabs>
      
      <Divider sx={{ mb: 3 }} />
      
      {selectedTab === tabs.indexOf('Images') && (
        <Grid container spacing={2}>
          {imageResults.map((item, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              {renderImageResult(item)}
            </Grid>
          ))}
        </Grid>
      )}
      
      {selectedTab === tabs.indexOf('Audio') && (
        <Grid container spacing={2}>
          {audioResults.map((item, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              {renderAudioResult(item)}
            </Grid>
          ))}
        </Grid>
      )}
      
      {selectedTab === tabs.indexOf('Videos') && (
        <Grid container spacing={2}>
          {videoResults.map((item, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              {renderVideoResult(item)}
            </Grid>
          ))}
        </Grid>
      )}
      
      {selectedTab === tabs.indexOf('All') && (
        <Grid container spacing={2}>
          {result.results.map((item, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              {item.modality === 'image' && renderImageResult(item)}
              {item.modality === 'audio' && renderAudioResult(item)}
              {item.modality === 'video' && renderVideoResult(item)}
            </Grid>
          ))}
        </Grid>
      )}
    </Paper>
  );
};

export default AnalysisVisualizer;


// frontend/src/pages/UploadPage.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Paper,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  CircularProgress,
  Divider,
} from '@mui/material';
import { Upload as UploadIcon, Check as CheckIcon } from '@mui/icons-material';

import UploadDropzone from '../components/UploadDropzone';
import DetectionOptions from '../components/DetectionOptions';
import api from '../services/api';

const UploadPage = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [files, setFiles] = useState([]);
  const [options, setOptions] = useState({
    modalities: {
      image: true,
      audio: true,
      video: true,
    },
    confidenceThreshold: 0.7,
    explainResults: true,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [taskId, setTaskId] = useState(null);
  
  const steps = [
    {
      label: 'Upload Media Files',
      description: 'Drag and drop or select image, audio, or video files to analyze.',
    },
    {
      label: 'Configure Detection Settings',
      description: 'Adjust detection modalities and parameters.',
    },
    {
      label: 'Process and Analyze',
      description: 'Submit files for deepfake detection analysis.',
    },
  ];
  
  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };
  
  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
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
      
      // Prepare modalities string
      let modalitiesString = Object.entries(options.modalities)
        .filter(([_, enabled]) => enabled)
        .map(([modality]) => modality)
        .join(',');
      
      if (!modalitiesString) {
        modalitiesString = 'all';
      }
      
      // Upload files
      const response = await api.uploadMedia(files, {
        modalities: modalitiesString,
        confidence_threshold: options.confidenceThreshold,
        explain_results: options.explainResults,
        onProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        },
      });
      
      // Set task ID
      setTaskId(response.data.task_id);
      
      // Move to next step
      handleNext();
      
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
      console.error('Upload error:', err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleViewResults = () => {
    // Navigate to results page if task ID is available
    if (taskId) {
      navigate(`/results/${taskId}`);
    } else {
      navigate('/results');
    }
  };
  
  const isStepComplete = (step) => {
    if (step === 0) {
      return files.length > 0;
    }
    if (step === 1) {
      return Object.values(options.modalities).some(Boolean);
    }
    return false;
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Upload Media for Analysis
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Upload images, audio, or video files to detect potential deepfakes using our advanced multi-modal analysis.
      </Typography>
      
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
        <Box sx={{ flex: 2 }}>
          <Stepper activeStep={activeStep} orientation="vertical">
            {steps.map((step, index) => (
              <Step key={step.label} completed={isStepComplete(index)}>
                <StepLabel>{step.label}</StepLabel>
                <StepContent>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {step.description}
                  </Typography>
                  
                  {index === 0 && (
                    <UploadDropzone
                      onFilesChange={handleFilesChange}
                      maxFiles={10}
                      maxSize={100 * 1024 * 1024} // 100 MB
                    />
                  )}
                  
                  {index === 1 && (
                    <DetectionOptions
                      options={options}
                      onChange={handleOptionsChange}
                    />
                  )}
                  
                  {index === 2 && (
                    <Paper sx={{ p: 3, borderRadius: 2, mb: 3 }}>
                      <Typography variant="h6" gutterBottom>
                        Detection Summary
                      </Typography>
                      <Typography variant="body2">
                        Files to analyze: {files.length}
                      </Typography>
                      <Typography variant="body2">
                        Selected modalities: {
                          Object.entries(options.modalities)
                            .filter(([_, enabled]) => enabled)
                            .map(([modality]) => modality.charAt(0).toUpperCase() + modality.slice(1))
                            .join(', ')
                        }
                      </Typography>
                      <Typography variant="body2">
                        Confidence threshold: {options.confidenceThreshold}
                      </Typography>
                      <Typography variant="body2">
                        Generate explanations: {options.explainResults ? 'Yes' : 'No'}
                      </Typography>
                    </Paper>
                  )}
                  
                  {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                      {error}
                    </Alert>
                  )}
                  
                  <Box sx={{ mb: 2, mt: 2 }}>
                    <Button
                      disabled={loading || !isStepComplete(index)}
                      onClick={index === steps.length - 1 ? handleUpload : handleNext}
                      variant="contained"
                      sx={{ mr: 1 }}
                      startIcon={index === steps.length - 1 ? 
                        (loading ? <CircularProgress size={20} color="inherit" /> : <UploadIcon />) : 
                        null}
                    >
                      {index === steps.length - 1 ? 'Start Analysis' : 'Continue'}
                    </Button>
                    <Button
                      disabled={index === 0 || loading}
                      onClick={handleBack}
                      sx={{ mt: { xs: 1, sm: 0 } }}
                    >
                      Back
                    </Button>
                  </Box>
                </StepContent>
              </Step>
            ))}
            
            {/* Final step after upload */}
            {activeStep === steps.length && (
              <Paper sx={{ p: 3, mt: 3, borderRadius: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <CheckIcon color="success" sx={{ mr: 1 }} />
                  <Typography variant="h6">Analysis Started</Typography>
                </Box>
                <Typography variant="body1" paragraph>
                  Your files have been successfully uploaded and the detection process has started.
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Typography variant="body2" paragraph>
                  Task ID: {taskId}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  The analysis may take a few minutes depending on the number and size of your files.
                  You can check the results page to view the status and results of your detection task.
                </Typography>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleViewResults}
                  sx={{ mr: 1 }}
                >
                  View Results
                </Button>
                <Button
                  onClick={() => {
                    setActiveStep(0);
                    setFiles([]);
                    setTaskId(null);
                    setUploadProgress(0);
                    setError(null);
                  }}
                >
                  Upload More Files
                </Button>
              </Paper>
            )}
          </Stepper>
        </Box>
        
        <Box sx={{ flex: 1 }}>
          <Paper sx={{ p: 3, borderRadius: 2, position: 'sticky', top: 100 }}>
            <Typography variant="h6" gutterBottom>
              About Deepfake Detection
            </Typography>
            <Typography variant="body2" paragraph>
              Our lightweight multi-modal detection system analyzes media across different dimensions:
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Image Analysis
            </Typography>
            <Typography variant="body2" paragraph>
              Detects face swaps, warping artifacts, and GAN-generated faces using frequency domain analysis
              and spatial inconsistencies.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Audio Analysis
            </Typography>
            <Typography variant="body2" paragraph>
              Identifies synthetic speech and voice cloning by detecting spectral artifacts and
              temporal inconsistencies in vocal patterns.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Video Analysis
            </Typography>
            <Typography variant="body2" paragraph>
              Discovers facial reenactments, lip-sync inconsistencies, and temporal artifacts
              through frame-by-frame analysis and audio-visual correlation.
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="caption" color="text.secondary">
              LightMultiDetect is designed for efficiency with minimal performance trade-offs,
              running 10-50x faster than comparable state-of-the-art models.
            </Typography>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default UploadPage;


// frontend/src/pages/ResultsPage.js
import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  InputBase,
  IconButton,
  Divider,
  Grid,
  CircularProgress,
  Alert,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from '@mui/material';
import { Search as SearchIcon, Add as AddIcon, Delete as DeleteIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

import ResultCard from '../components/ResultCard';
import api from '../services/api';

const ResultsPage = () => {
  const navigate = useNavigate();
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteTaskId, setDeleteTaskId] = useState(null);
  
  useEffect(() => {
    fetchResults();
  }, []);
  
  const fetchResults = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // In a real app, this would be an API call to fetch the user's results
      // For demo purposes, we'll create some mock data
      const mockResults = [
        {
          taskId: '9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d',
          status: 'completed',
          modality: 'multimodal',
          numFiles: {
            image: 2,
            audio: 1,
            video: 1,
            total: 4
          },
          results: [
            { predicted_label: 'fake', confidence: 0.92 },
            { predicted_label: 'real', confidence: 0.89 },
            { predicted_label: 'fake', confidence: 0.76 },
            { predicted_label: 'real', confidence: 0.95 }
          ],
          createdAt: Date.now() / 1000 - 3600,
          completedAt: Date.now() / 1000 - 3500
        },
        {
          taskId: '1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed',
          status: 'processing',
          modality: 'image',
          numFiles: 3,
          createdAt: Date.now() / 1000 - 600
        },
        {
          taskId: '6ec0bd7f-11c0-43da-975e-2a8ad9ebae0b',
          status: 'failed',
          modality: 'video',
          numFiles: 1,
          message: 'Failed to process video file: format not supported',
          createdAt: Date.now() / 1000 - 86400
        }
      ];
      
      setResults(mockResults);
    } catch (err) {
      setError('Failed to load results. Please try again.');
      console.error('Error fetching results:', err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSearch = (e) => {
    e.preventDefault();
    // Filter results based on search query
    // This would typically be an API call with search parameters
    console.log('Searching for:', searchQuery);
  };
  
  const openDeleteDialog = (taskId) => {
    setDeleteTaskId(taskId);
    setDeleteDialogOpen(true);
  };
  
  const handleDelete = async () => {
    try {
      // In a real app, this would be an API call to delete the result
      console.log('Deleting task:', deleteTaskId);
      
      // Remove from local state
      setResults(results.filter(result => result.taskId !== deleteTaskId));
      
      // Close dialog
      setDeleteDialogOpen(false);
      setDeleteTaskId(null);
    } catch (err) {
      setError('Failed to delete result. Please try again.');
      console.error('Error deleting result:', err);
    }
  };
  
  const navigateToUpload = () => {
    navigate('/upload');
  };
  
  // Filter results based on search query
  const filteredResults = results.filter(result => 
    result.taskId.includes(searchQuery) ||
    result.status.includes(searchQuery.toLowerCase()) ||
    (result.modality && result.modality.includes(searchQuery.toLowerCase()))
  );
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Detection Results</Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={navigateToUpload}
        >
          New Analysis
        </Button>
      </Box>
      
      <Paper
        component="form"
        sx={{
          p: 1,
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          mb: 3,
          borderRadius: 2,
        }}
        onSubmit={handleSearch}
      >
        <InputBase
          sx={{ ml: 1, flex: 1 }}
          placeholder="Search by task ID or status"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <Divider sx={{ height: 28, m: 0.5 }} orientation="vertical" />
        <IconButton type="submit" sx={{ p: 1 }}>
          <SearchIcon />
        </IconButton>
      </Paper>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      ) : filteredResults.length === 0 ? (
        <Paper sx={{ p: 3, borderRadius: 2, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>
            No results found
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            {results.length === 0 ? 
              "You haven't submitted any detection tasks yet." : 
              "No results match your search criteria."}
          </Typography>
          {results.length === 0 && (
            <Button
              variant="contained"
              color="primary"
              startIcon={<AddIcon />}
              onClick={navigateToUpload}
            >
              Start New Analysis
            </Button>
          )}
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {filteredResults.map((result) => (
            <Grid item xs={12} key={result.taskId}>
              <ResultCard
                result={result}
                onDelete={openDeleteDialog}
              />
            </Grid>
          ))}
        </Grid>
      )}
      
      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this detection task and all its results? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error" startIcon={<DeleteIcon />}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ResultsPage;


// frontend/src/pages/ResultDetailPage.js
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Grid,
  CircularProgress,
  Alert,
  Button,
  Divider,
  LinearProgress,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Pending as PendingIcon,
  Image as ImageIcon,
  AudioFile as AudioIcon,
  VideoFile as VideoIcon,
} from '@mui/icons-material';

import AnalysisVisualizer from '../components/AnalysisVisualizer';
import api from '../services/api';

const ResultDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [polling, setPolling] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  
  useEffect(() => {
    fetchResult();
  }, [id]);
  
  useEffect(() => {
    // Set up polling for non-completed tasks
    let interval;
    
    if (result && (result.status === 'pending' || result.status === 'processing')) {
      interval = setInterval(() => {
        setPolling(true);
        fetchResult(false);
      }, 5000); // Poll every 5 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [result]);
  
  const fetchResult = async (showLoading = true) => {
    try {
      if (showLoading) {
        setLoading(true);
      }
      setError(null);
      
      // In a real app, this would be an API call to fetch the result
      // For demo purposes, we'll create some mock data
      const mockResult = {
        task_id: id,
        status: 'completed',
        modality: 'multimodal',
        num_files: {
          image: 2,
          audio: 1,
          video: 1,
          total: 4
        },
        results: [
          {
            file_name: 'image1.jpg',
            predicted_label: 'fake',
            confidence: 0.92,
            modality: 'image',
            visualization_file: 'image1_viz.png',
            processing_time: 0.45,
            explanation: {
              important_regions: [
                {
                  x: 120,
                  y: 80,
                  width: 50,
                  height: 30,
                  importance: 0.85
                }
              ],
              frequency_analysis: {
                low_freq_energy: 0.3,
                mid_freq_energy: 0.4,
                high_freq_energy: 0.3
              }
            }
          },
          {
            file_name: 'image2.jpg',
            predicted_label: 'real',
            confidence: 0.89,
            modality: 'image',
            visualization_file: 'image2_viz.png',
            processing_time: 0.38,
            explanation: {
              important_regions: [],
              frequency_analysis: {
                low_freq_energy: 0.4,
                mid_freq_energy: 0.45,
                high_freq_energy: 0.15
              }
            }
          },
          {
            file_name: 'audio1.mp3',
            predicted_label: 'fake',
            confidence: 0.76,
            modality: 'audio',
            visualization_file: 'audio1_viz.png',
            processing_time: 0.82,
            explanation: {
              temporal_consistency: 0.65,
              spectral_artifacts: {
                presence: true,
                strength: 0.7
              },
              frequency_bands: {
                low: 0.3,
                mid: 0.5,
                high: 0.2
              }
            }
          },
          {
            file_name: 'video1.mp4',
            predicted_label: 'real',
            confidence: 0.95,
            modality: 'video',
            visualization_file: 'video1_viz.png',
            processing_time: 2.45,
            explanation: {
              temporal_consistency: 0.9,
              facial_landmarks: {
                stability: 0.92,
                abnormalities: false
              },
              audio_visual_sync: 0.88,
              important_frames: [
                {
                  frame_idx: 45,
                  confidence: 0.87
                },
                {
                  frame_idx: 127,
                  confidence: 0.85
                }
              ]
            }
          }
        ],
        created_at: Date.now() / 1000 - 3600,
        completed_at: Date.now() / 1000 - 3500
      };
      
      setResult(mockResult);
    } catch (err) {
      setError('Failed to load result. Please try again.');
      console.error('Error fetching result:', err);
    } finally {
      setLoading(false);
      setPolling(false);
    }
  };
  
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'processing':
      case 'pending':
      default:
        return <PendingIcon color="warning" />;
    }
  };
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'processing':
        return 'warning';
      case 'pending':
      default:
        return 'default';
    }
  };
  
  const handleDelete = async () => {
    try {
      // In a real app, this would be an API call to delete the result
      console.log('Deleting task:', id);
      
      // Navigate back to results page
      navigate('/results');
    } catch (err) {
      setError('Failed to delete result. Please try again.');
      console.error('Error deleting result:', err);
    } finally {
      setDeleteDialogOpen(false);
    }
  };
  
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };
  
  const calculateStats = () => {
    if (!result || !result.results || result.results.length === 0) {
      return {
        totalFiles: 0,
        fakeCount: 0,
        realCount: 0,
        fakePercentage: 0,
        avgConfidence: 0,
        avgProcessingTime: 0
      };
    }
    
    const totalFiles = result.results.length;
    const fakeCount = result.results.filter(r => r.predicted_label === 'fake').length;
    const realCount = totalFiles - fakeCount;
    const fakePercentage = (fakeCount / totalFiles) * 100;
    const avgConfidence = result.results.reduce((acc, r) => acc + r.confidence, 0) / totalFiles;
    const avgProcessingTime = result.results.reduce((acc, r) => acc + r.processing_time, 0) / totalFiles;
    
    return {
      totalFiles,
      fakeCount,
      realCount,
      fakePercentage,
      avgConfidence,
      avgProcessingTime
    };
  };
  
  const stats = calculateStats();
  
  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <IconButton onClick={() => navigate('/results')} sx={{ mr: 1 }}>
          <ArrowBackIcon />
        </IconButton>
        <Typography variant="h4">Detection Result Details</Typography>
      </Box>
      
      {loading && !polling ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      ) : result ? (
        <Box>
          <Paper sx={{ p: 3, borderRadius: 2, mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Box>
                <Typography variant="h6" gutterBottom>
                  Task Information
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                  <Chip
                    icon={getStatusIcon(result.status)}
                    label={result.status.toUpperCase()}
                    color={getStatusColor(result.status)}
                  />
                  {result.modality && (
                    <Chip
                      label={result.modality.toUpperCase()}
                      variant="outlined"
                    />
                  )}
                  <Chip
                    label={`${stats.totalFiles} files`}
                    variant="outlined"
                  />
                </Box>
              </Box>
              <Box>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={() => setDeleteDialogOpen(true)}
                  sx={{ ml: 1 }}
                >
                  Delete
                </Button>
                {(result.status === 'pending' || result.status === 'processing') && (
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={() => fetchResult()}
                    sx={{ ml: 1 }}
                    disabled={polling}
                  >
                    Refresh
                  </Button>
                )}
              </Box>
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            {result.status === 'processing' && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="body2" gutterBottom>
                  Processing in progress...
                </Typography>
                <LinearProgress sx={{ mt: 1 }} />
              </Box>
            )}
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="body2" color="text.secondary">
                  Task ID
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {result.task_id}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="body2" color="text.secondary">
                  Created
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {formatTimestamp(result.created_at)}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="body2" color="text.secondary">
                  Completed
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {formatTimestamp(result.completed_at)}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="body2" color="text.secondary">
                  Processing Time
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {result.completed_at && result.created_at
                    ? `${Math.round((result.completed_at - result.created_at) * 100) / 100} seconds`
                    : 'N/A'}
                </Typography>
              </Grid>
            </Grid>
            
            {result.num_files && typeof result.num_files === 'object' && (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 2 }}>
                {result.num_files.image > 0 && (
                  <Chip
                    icon={<ImageIcon />}
                    label={`${result.num_files.image} images`}
                    variant="outlined"
                    size="small"
                  />
                )}
                {result.num_files.audio > 0 && (
                  <Chip
                    icon={<AudioIcon />}
                    label={`${result.num_files.audio} audio files`}
                    variant="outlined"
                    size="small"
                  />
                )}
                {result.num_files.video > 0 && (
                  <Chip
                    icon={<VideoIcon />}
                    label={`${result.num_files.video} videos`}
                    variant="outlined"
                    size="small"
                  />
                )}
              </Box>
            )}
          </Paper>
          
          {result.status === 'completed' && result.results && (
            <Box>
              <Paper sx={{ p: 3, borderRadius: 2, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Detection Summary
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={6} md={3}>
                    <Paper
                      sx={{
                        p: 2,
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: 'background.default',
                        borderRadius: 2,
                      }}
                    >
                      <Typography variant="h3" color="primary">
                        {stats.totalFiles}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Files Analyzed
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} sm={6} md={3}>
                    <Paper
                      sx={{
                        p: 2,
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: 'error.main',
                        color: 'error.contrastText',
                        borderRadius: 2,
                      }}
                    >
                      <Typography variant="h3">
                        {stats.fakeCount}
                      </Typography>
                      <Typography variant="body2">
                        Detected as Fake
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} sm={6} md={3}>
                    <Paper
                      sx={{
                        p: 2,
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: 'success.main',
                        color: 'success.contrastText',
                        borderRadius: 2,
                      }}
                    >
                      <Typography variant="h3">
                        {stats.realCount}
                      </Typography>
                      <Typography variant="body2">
                        Detected as Real
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} sm={6} md={3}>
                    <Paper
                      sx={{
                        p: 2,
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: 'background.default',
                        borderRadius: 2,
                      }}
                    >
                      <Typography variant="h3" color="text.primary">
                        {Math.round(stats.avgConfidence * 100)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Average Confidence
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
                
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Fake Content Distribution
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <Box sx={{ flex: 1, mr: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={stats.fakePercentage}
                        color="error"
                        sx={{ height: 10, borderRadius: 5 }}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {Math.round(stats.fakePercentage)}%
                    </Typography>
                  </Box>
                </Box>
                
                <Box sx={{ mt: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    Average processing time per file: {stats.avgProcessingTime.toFixed(2)} seconds
                  </Typography>
                </Box>
              </Paper>
              
              <AnalysisVisualizer result={result} />
            </Box>
          )}
          
          {result.status === 'failed' && (
            <Paper sx={{ p: 3, borderRadius: 2, bgcolor: 'error.main', color: 'error.contrastText' }}>
              <Typography variant="h6" gutterBottom>
                Processing Failed
              </Typography>
              <Typography variant="body1">
                {result.message || 'An unknown error occurred during processing.'}
              </Typography>
            </Paper>
          )}
        </Box>
      ) : (
        <Alert severity="error" sx={{ mb: 3 }}>
          Result not found or has been deleted.
        </Alert>
      )}
      
      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this detection task and all its results? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error" startIcon={<DeleteIcon />}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ResultDetailPage;


// frontend/src/pages/LoginPage.js
import React, { useState } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Link,
  Grid,
  CircularProgress,
  Alert,
  Divider,
} from '@mui/material';
import { Lock as LockIcon } from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';

const LoginPage = () => {
  const navigate = useNavigate();
  const { isAuthenticated, login, error: authError } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Redirect if already authenticated
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, navigate]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate inputs
    if (!username || !password) {
      setError('Please enter both username and password.');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      // Attempt login
      await login(username, password);
      
      // Redirect to dashboard on successful login
      navigate('/');
    // frontend/src/pages/LoginPage.js (continued)
} catch (err) {
    console.error('Login error:', err);
    setError(err.message || 'Failed to login. Please check your credentials.');
  } finally {
    setLoading(false);
  }
};

return (
  <Box
    sx={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #121212 0%, #2c2c2c 100%)',
      p: 2,
    }}
  >
    <Paper
      sx={{
        p: 4,
        maxWidth: 500,
        width: '100%',
        borderRadius: 3,
        boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
        position: 'relative',
        overflow: 'hidden',
        '&:before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '4px',
          background: 'linear-gradient(90deg, #3f51b5, #f50057)',
        },
      }}
      elevation={8}
    >
      <Box sx={{ mb: 3, textAlign: 'center' }}>
        <LockIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
        <Typography variant="h4" component="h1" gutterBottom>
          LightMultiDetect
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Sign in to access the deepfake detection platform
        </Typography>
      </Box>
      
      {(error || authError) && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error || authError}
        </Alert>
      )}
      
      <form onSubmit={handleSubmit}>
        <TextField
          label="Username"
          variant="outlined"
          fullWidth
          margin="normal"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          disabled={loading}
          required
        />
        <TextField
          label="Password"
          type="password"
          variant="outlined"
          fullWidth
          margin="normal"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={loading}
          required
        />
        <Button
          type="submit"
          fullWidth
          variant="contained"
          color="primary"
          size="large"
          disabled={loading}
          sx={{ mt: 3, mb: 2, py: 1.2 }}
        >
          {loading ? <CircularProgress size={24} color="inherit" /> : 'Sign In'}
        </Button>
      </form>
      
      <Divider sx={{ my: 2 }}>
        <Typography variant="caption" color="text.secondary">
          OR
        </Typography>
      </Divider>
      
      <Box sx={{ textAlign: 'center' }}>
        <Typography variant="body2">
          Don't have an account?{' '}
          <Link component={RouterLink} to="/register" underline="hover">
            Sign up
          </Link>
        </Typography>
      </Box>
      
      <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
        <Typography variant="caption" align="center" color="text.secondary" display="block">
          For demo purposes, use: username: "admin" / password: "admin"
        </Typography>
      </Box>
    </Paper>
    
    {/* Background grid pattern */}
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: 'radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px)',
        backgroundSize: '30px 30px',
        pointerEvents: 'none',
        zIndex: 0,
      }}
    />
  </Box>
);
};

export default LoginPage;


// frontend/src/pages/RegisterPage.js
import React, { useState } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
Box,
Typography,
TextField,
Button,
Paper,
Link,
Grid,
CircularProgress,
Alert,
Divider,
} from '@mui/material';
import { PersonAdd as PersonAddIcon } from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';

const RegisterPage = () => {
const navigate = useNavigate();
const { isAuthenticated, register, error: authError } = useAuth();
const [username, setUsername] = useState('');
const [email, setEmail] = useState('');
const [password, setPassword] = useState('');
const [confirmPassword, setConfirmPassword] = useState('');
const [loading, setLoading] = useState(false);
const [error, setError] = useState(null);

// Redirect if already authenticated
React.useEffect(() => {
  if (isAuthenticated) {
    navigate('/');
  }
}, [isAuthenticated, navigate]);

const validateEmail = (email) => {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
};

const handleSubmit = async (e) => {
  e.preventDefault();
  
  // Validate inputs
  if (!username || !email || !password || !confirmPassword) {
    setError('Please fill in all fields.');
    return;
  }
  
  if (!validateEmail(email)) {
    setError('Please enter a valid email address.');
    return;
  }
  
  if (password !== confirmPassword) {
    setError('Passwords do not match.');
    return;
  }
  
  if (password.length < 6) {
    setError('Password must be at least 6 characters long.');
    return;
  }
  
  try {
    setLoading(true);
    setError(null);
    
    // Attempt registration
    await register(username, email, password);
    
    // Redirect to dashboard on successful registration
    navigate('/');
  } catch (err) {
    console.error('Registration error:', err);
    setError(err.message || 'Failed to register. Please try again.');
  } finally {
    setLoading(false);
  }
};

return (
  <Box
    sx={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #121212 0%, #2c2c2c 100%)',
      p: 2,
    }}
  >
    <Paper
      sx={{
        p: 4,
        maxWidth: 500,
        width: '100%',
        borderRadius: 3,
        boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
        position: 'relative',
        overflow: 'hidden',
        '&:before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '4px',
          background: 'linear-gradient(90deg, #3f51b5, #f50057)',
        },
      }}
      elevation={8}
    >
      <Box sx={{ mb: 3, textAlign: 'center' }}>
        <PersonAddIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
        <Typography variant="h4" component="h1" gutterBottom>
          Create Account
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Sign up to use the deepfake detection platform
        </Typography>
      </Box>
      
      {(error || authError) && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error || authError}
        </Alert>
      )}
      
      <form onSubmit={handleSubmit}>
        <TextField
          label="Username"
          variant="outlined"
          fullWidth
          margin="normal"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          disabled={loading}
          required
        />
        <TextField
          label="Email"
          type="email"
          variant="outlined"
          fullWidth
          margin="normal"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={loading}
          required
        />
        <TextField
          label="Password"
          type="password"
          variant="outlined"
          fullWidth
          margin="normal"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={loading}
          required
        />
        <TextField
          label="Confirm Password"
          type="password"
          variant="outlined"
          fullWidth
          margin="normal"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          disabled={loading}
          required
        />
        <Button
          type="submit"
          fullWidth
          variant="contained"
          color="primary"
          size="large"
          disabled={loading}
          sx={{ mt: 3, mb: 2, py: 1.2 }}
        >
          {loading ? <CircularProgress size={24} color="inherit" /> : 'Sign Up'}
        </Button>
      </form>
      
      <Divider sx={{ my: 2 }}>
        <Typography variant="caption" color="text.secondary">
          OR
        </Typography>
      </Divider>
      
      <Box sx={{ textAlign: 'center' }}>
        <Typography variant="body2">
          Already have an account?{' '}
          <Link component={RouterLink} to="/login" underline="hover">
            Sign in
          </Link>
        </Typography>
      </Box>
    </Paper>
    
    {/* Background grid pattern */}
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: 'radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px)',
        backgroundSize: '30px 30px',
        pointerEvents: 'none',
        zIndex: 0,
      }}
    />
  </Box>
);
};

export default RegisterPage;


// frontend/src/pages/Dashboard.js
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
Box,
Typography,
Grid,
Paper,
Button,
Divider,
Card,
CardContent,
LinearProgress,
CircularProgress,
IconButton,
Alert,
} from '@mui/material';
import {
CloudUpload as CloudUploadIcon,
Assessment as AssessmentIcon,
Speed as SpeedIcon,
Memory as MemoryIcon,
Shield as ShieldIcon,
Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';

// Chart components (using recharts)
import {
LineChart, Line, AreaChart, Area, BarChart, Bar,
PieChart, Pie, Cell, 
XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const Dashboard = () => {
const theme = useTheme();
const navigate = useNavigate();
const { user } = useAuth();
const [recentResults, setRecentResults] = useState([]);
const [stats, setStats] = useState({
  total: 0,
  pending: 0,
  processing: 0,
  completed: 0,
  failed: 0,
  fakeDetected: 0,
  realDetected: 0,
});
const [loading, setLoading] = useState(true);
const [statsLoading, setStatsLoading] = useState(true);
const [error, setError] = useState(null);

// Mock data for charts
const detectionHistory = [
  { date: '4/3', fake: 5, real: 12 },
  { date: '4/4', fake: 8, real: 10 },
  { date: '4/5', fake: 12, real: 7 },
  { date: '4/6', fake: 10, real: 14 },
  { date: '4/7', fake: 15, real: 8 },
  { date: '4/8', fake: 13, real: 9 },
  { date: '4/9', fake: 18, real: 6 },
];

const detectionByType = [
  { name: 'Image', count: 35 },
  { name: 'Audio', count: 12 },
  { name: 'Video', count: 23 },
];

const performanceMetrics = [
  { name: 'Accuracy', value: 96 },
  { name: 'Recall', value: 94 },
  { name: 'Precision', value: 92 },
  { name: 'F1 Score', value: 93 },
];

const COLORS = [theme.palette.primary.main, theme.palette.secondary.main, theme.palette.error.main, theme.palette.success.main];

useEffect(() => {
  fetchData();
}, []);

const fetchData = async () => {
  try {
    setLoading(true);
    setError(null);
    
    // In a real app, this would be API calls to fetch data
    // For this demo, we'll simulate with mock data
    
    // Mock recent results
    const mockResults = [
      {
        taskId: '9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d',
        status: 'completed',
        modality: 'multimodal',
        numFiles: 4,
        createdAt: Date.now() / 1000 - 3600,
        completedAt: Date.now() / 1000 - 3500
      },
      {
        taskId: '1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed',
        status: 'processing',
        modality: 'image',
        numFiles: 3,
        createdAt: Date.now() / 1000 - 600
      },
      {
        taskId: '6ec0bd7f-11c0-43da-975e-2a8ad9ebae0b',
        status: 'completed',
        modality: 'video',
        numFiles: 1,
        createdAt: Date.now() / 1000 - 86400,
        completedAt: Date.now() / 1000 - 86300
      }
    ];
    
    setRecentResults(mockResults);
    
    // Mock statistics
    const mockStats = {
      total: 70,
      pending: 2,
      processing: 3,
      completed: 60,
      failed: 5,
      fakeDetected: 28,
      realDetected: 32,
    };
    
    setStats(mockStats);
    
  } catch (err) {
    console.error('Error fetching dashboard data:', err);
    setError('Failed to load dashboard data.');
  } finally {
    setLoading(false);
    setStatsLoading(false);
  }
};

const handleNewAnalysis = () => {
  navigate('/upload');
};

const handleViewResults = () => {
  navigate('/results');
};

const handleRefresh = () => {
  fetchData();
};

if (loading && !recentResults.length) {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
      <CircularProgress />
    </Box>
  );
}

// Calculate fake/real percentage
const totalDetected = stats.fakeDetected + stats.realDetected;
const fakePercentage = totalDetected > 0 ? (stats.fakeDetected / totalDetected) * 100 : 0;
const realPercentage = totalDetected > 0 ? (stats.realDetected / totalDetected) * 100 : 0;

return (
  <Box>
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
      <Typography variant="h4">Dashboard</Typography>
      <Box>
        <IconButton onClick={handleRefresh} disabled={loading}>
          <RefreshIcon />
        </IconButton>
        <Button
          variant="contained"
          color="primary"
          startIcon={<CloudUploadIcon />}
          onClick={handleNewAnalysis}
          sx={{ ml: 1 }}
        >
          New Analysis
        </Button>
      </Box>
    </Box>
    
    {error && (
      <Alert severity="error" sx={{ mb: 3 }}>
        {error}
      </Alert>
    )}
    
    {/* Welcome card */}
    <Paper
      sx={{
        p: 3,
        mb: 4,
        borderRadius: 2,
        backgroundImage: 'linear-gradient(135deg, rgba(63,81,181,0.1) 0%, rgba(63,81,181,0.05) 100%)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          right: 0,
          bottom: 0,
          width: { xs: '100%', md: '40%' },
          background: 'radial-gradient(circle at right, rgba(63,81,181,0.08) 0%, transparent 70%)',
          zIndex: 0,
        }}
      />
      
      <Grid container spacing={3} sx={{ position: 'relative', zIndex: 1 }}>
        <Grid item xs={12} md={8}>
          <Typography variant="h5" gutterBottom>
            Welcome, {user?.username || 'User'}!
          </Typography>
          <Typography variant="body1" paragraph>
            LightMultiDetect provides ultra-efficient deepfake detection across multiple modalities,
            combining state-of-the-art accuracy with minimal resource requirements.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Button
              variant="outlined"
              color="primary"
              startIcon={<CloudUploadIcon />}
              onClick={handleNewAnalysis}
            >
              New Analysis
            </Button>
            <Button
              variant="outlined"
              startIcon={<AssessmentIcon />}
              onClick={handleViewResults}
            >
              View Results
            </Button>
          </Box>
        </Grid>
        <Grid item xs={12} md={4}>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              height: '100%',
              justifyContent: 'center',
              alignItems: { xs: 'flex-start', md: 'flex-end' },
              textAlign: { xs: 'left', md: 'right' },
            }}
          >
            <Typography variant="h3" color="primary">
              {statsLoading ? <CircularProgress size={24} /> : stats.total}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total analyses performed
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Paper>
    
    {/* Statistics cards */}
    <Grid container spacing={3} sx={{ mb: 4 }}>
      <Grid item xs={12} sm={6} md={3}>
        <Paper sx={{ p: 2, borderRadius: 2, height: '100%' }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Detection Accuracy
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ position: 'relative', display: 'inline-flex', mr: 2 }}>
              <CircularProgress
                variant="determinate"
                value={96}
                size={60}
                thickness={4}
                sx={{ color: theme.palette.primary.main }}
              />
              <Box
                sx={{
                  top: 0,
                  left: 0,
                  bottom: 0,
                  right: 0,
                  position: 'absolute',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Typography variant="caption" component="div" color="text.secondary">
                  96%
                </Typography>
              </Box>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Avg. confidence
              </Typography>
              <Typography variant="h6">
                94.2%
              </Typography>
            </Box>
          </Box>
        </Paper>
      </Grid>
      
      <Grid item xs={12} sm={6} md={3}>
        <Paper sx={{ p: 2, borderRadius: 2, height: '100%' }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Processing Speed
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <SpeedIcon sx={{ fontSize: 40, color: theme.palette.success.main, mr: 2 }} />
            <Box>
              <Typography variant="h5">
                25ms
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg. inference time
              </Typography>
            </Box>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            10-50x faster than SOTA models
          </Typography>
        </Paper>
      </Grid>
      
      <Grid item xs={12} sm={6} md={3}>
        <Paper sx={{ p: 2, borderRadius: 2, height: '100%' }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Status
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
            <Box sx={{ mr: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Pending
              </Typography>
              <Typography variant="h6">
                {statsLoading ? <CircularProgress size={16} /> : stats.pending}
              </Typography>
            </Box>
            <Box sx={{ mr: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Processing
              </Typography>
              <Typography variant="h6">
                {statsLoading ? <CircularProgress size={16} /> : stats.processing}
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Completed
              </Typography>
              <Typography variant="h6">
                {statsLoading ? <CircularProgress size={16} /> : stats.completed}
              </Typography>
            </Box>
          </Box>
        </Paper>
      </Grid>
      
      <Grid item xs={12} sm={6} md={3}>
        <Paper sx={{ p: 2, borderRadius: 2, height: '100%' }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Resource Usage
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <MemoryIcon sx={{ fontSize: 40, color: theme.palette.info.main, mr: 2 }} />
            <Box>
              <Typography variant="h5">
                4.8MB
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Model size
              </Typography>
            </Box>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Optimized for edge devices
          </Typography>
        </Paper>
      </Grid>
    </Grid>
    
    {/* Charts */}
    <Grid container spacing={3} sx={{ mb: 4 }}>
      <Grid item xs={12} md={8}>
        <Paper sx={{ p: 3, borderRadius: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Detection History
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Real vs. fake detections over time
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart
              data={detectionHistory}
              margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis dataKey="date" stroke={theme.palette.text.secondary} />
              <YAxis stroke={theme.palette.text.secondary} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: theme.palette.background.paper,
                  borderColor: theme.palette.divider
                }}
              />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="fake" 
                stackId="1"
                stroke={theme.palette.error.main} 
                fill={theme.palette.error.main} 
                fillOpacity={0.5}
              />
              <Area 
                type="monotone" 
                dataKey="real" 
                stackId="1"
                stroke={theme.palette.success.main} 
                fill={theme.palette.success.main}
                fillOpacity={0.5}
              />
            </AreaChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>
      
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 3, borderRadius: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Detection by Type
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Distribution across media types
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={detectionByType}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="count"
              >
                {detectionByType.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: theme.palette.background.paper,
                  borderColor: theme.palette.divider
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3, borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom>
            Fake vs. Real Distribution
          </Typography>
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2">
                {statsLoading ? <CircularProgress size={12} /> : stats.fakeDetected} Fake
              </Typography>
              <Typography variant="body2">
                {Math.round(fakePercentage)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={fakePercentage}
              color="error"
              sx={{ height: 8, borderRadius: 4, mb: 1 }}
            />
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2">
                {statsLoading ? <CircularProgress size={12} /> : stats.realDetected} Real
              </Typography>
              <Typography variant="body2">
                {Math.round(realPercentage)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={realPercentage}
              color="success"
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Box>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Total analyzed media: {statsLoading ? <CircularProgress size={12} /> : totalDetected}
            </Typography>
          </Box>
        </Paper>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3, borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom>
            Performance Metrics
          </Typography>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={performanceMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis dataKey="name" stroke={theme.palette.text.secondary} />
              <YAxis domain={[0, 100]} stroke={theme.palette.text.secondary} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: theme.palette.background.paper,
                  borderColor: theme.palette.divider
                }}
                formatter={(value) => [`${value}%`, 'Value']}
              />
              <Bar dataKey="value" fill={theme.palette.primary.main} />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>
    </Grid>
    
    {/* Recent results */}
    <Paper sx={{ p: 3, borderRadius: 2, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Recent Analyses</Typography>
        <Button
          size="small"
          onClick={handleViewResults}
          endIcon={<AssessmentIcon />}
        >
          View All
        </Button>
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      {recentResults.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
          No recent detection tasks found.
        </Typography>
      ) : (
        recentResults.map((result, index) => (
          <React.Fragment key={result.taskId}>
            {index > 0 && <Divider sx={{ my: 2 }} />}
            <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  bgcolor: 
                    result.status === 'completed' ? 'success.main' :
                    result.status === 'failed' ? 'error.main' :
                    'warning.main',
                  mr: 2,
                  mt: 0.5,
                }}
              />
              <Box sx={{ flex: 1 }}>
                <Typography variant="body2">
                  {result.modality && `${result.modality.charAt(0).toUpperCase() + result.modality.slice(1)} detection`} ({result.numFiles} files)
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  ID: {result.taskId.substring(0, 8)}... • {new Date(result.createdAt * 1000).toLocaleString()}
                </Typography>
              </Box>
              <Button
                size="small"
                variant="outlined"
                onClick={() => navigate(`/results/${result.taskId}`)}
              >
                Details
              </Button>
            </Box>
          </React.Fragment>
        ))
      )}
    </Paper>
    
    {/* Platform information */}
    <Paper sx={{ p: 3, borderRadius: 2 }}>
      <Typography variant="h6" gutterBottom>
        About LightMultiDetect
      </Typography>
      <Typography variant="body2" paragraph>
        LightMultiDetect is a cutting-edge platform for detecting deepfakes across multiple modalities.
        Our lightweight models are designed to run efficiently on a variety of devices while maintaining
        high detection accuracy.
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} sm={4}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
            <ShieldIcon color="primary" sx={{ mr: 1 }} />
            <Box>
              <Typography variant="subtitle2">
                Security Features
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Advanced cross-domain deepfake detection with temporal and frequency analysis
              </Typography>
            </Box>
          </Box>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
            <SpeedIcon color="secondary" sx={{ mr: 1 }} />
            <Box>
              <Typography variant="subtitle2">
                High Performance
              </Typography>
              <Typography variant="body2" color="text.secondary">
                10-50x faster inference than comparable state-of-the-art models
              </Typography>
            </Box>
          </Box>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
            <MemoryIcon color="error" sx={{ mr: 1 }} />
            <Box>
              <Typography variant="subtitle2">
                Resource Efficient
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Compact model size (4.8MB) with minimal memory requirements
              </Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  </Box>
);
};

export default Dashboard;


// frontend/src/pages/ProfilePage.js
import React, { useState } from 'react';
import {
Box,
Typography,
Paper,
TextField,
Button,
Avatar,
Grid,
Divider,
IconButton,
Alert,
Dialog,
DialogTitle,
DialogContent,
DialogContentText,
DialogActions,
CircularProgress,
Card,
CardContent,
List,
ListItem,
ListItemIcon,
ListItemText,
Switch,
} from '@mui/material';
import {
Save as SaveIcon,
Edit as EditIcon,
Person as PersonIcon,
Email as EmailIcon,
Notifications as NotificationsIcon,
Security as SecurityIcon,
LockReset as LockResetIcon,
AccountCircle as AccountCircleIcon,
Visibility as VisibilityIcon,
VisibilityOff as VisibilityOffIcon,
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';

const ProfilePage = () => {
const { user, logout } = useAuth();
const [edit, setEdit] = useState(false);
const [loading, setLoading] = useState(false);
const [error, setError] = useState(null);
const [success, setSuccess] = useState(null);
const [showPasswordDialog, setShowPasswordDialog] = useState(false);
const [showDeleteDialog, setShowDeleteDialog] = useState(false);

const [formData, setFormData] = useState({
  username: user?.username || '',
  email: user?.email || '',
  firstName: 'John',
  lastName: 'Doe',
  notificationsEnabled: true,
});

const [passwordData, setPasswordData] = useState({
  currentPassword: '',
  newPassword: '',
  confirmPassword: '',
});

const [showPassword, setShowPassword] = useState({
  current: false,
  new: false,
  confirm: false,
});

const handleChange = (e) => {
  const { name, value, checked } = e.target;
  
  if (name === 'notificationsEnabled') {
    setFormData({
      ...formData,
      [name]: checked,
    });
  } else {
    setFormData({
      ...formData,
      [name]: value,
    });
  }
};

const handlePasswordChange = (e) => {
  const { name, value } = e.target;
  
  setPasswordData({
    ...passwordData,
    [name]: value,
  });
};

const handleToggleEdit = () => {
  setEdit(!edit);
  
  // Reset form data if canceling edit
  if (edit) {
    setFormData({
      username: user?.username || '',
      email: user?.email || '',
      firstName: 'John',
      lastName: 'Doe',
      notificationsEnabled: true,
    });
  }
};

const handleSubmit = async (e) => {
  e.preventDefault();
  
  try {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    // In a real app, this would be an API call to update the profile
    console.log('Updating profile:', formData);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    setSuccess('Profile updated successfully.');
    setEdit(false);
  } catch (err) {
    console.error('Error updating profile:', err);
    setError(err.message || 'Failed to update profile.');
  } finally {
    setLoading(false);
  }
};

const handlePasswordSubmit = async (e) => {
  e.preventDefault();
  
  // Validate password
  if (passwordData.newPassword !== passwordData.confirmPassword) {
    setError('New passwords do not match.');
    return;
  }
  
  if (passwordData.newPassword.length < 6) {
    setError('Password must be at least 6 characters long.');
    return;
  }
  
  try {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    // In a real app, this would be an API call to change the password
    console.log('Changing password');
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Reset password form
    setPasswordData({
      currentPassword: '',
      newPassword: '',
      confirmPassword: '',
    });
    
    setSuccess('Password changed successfully.');
    setShowPasswordDialog(false);
  } catch (err) {
    console.error('Error changing password:', err);
    setError(err.message || 'Failed to change password.');
  } finally {
    setLoading(false);
  }
};

const handleDeleteAccount = async () => {
  try {
    setLoading(true);
    
    // In a real app, this would be an API call to delete the account
    console.log('Deleting account');
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Logout user
    logout();
    
    // Redirect to login page will happen automatically due to auth context
  } catch (err) {
    console.error('Error deleting account:', err);
    setError(err.message || 'Failed to delete account.');
    setShowDeleteDialog(false);
  } finally {
    setLoading(false);
  }
};

return (
  <Box>
    <Typography variant="h4" gutterBottom>
      Profile Settings
    </Typography>
    
    <Grid container spacing={3}>
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 3, borderRadius: 2, mb: { xs: 3, md: 0 } }}>
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 3 }}>
            <Avatar
              sx={{
                width: 100,
                height: 100,
                bgcolor: 'primary.main',
                fontSize: 40,
                mb: 2,
              }}
            >
              {formData.firstName?.[0] || formData.username?.[0] || 'U'}
            </Avatar>
            <Typography variant="h6">
              {formData.firstName} {formData.lastName}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {formData.username}
            </Typography>
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <List>
            <ListItem button>
              <ListItemIcon>
                <AccountCircleIcon />
              </ListItemIcon>
              <ListItemText
                primary="Account Settings"
                secondary="Update your personal details"
              />
            </ListItem>
            <ListItem button onClick={() => setShowPasswordDialog(true)}>
              <ListItemIcon>
                <LockResetIcon />
              </ListItemIcon>
              <ListItemText
                primary="Change Password"
                secondary="Update your password"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <NotificationsIcon />
              </ListItemIcon>
              <ListItemText
                primary="Notifications"
                secondary="Manage notification settings"
              />
              <Switch
                edge="end"
                checked={formData.notificationsEnabled}
                onChange={handleChange}
                name="notificationsEnabled"
                disabled={!edit}
              />
            </ListItem>
          </List>
          
          <Divider sx={{ my: 2 }} />
          
          <Button
            fullWidth
            variant="outlined"
            color="error"
            onClick={() => setShowDeleteDialog(true)}
          >
            Delete Account
          </Button>
        </Paper>
      </Grid>
      
      <Grid item xs={12} md={8}>
        {success && (
          <Alert severity="success" sx={{ mb: 3 }}>
            {success}
          </Alert>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        <Paper sx={{ p: 3, borderRadius: 2, mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h6">Account Information</Typography>
            <IconButton onClick={handleToggleEdit} color={edit ? 'error' : 'primary'}>
              {edit ? <SaveIcon /> : <EditIcon />}
            </IconButton>
          </Box>
          
          <form onSubmit={handleSubmit}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="First Name"
                  name="firstName"
                  value={formData.firstName}
                  onChange={handleChange}
                  disabled={!edit}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Last Name"
                  name="lastName"
                  value={formData.lastName}
                  onChange={handleChange}
                  disabled={!edit}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Username"
                  name="username"
                  value={formData.username}
                  onChange={handleChange}
                  disabled
                  InputProps={{
                    startAdornment: (
                      <PersonIcon color="action" sx={{ mr: 1 }} />
                    ),
                  }}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  disabled={!edit}
                  InputProps={{
                    startAdornment: (
                      <EmailIcon color="action" sx={{ mr: 1 }} />
                    ),
                  }}
                />
              </Grid>
            </Grid>
            
            {edit && (
              <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  variant="outlined"
                  onClick={handleToggleEdit}
                  sx={{ mr: 1 }}
                  disabled={loading}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="contained"
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : <SaveIcon />}
                >
                  Save Changes
                </Button>
              </Box>
            )}
          </form>
        </Paper>
        
        <Paper sx={{ p: 3, borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom>
            Account Security
          </Typography>
          
          <Card variant="outlined" sx={{ mb: 2 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <SecurityIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="body1">
                  Password
                </Typography>
                <Box sx={{ flexGrow: 1 }} />
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => setShowPasswordDialog(true)}
                >
                  Change
                </Button>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Last changed: 30 days ago
              </Typography>
            </CardContent>
          </Card>
          
          <Typography variant="body2" paragraph>
            Protect your account by using a strong password and enabling additional security features.
          </Typography>
          
          <List>
            <ListItem>
              <ListItemIcon>
                <SecurityIcon color="success" />
              </ListItemIcon>
              <ListItemText
                primary="Your account is secure"
                secondary="Password strength: Strong"
              />
            </ListItem>
          </List>
        </Paper>
      </Grid>
    </Grid>
    
    {/* Password change dialog */}
    <Dialog open={showPasswordDialog} onClose={() => setShowPasswordDialog(false)}>
      <DialogTitle>Change Password</DialogTitle>
      <form onSubmit={handlePasswordSubmit}>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            Please enter your current password and a new password to update your account security.
          </DialogContentText>
          
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          
          <TextField
            fullWidth
            label="Current Password"
            type={showPassword.current ? 'text' : 'password'}
            name="currentPassword"
            value={passwordData.currentPassword}
            onChange={handlePasswordChange}
            margin="dense"
            required
            InputProps={{
              endAdornment: (
                <IconButton
                  onClick={() => setShowPassword({ ...showPassword, current: !showPassword.current })}
                  edge="end"
                >
                  {showPassword.current ? <VisibilityOffIcon /> : <VisibilityIcon />}
                </IconButton>
              ),
            }}
          />
          <TextField
            fullWidth
            label="New Password"
            type={showPassword.new ? 'text' : 'password'}
            name="newPassword"
            value={passwordData.newPassword}
            onChange={handlePasswordChange}
            margin="dense"
            required
            InputProps={{
              endAdornment: (
                <IconButton
                  onClick={() => setShowPassword({ ...showPassword, new: !showPassword.new })}
                  edge="end"
                >
                  {showPassword.new ? <VisibilityOffIcon /> : <VisibilityIcon />}
                </IconButton>
              ),
            }}
          />
          <TextField
            fullWidth
            label="Confirm New Password"
            type={showPassword.confirm ? 'text' : 'password'}
            name="confirmPassword"
            value={passwordData.confirmPassword}
            onChange={handlePasswordChange}
            margin="dense"
            required
            InputProps={{
              endAdornment: (
                <IconButton
                  onClick={() => setShowPassword({ ...showPassword, confirm: !showPassword.confirm })}
                  edge="end"
                >
                  {showPassword.confirm ? <VisibilityOffIcon /> : <VisibilityIcon />}
                </IconButton>
              ),
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPasswordDialog(false)} disabled={loading}>
            Cancel
          </Button>
          <Button
            type="submit"
            variant="contained"
            disabled={loading}
            startIcon={loading ? <CircularProgress size={20} /> : <SaveIcon />}
          >
            Update Password
          </Button>
        </DialogActions>
      </form>
    </Dialog>
    
    {/* Delete account dialog */}
    <Dialog open={showDeleteDialog} onClose={() => setShowDeleteDialog(false)}>
      <DialogTitle>Delete Account</DialogTitle>
      <DialogContent>
        <DialogContentText>
          Are you sure you want to delete your account? This action cannot be undone and will permanently erase all your data.
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setShowDeleteDialog(false)} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleDeleteAccount}
          color="error"
          variant="contained"
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : null}
        >
          Delete Account
        </Button>
      </DialogActions>
    </Dialog>
  </Box>
);
};

export default ProfilePage;


// frontend/src/pages/NotFoundPage.js
import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
Box,
Typography,
Button,
Paper,
} from '@mui/material';
import { Home as HomeIcon } from '@mui/icons-material';

const NotFoundPage = () => {
const navigate = useNavigate();

return (
  <Box
    sx={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #121212 0%, #2c2c2c 100%)',
      p: 2,
    }}
  >
    <Paper
      sx={{
        p: 4,
        maxWidth: 500,
        width: '100%',
        borderRadius: 3,
        boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
        textAlign: 'center',
      }}
      elevation={8}
    >
      <Typography variant="h1" color="primary" sx={{ fontSize: 100, fontWeight: 'bold' }}>
        404
      </Typography>
      <Typography variant="h4" gutterBottom>
        Page Not Found
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        The page you are looking for doesn't exist or has been moved.
      </Typography>
      <Button
        variant="contained"
        startIcon={<HomeIcon />}
        onClick={() => navigate('/')}
        size="large"
        sx={{ mt: 2 }}
      >
        Go Home
      </Button>
      
      {/* Background futuristic grid */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: -1,
          opacity: 0.1,
          backgroundImage: `
            linear-gradient(#3f51b5 1px, transparent 1px),
            linear-gradient(90deg, #3f51b5 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px',
          pointerEvents: 'none',
        }}
      />
    </Paper>
  </Box>
);
};

export default NotFoundPage;