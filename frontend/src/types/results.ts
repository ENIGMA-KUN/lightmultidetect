// Type definitions for result-related interfaces

// Basic result information
export interface ResultInfo {
    id: string;
    media_type: 'image' | 'audio' | 'video';
    is_fake: boolean;
    confidence_score: number;
    created_at: string;
  }
  
  // Detailed result with all analysis data
  export interface DetailedResult extends ResultInfo {
    detection_details: Record<string, any>;
    models_used: Record<string, string>;
    visualizations?: VisualizationData;
  }
  
  // Visualization data for different analysis types
  export interface VisualizationData {
    heatmap?: HeatmapData;
    temporal?: TemporalData;
    frequency?: Record<string, any>;
  }
  
  // Heatmap visualization data
  export interface HeatmapData {
    url: string;
    width: number;
    height: number;
    regions: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      confidence: number;
      label?: string;
    }>;
  }
  
  // Temporal analysis visualization data
  export interface TemporalData {
    timestamps: number[];
    values: number[];
    threshold: number;
  }
  
  // Result statistics
  export interface ResultStatistics {
    period_days: number;
    total_results: number;
    media_type_distribution: {
      image: number;
      audio: number;
      video: number;
    };
    real_count: number;
    fake_count: number;
    fake_percentage: number;
    average_confidence: number;
  }
  
  // Query options for results
  export interface ResultQuery {
    task_id?: string;
    result_id?: string;
  }
  
  // Status response for a query
  export interface ResultStatus {
    status: string;
    progress: number;
    message?: string;
    result_id?: string;
  }