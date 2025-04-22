import plotly.graph_objects as go
import numpy as np

def default_coordinate_conversion(landmarks, depth_frame=None, image_width=640, image_height=480):
    """
    Convert landmarks from the MediaPipe format to 3D coordinates.
    This is a basic implementation that can be enhanced with actual depth data.
    
    Args:
        landmarks: MediaPipe hand landmarks
        depth_frame: Optional depth frame to get Z coordinates
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        x_coords, y_coords, z_coords: Lists of coordinates for plotting
    """
    if landmarks is None:
        # Return empty lists if no landmarks
        return [], [], []
        
    x_coords = []
    y_coords = []
    z_coords = []
    
    for lm in landmarks.landmark:
        # Convert from normalized coordinates to pixel coordinates
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        
        # Use landmark.z or get depth from depth frame if available
        z = lm.z * 100  # Scale for visualization
        
        # Optional: If depth frame is available, use it for Z
        if depth_frame is not None and 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
            pixel_depth = depth_frame[y, x]
            if pixel_depth > 0:  # Valid depth
                z = pixel_depth / 10.0  # Convert mm to cm
        
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
    
    return x_coords, y_coords, z_coords

def create_hand_figure():
    """
    Create a 3D figure for hand visualization.
    
    Returns:
        fig: A plotly figure object for 3D hand visualization
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=[], y=[], z=[],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.8
        ),
        name='Hand Landmarks'
    )])
    
    # Add lines for finger connections (standard MediaPipe hand connections)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='lines',
        line=dict(color='green', width=3),
        name='Connections'
    ))
    
    # Configure the layout
    fig.update_layout(
        title="Hand Tracking Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0, y=1),
        template="plotly_dark"
    )
    
    return fig

def update_hand_visualization(fig, landmarks, coordinate_conversion_fn=default_coordinate_conversion):
    """
    Update the 3D hand visualization with new landmark data.
    
    Args:
        fig: The plotly figure to update
        landmarks: MediaPipe hand landmarks
        coordinate_conversion_fn: Function to convert landmarks to 3D coordinates
        
    Returns:
        fig: The updated figure
    """
    if landmarks is None or not hasattr(landmarks, 'landmark'):
        # Clear the figure if no landmarks
        fig.data[0].x = []
        fig.data[0].y = []
        fig.data[0].z = []
        fig.data[1].x = []
        fig.data[1].y = []
        fig.data[1].z = []
        return fig
    
    # Convert landmarks to 3D coordinates
    x_coords, y_coords, z_coords = coordinate_conversion_fn(landmarks)
    
    # Update the markers
    fig.data[0].x = x_coords
    fig.data[0].y = y_coords
    fig.data[0].z = z_coords
    
    # Define connections (based on MediaPipe hand connections)
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (0, 5), (5, 9), (9, 13), (13, 17)
    ]
    
    # Create line segments for connections
    x_lines = []
    y_lines = []
    z_lines = []
    
    if len(x_coords) >= 21:  # Make sure we have all landmarks
        for start, end in connections:
            x_lines.extend([x_coords[start], x_coords[end], None])
            y_lines.extend([y_coords[start], y_coords[end], None])
            z_lines.extend([z_coords[start], z_coords[end], None])
    
    # Update the lines
    fig.data[1].x = x_lines
    fig.data[1].y = y_lines
    fig.data[1].z = z_lines
    
    return fig 