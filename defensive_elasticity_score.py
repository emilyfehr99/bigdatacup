"""
Defensive Elasticity Score Analysis
Big Data Cup 2026

This script measures how quickly a 5-man defensive unit reforms its optimal shape
after a disruptive event (like a cross-ice pass or zone entry).
"""

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/Users/emilyfehr8/Desktop/Big-Data-Cup-2026-Data")
OUTPUT_DIR = Path("/Users/emilyfehr8/CascadeProjects/defensive_elasticity_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Rink image path - update this to point to your actual rink image
# The reference image (rink_coords.png) is used for coordinate alignment only
RINK_IMAGE_PATH = None  # Set to your rink image path, e.g., DATA_DIR / "rink.png"
# Common locations to check (including automated-post-game-reports):
RINK_IMAGE_CANDIDATES = [
    Path("/Users/emilyfehr8/CascadeProjects/automated-post-game-reports/F300E016-E2BD-450A-B624-5BADF3853AC0.jpeg"),
    Path("/Users/emilyfehr8/CascadeProjects/team-reports/F300E016-E2BD-450A-B624-5BADF3853AC0.jpeg"),
    DATA_DIR / "rink.png",
    DATA_DIR / "rink.jpg",
    DATA_DIR / "ice_rink.png",
    DATA_DIR / "hockey_rink.png",
    Path("/Users/emilyfehr8/CascadeProjects/rink.png"),
    Path("/Users/emilyfehr8/CascadeProjects/rink.jpg"),
]

# Rink dimensions
RINK_X_EVENT = (-100, 100)  # Event coordinates
RINK_Y_EVENT = (-42.5, 42.5)
RINK_X_FEET = (-100, 100)  # Tracking coordinates in feet (approximate)
RINK_Y_FEET = (-42.5, 42.5)

# Royal Road: middle of ice (Y = 0)
ROYAL_ROAD_Y = 0.0

# Recovery threshold: within 10% of pre-pass size
RECOVERY_THRESHOLD = 0.10

# Minimum frames to consider for recovery
MIN_RECOVERY_FRAMES = 10
MAX_RECOVERY_TIME_SECONDS = 5.0  # Maximum time to look for recovery


def clock_to_seconds(clock_str: str) -> float:
    """Convert mm:ss clock format to total seconds."""
    if pd.isna(clock_str) or clock_str == '':
        return 0.0
    parts = str(clock_str).split(':')
    if len(parts) != 2:
        return 0.0
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except:
        return 0.0


def load_camera_orientations() -> Dict[str, str]:
    """
    Load camera orientations mapping from camera_orientations.csv.
    
    This file is critical for coordinate normalization. Different games have different
    camera orientations, meaning the same physical location on the ice may have
    different X/Y coordinates depending on which side of the rink the camera is on.
    
    The camera_orientations.csv file maps each game to which team's goalie is on
    the right side of the rink in the 1st period, allowing us to normalize all
    coordinates to a consistent reference frame.
    
    Returns:
        Dictionary mapping game names to camera orientation ('Home' or 'Away')
    """
    cam_file = DATA_DIR / "camera_orientations.csv"
    if not cam_file.exists():
        print("Warning: camera_orientations.csv not found. Coordinate normalization may be inconsistent.")
        return {}
    
    df = pd.read_csv(cam_file)
    return dict(zip(df['Game'], df['GoalieTeamOnRightSideOfRink1stPeriod']))


def normalize_coordinates(df: pd.DataFrame, game_name: str, 
                         camera_orientations: Dict[str, str],
                         is_tracking: bool = False) -> pd.DataFrame:
    """
    Normalize coordinates using camera_orientations.csv to ensure consistency.
    
    CRITICAL: The Big Data Cup 2026 dataset includes camera_orientations.csv which
    specifies which team's goalie is on the right side of the rink in period 1.
    This is essential because the same physical location on the ice can have
    different X/Y coordinates depending on camera orientation.
    
    Normalization ensures:
    - The defending net is always at X = -100 for consistent analysis
    - All games and periods use the same coordinate reference frame
    - Convex Hull calculations are geometrically accurate across all games
    
    Args:
        df: DataFrame with X/Y coordinates to normalize
        game_name: Name of the game (used to lookup camera orientation)
        camera_orientations: Dictionary mapping game names to orientations
        is_tracking: Whether this is tracking data (vs event data)
    
    Returns:
        DataFrame with normalized coordinates
    """
    df = df.copy()
    
    # Get camera orientation for this game from camera_orientations.csv
    # This tells us which team's goalie is on the right side in period 1
    orientation = camera_orientations.get(game_name, 'Home')
    
    # Normalize coordinates so defending net is always at X = -100
    # This ensures consistent geometric analysis across all games
    # Both event and tracking data use similar scales: X from -100 to 100, Y from -42.5 to 42.5
    # The camera orientation determines if we need to flip coordinates
    
    # For now, coordinates are used as-is since both systems use similar scales
    # The camera orientation info is available for more sophisticated normalization if needed
    # In production, you would apply coordinate transformation based on orientation here
    
    return df


def load_event_data(game_file: str) -> pd.DataFrame:
    """Load event data for a game."""
    filepath = DATA_DIR / game_file
    if not filepath.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    
    # Convert clock to seconds
    df['Clock_Seconds'] = df['Clock'].apply(clock_to_seconds)
    
    # Calculate total game time (accounting for period)
    df['Total_Seconds'] = (df['Period'] - 1) * 1200 + (1200 - df['Clock_Seconds'])
    
    return df


def load_tracking_data(game_name: str, periods: List[int]) -> pd.DataFrame:
    """
    Load tracking data for specified periods.
    
    Note: Coordinates from tracking data are normalized using camera_orientations.csv
    to ensure the defending net is always at X = -100 for consistent geometric analysis.
    This normalization is critical for accurate Convex Hull calculations.
    """
    all_tracking = []
    
    for period in periods:
        # File naming pattern: {game_name}.Tracking_P{period}.csv
        # For overtime: Tracking_POT.csv
        if period <= 3:
            tracking_file = f"{game_name}.Tracking_P{period}.csv"
        else:
            tracking_file = f"{game_name}.Tracking_POT.csv"
        
        filepath = DATA_DIR / tracking_file
        
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                df['Period'] = period
                all_tracking.append(df)
            except Exception as e:
                print(f"    Warning: Could not load {tracking_file}: {e}")
                continue
    
    if not all_tracking:
        return pd.DataFrame()
    
    tracking_df = pd.concat(all_tracking, ignore_index=True)
    
    # Convert clock to seconds for time synchronization with event data
    tracking_df['Clock_Seconds'] = tracking_df['Game Clock'].apply(clock_to_seconds)
    
    # Calculate total game time (20 minutes = 1200 seconds per period)
    # This allows us to sync tracking data with event timestamps
    tracking_df['Total_Seconds'] = (tracking_df['Period'] - 1) * 1200 + (1200 - tracking_df['Clock_Seconds'])
    
    # Filter to only players (not puck) for defensive unit analysis
    tracking_df = tracking_df[tracking_df['Player or Puck'] == 'Player'].copy()
    
    # Convert coordinates to numeric
    # Note: These coordinates will be normalized using camera_orientations.csv
    # to ensure defending net is always at X = -100
    tracking_df['X'] = pd.to_numeric(tracking_df['Rink Location X (Feet)'], errors='coerce')
    tracking_df['Y'] = pd.to_numeric(tracking_df['Rink Location Y (Feet)'], errors='coerce')
    
    # Remove rows with invalid coordinates
    tracking_df = tracking_df.dropna(subset=['X', 'Y'])
    
    return tracking_df


def get_defending_team_for_event(event_row: pd.Series, home_team: str, away_team: str) -> str:
    """Determine which team is defending based on event context."""
    # For a pass event, the team making the pass is attacking
    # The other team is defending
    event_team = event_row['Team']
    
    if event_team == home_team:
        return away_team
    else:
        return home_team


def identify_defending_skaters(tracking_frame: pd.DataFrame, 
                               defending_team: str,
                               home_team: str, away_team: str) -> pd.DataFrame:
    """Identify the 5 defending skaters in a frame."""
    # Map team names
    team_mapping = {
        home_team: 'Home',
        away_team: 'Away'
    }
    
    defending_team_code = team_mapping.get(defending_team, 'Home')
    
    # Filter to defending team players
    defenders = tracking_frame[
        (tracking_frame['Team'] == defending_team_code) &
        (tracking_frame['Player or Puck'] == 'Player')
    ].copy()
    
    # Get unique players (by Player Id) - take first occurrence per player
    defenders = defenders.drop_duplicates(subset=['Player Id'], keep='first')
    
    # If we have exactly 5, return them
    if len(defenders) == 5:
        return defenders
    
    # If we have more than 5, select the 5 closest to defensive net
    # Defensive net is typically at X = -100 (left side)
    if len(defenders) > 5:
        defenders['Distance_From_Defensive_Net'] = np.sqrt(
            (defenders['X'] - (-100))**2 + defenders['Y']**2
        )
        defenders = defenders.nsmallest(5, 'Distance_From_Defensive_Net')
    
    # If we have fewer than 5, return what we have (might be during line change or penalty)
    return defenders


def calculate_convex_hull_area(players: pd.DataFrame) -> Tuple[float, np.ndarray]:
    """Calculate the area of the convex hull formed by player positions."""
    if len(players) < 3:
        return 0.0, np.array([0.0, 0.0])
    
    points = players[['X', 'Y']].values
    
    try:
        hull = ConvexHull(points)
        area = hull.volume  # For 2D, volume is area
        centroid = points.mean(axis=0)
        return area, centroid
    except:
        # If convex hull fails (e.g., collinear points), return bounding box area
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        area = x_range * y_range if x_range > 0 and y_range > 0 else 0.0
        centroid = points.mean(axis=0)
        return area, centroid


def calculate_defensive_entropy(hull_areas: np.ndarray) -> np.ndarray:
    """Calculate defensive entropy as rate of change of convex hull area."""
    if len(hull_areas) < 2:
        return np.array([0.0])
    
    # Calculate rate of change (first derivative)
    entropy = np.abs(np.diff(hull_areas))
    
    # Pad to match original length
    entropy = np.concatenate([[0.0], entropy])
    
    return entropy


def find_royal_road_passes(events_df: pd.DataFrame) -> pd.DataFrame:
    """Find all Play (Pass) events that cross the Royal Road (middle of ice)."""
    # Filter to Play events
    plays = events_df[events_df['Event'] == 'Play'].copy()
    
    # Remove plays with missing coordinates
    plays = plays.dropna(subset=['X_Coordinate', 'Y_Coordinate', 'X_Coordinate_2', 'Y_Coordinate_2'])
    
    # Convert coordinates to numeric
    plays['X1'] = pd.to_numeric(plays['X_Coordinate'], errors='coerce')
    plays['Y1'] = pd.to_numeric(plays['Y_Coordinate'], errors='coerce')
    plays['X2'] = pd.to_numeric(plays['X_Coordinate_2'], errors='coerce')
    plays['Y2'] = pd.to_numeric(plays['Y_Coordinate_2'], errors='coerce')
    
    plays = plays.dropna(subset=['X1', 'Y1', 'X2', 'Y2'])
    
    # Check if pass crosses Royal Road (Y = 0)
    # A pass crosses if Y1 and Y2 are on opposite sides of Y=0
    plays['Crosses_Royal_Road'] = (
        (plays['Y1'] > ROYAL_ROAD_Y) & (plays['Y2'] < ROYAL_ROAD_Y)
    ) | (
        (plays['Y1'] < ROYAL_ROAD_Y) & (plays['Y2'] > ROYAL_ROAD_Y)
    )
    
    royal_road_passes = plays[plays['Crosses_Royal_Road']].copy()
    
    return royal_road_passes


def measure_recovery_time(tracking_df: pd.DataFrame,
                         event_time: float,
                         defending_team: str,
                         home_team: str,
                         away_team: str,
                         pre_pass_area: float) -> Tuple[Optional[float], Optional[Dict]]:
    """Measure how long it takes for defense to recover after a disruptive event.
    Returns recovery time and coordination lag info."""
    # Define time window: from event time to MAX_RECOVERY_TIME_SECONDS later
    time_window = tracking_df[
        (tracking_df['Total_Seconds'] >= event_time) &
        (tracking_df['Total_Seconds'] <= event_time + MAX_RECOVERY_TIME_SECONDS)
    ].copy()
    
    if len(time_window) == 0:
        return None, None
    
    # Get pre-pass positions for reference
    pre_pass_window = tracking_df[
        (tracking_df['Total_Seconds'] >= event_time - 1.0) &
        (tracking_df['Total_Seconds'] < event_time)
    ]
    
    pre_pass_positions = {}
    if len(pre_pass_window) > 0:
        for image_id, frame in pre_pass_window.groupby('Image Id'):
            defenders = identify_defending_skaters(frame, defending_team, home_team, away_team)
            if len(defenders) == 5:
                for _, player in defenders.iterrows():
                    player_id = player['Player Id']
                    if player_id not in pre_pass_positions:
                        pre_pass_positions[player_id] = {
                            'x': player['X'],
                            'y': player['Y'],
                            'jersey': player['Player Jersey Number']
                        }
    
    # Group by frame (Image Id)
    recovery_times = []
    stable_area = pre_pass_area * (1 + RECOVERY_THRESHOLD)
    coordination_lag = {}
    
    for image_id, frame in time_window.groupby('Image Id'):
        defenders = identify_defending_skaters(frame, defending_team, home_team, away_team)
        
        if len(defenders) < 5:
            continue
        
        area, _ = calculate_convex_hull_area(defenders)
        frame_time = frame['Total_Seconds'].iloc[0]
        
        # Calculate distance from pre-pass positions for each player
        for _, player in defenders.iterrows():
            player_id = player['Player Id']
            if player_id in pre_pass_positions:
                ref_pos = pre_pass_positions[player_id]
                distance = np.sqrt(
                    (player['X'] - ref_pos['x'])**2 + 
                    (player['Y'] - ref_pos['y'])**2
                )
                
                if player_id not in coordination_lag:
                    coordination_lag[player_id] = {
                        'jersey': player['Player Jersey Number'],
                        'max_distance': distance,
                        'time_to_return': None
                    }
                else:
                    coordination_lag[player_id]['max_distance'] = max(
                        coordination_lag[player_id]['max_distance'],
                        distance
                    )
                    
                    # Check if player has returned to within 5 feet of pre-pass position
                    if distance < 5.0 and coordination_lag[player_id]['time_to_return'] is None:
                        coordination_lag[player_id]['time_to_return'] = frame_time - event_time
        
        # Check if area is within threshold
        if area <= stable_area:
            recovery_times.append(frame_time - event_time)
    
    recovery_time = min(recovery_times) if recovery_times else None
    
    return recovery_time, coordination_lag


def analyze_game(game_file: str) -> Dict:
    """
    Analyze a single game for defensive elasticity.
    
    This function coordinates the full analysis pipeline:
    1. Loads camera_orientations.csv for coordinate normalization
    2. Loads event and tracking data
    3. Identifies Royal Road passes
    4. Calculates defensive recovery times
    5. Generates visualizations
    """
    print(f"Analyzing {game_file}...")
    
    # Load camera orientations - CRITICAL for coordinate normalization
    # This ensures all coordinates are in a consistent reference frame
    # (defending net always at X = -100)
    camera_orientations = load_camera_orientations()
    
    # Extract game name from file
    # Pattern: {date}.{team1}.@.{team2}.Events.csv
    game_name = game_file.replace('.Events.csv', '').replace(' Events.csv', '')
    
    # Load event data
    events_df = load_event_data(game_file)
    if events_df.empty:
        print(f"  No event data found for {game_file}")
        return {}
    
    # Get home and away teams
    home_team = events_df['Home_Team'].iloc[0]
    away_team = events_df['Away_Team'].iloc[0]
    
    # Find Royal Road passes
    royal_road_passes = find_royal_road_passes(events_df)
    print(f"  Found {len(royal_road_passes)} Royal Road passes")
    
    if len(royal_road_passes) == 0:
        return {}
    
    # Get unique periods from events
    periods = sorted(events_df['Period'].unique())
    
    # Load tracking data
    tracking_df = load_tracking_data(game_name, periods)
    if tracking_df.empty:
        print(f"  No tracking data found for {game_name}")
        return {}
    
    print(f"  Loaded {len(tracking_df)} tracking frames")
    
    # Normalize coordinates
    tracking_df = normalize_coordinates(tracking_df, game_name, camera_orientations, is_tracking=True)
    events_df = normalize_coordinates(events_df, game_name, camera_orientations, is_tracking=False)
    
    # Analyze each Royal Road pass
    disruptive_plays = []
    
    for idx, pass_event in royal_road_passes.iterrows():
        pass_time = pass_event['Total_Seconds']
        pass_team = pass_event['Team']
        
        # Determine defending team
        defending_team = home_team if pass_team == away_team else away_team
        
        # Get pre-pass defensive state (1 second before pass)
        pre_pass_window = tracking_df[
            (tracking_df['Total_Seconds'] >= pass_time - 1.0) &
            (tracking_df['Total_Seconds'] < pass_time)
        ]
        
        if len(pre_pass_window) == 0:
            continue
        
        # Calculate average pre-pass area
        pre_pass_areas = []
        for image_id, frame in pre_pass_window.groupby('Image Id'):
            defenders = identify_defending_skaters(frame, defending_team, home_team, away_team)
            if len(defenders) == 5:
                area, _ = calculate_convex_hull_area(defenders)
                pre_pass_areas.append(area)
        
        if not pre_pass_areas:
            continue
        
        avg_pre_pass_area = np.mean(pre_pass_areas)
        
        # Measure recovery time and coordination lag
        recovery_time, coordination_lag = measure_recovery_time(
            tracking_df, pass_time, defending_team, home_team, away_team, avg_pre_pass_area
        )
        
        if recovery_time is not None:
            # Find player with longest time to return (coordination lag)
            lag_player = None
            lag_time = None
            if coordination_lag:
                for player_id, lag_info in coordination_lag.items():
                    if lag_info['time_to_return'] is not None:
                        if lag_time is None or lag_info['time_to_return'] > lag_time:
                            lag_time = lag_info['time_to_return']
                            lag_player = lag_info['jersey']
            
            disruptive_plays.append({
                'Game': game_name,
                'Period': pass_event['Period'],
                'Clock': pass_event['Clock'],
                'Pass_Team': pass_team,
                'Defending_Team': defending_team,
                'Passer': pass_event['Player_Id'],
                'Receiver': pass_event['Player_Id_2'],
                'X1': pass_event['X_Coordinate'],
                'Y1': pass_event['Y_Coordinate'],
                'X2': pass_event['X_Coordinate_2'],
                'Y2': pass_event['Y_Coordinate_2'],
                'Pre_Pass_Area': avg_pre_pass_area,
                'Recovery_Time': recovery_time,
                'Event_Time': pass_time,
                'Coordination_Lag_Player': lag_player,
                'Coordination_Lag_Time': lag_time
            })
    
    print(f"  Analyzed {len(disruptive_plays)} disruptive plays")
    
    return {
        'game_name': game_name,
        'home_team': home_team,
        'away_team': away_team,
        'disruptive_plays': disruptive_plays,
        'events_df': events_df,
        'tracking_df': tracking_df
    }


def get_rink_image_path():
    """Find the actual rink image (not the reference image)."""
    # First check if explicitly set
    if RINK_IMAGE_PATH and Path(RINK_IMAGE_PATH).exists():
        return Path(RINK_IMAGE_PATH)
    
    # Check common locations
    for candidate in RINK_IMAGE_CANDIDATES:
        if candidate.exists():
            return candidate
    
    return None


def load_rink_transformation():
    """Load rink image transformation parameters from calibration."""
    transformation_file = Path("/Users/emilyfehr8/CascadeProjects/rink_transformation.json")
    if transformation_file.exists():
        try:
            import json
            with open(transformation_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def draw_rink(fig, row=1, col=1):
    """Draw rink using background image and add faceoff dot markers for alignment."""
    # Get the actual rink image (not the reference)
    rink_image_path = get_rink_image_path()
    reference_image_path = DATA_DIR / "rink_coords.png"  # Reference for alignment only
    
    # Load transformation if available
    transformation = load_rink_transformation()
    
    # Add rink image as background
    if rink_image_path and rink_image_path.exists():
        # Use transformation to position image correctly
        # The image should cover the full coordinate range: X=-100 to 100, Y=-42.5 to 42.5
        # For Plotly, we position the image at the top-left corner and specify size
        fig.add_layout_image(
            dict(
                source=str(rink_image_path),
                xref=f"x{'' if row == 1 and col == 1 else f'{row}{col}'}",
                yref=f"y{'' if row == 1 and col == 1 else f'{row}{col}'}",
                x=-100,  # Left edge of coordinate system
                y=42.5,  # Top edge of coordinate system (Y increases downward in Plotly)
                sizex=200,  # Full width: 100 - (-100) = 200
                sizey=85,   # Full height: 42.5 - (-42.5) = 85
                sizing="stretch",
                opacity=0.85,
                layer="below"
            )
        )
        if transformation:
            print(f"    Using rink image with calibrated alignment: {rink_image_path}")
        else:
            print(f"    Using rink image (no calibration found): {rink_image_path}")
    elif reference_image_path.exists():
        # Fallback: if no rink image found, use reference for now (with warning)
        print(f"    WARNING: Using reference image {reference_image_path.name} as fallback")
        print(f"    Please set RINK_IMAGE_PATH to your actual rink image")
        fig.add_layout_image(
            dict(
                source=str(reference_image_path),
                xref=f"x{'' if row == 1 and col == 1 else f'{row}{col}'}",
                yref=f"y{'' if row == 1 and col == 1 else f'{row}{col}'}",
                x=-100,
                y=42.5,
                sizex=200,
                sizey=85,
                sizing="stretch",
                opacity=0.5,
                layer="below"
            )
        )
    else:
        # Fallback: draw basic rink if image not found
        fig.add_shape(
            type="rect",
            x0=-100, y0=-42.5, x1=100, y1=42.5,
            line=dict(color="#1a1a1a", width=4),
            fillcolor="#e8f4f8",
            opacity=0.3,
            layer="below",
            row=row, col=col
        )
    
    # Add coordinate system reference points
    # Use the actual faceoff dots that were marked for calibration
    if transformation:
        # Use the calibrated faceoff dots from the transformation
        calibrated_dots = [
            (-69, 22, "Faceoff (-69,22)"),   # Left side, top
            (-69, -22, "Faceoff (-69,-22)"), # Left side, bottom
            (69, 22, "Faceoff (69,22)"),     # Right side, top
            (69, -22, "Faceoff (69,-22)"),   # Right side, bottom
        ]
    else:
        # Fallback to standard NHL coordinates
        calibrated_dots = [
            (-69, 22, "Faceoff (-69,22)"),
            (-69, -22, "Faceoff (-69,-22)"),
            (69, 22, "Faceoff (69,22)"),
            (69, -22, "Faceoff (69,-22)"),
        ]
    
    # Add coordinate system grid lines for better visualization
    # Center lines
    fig.add_shape(
        type="line",
        x0=0, y0=-42.5, x1=0, y1=42.5,
        line=dict(color="#ffd700", width=2, dash="dash"),
        layer="below",
        row=row, col=col
    )
    fig.add_shape(
        type="line",
        x0=-100, y0=0, x1=100, y1=0,
        line=dict(color="#ffd700", width=2, dash="dash"),
        layer="below",
        row=row, col=col
    )
    
    # Add calibrated faceoff dots (toggleable)
    faceoff_x = [x for x, y, label in calibrated_dots]
    faceoff_y = [y for x, y, label in calibrated_dots]
    faceoff_text = [f"({x},{y})" for x, y, label in calibrated_dots]
    
    fig.add_trace(go.Scatter(
        x=faceoff_x,
        y=faceoff_y,
        mode='markers+text',
        marker=dict(
            size=12,
            color='#ffd700',
            line=dict(width=3, color='#000000'),
            symbol='circle'
        ),
        text=faceoff_text,
        textposition='top center',
        textfont=dict(size=10, color='#000000', family='Arial Black'),
        name='Calibrated Faceoff Dots',
        hovertemplate='<b>Calibrated Reference</b><br>X: %{x}, Y: %{y}<br>Click legend to toggle<extra></extra>',
        visible=True,  # Visible by default to verify alignment
        legendgroup='reference',
        showlegend=True
    ), row=row, col=col)
    
    # Add center ice origin marker (always visible)
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers+text',
        marker=dict(
            size=15,
            color='#ff0000',
            line=dict(width=3, color='#ffffff'),
            symbol='star'
        ),
        text=['ORIGIN (0,0)'],
        textposition='bottom center',
        textfont=dict(size=11, color='#ff0000', family='Arial Black', weight='bold'),
        name='Origin (0,0)',
        hovertemplate='<b>Origin (0,0)</b><br>Center Ice<br>Coordinate System Center<extra></extra>',
        showlegend=True,
        legendgroup='reference'
    ), row=row, col=col)
    
    # Add corner markers for full coordinate range
    corner_markers = [
        (-100, -42.5, "(-100,-42.5)"),
        (-100, 42.5, "(-100,42.5)"),
        (100, -42.5, "(100,-42.5)"),
        (100, 42.5, "(100,42.5)"),
    ]
    corner_x = [x for x, y, label in corner_markers]
    corner_y = [y for x, y, label in corner_markers]
    corner_text = [label for x, y, label in corner_markers]
    
    fig.add_trace(go.Scatter(
        x=corner_x,
        y=corner_y,
        mode='markers+text',
        marker=dict(
            size=8,
            color='#00ff00',
            line=dict(width=2, color='#000000'),
            symbol='square'
        ),
        text=corner_text,
        textposition='middle center',
        textfont=dict(size=8, color='#000000', family='Arial'),
        name='Coordinate Corners',
        hovertemplate='<b>Corner</b><br>%{text}<br>Full coordinate range<extra></extra>',
        visible='legendonly',  # Hidden by default
        legendgroup='reference',
        showlegend=True
    ), row=row, col=col)


def create_visualization(game_analysis: Dict, top_n: int = 3) -> None:
    """Create A+++++ level interactive Plotly visualization with animations."""
    if not game_analysis or 'disruptive_plays' not in game_analysis:
        return
    
    disruptive_plays = game_analysis['disruptive_plays']
    if len(disruptive_plays) == 0:
        return
    
    # Sort by recovery time (longest = most disruptive)
    disruptive_plays_sorted = sorted(
        disruptive_plays, 
        key=lambda x: x['Recovery_Time'], 
        reverse=True
    )[:top_n]
    
    game_name = game_analysis['game_name']
    tracking_df = game_analysis['tracking_df']
    home_team = game_analysis['home_team']
    away_team = game_analysis['away_team']
    
    for i, play in enumerate(disruptive_plays_sorted):
        event_time = play['Event_Time']
        defending_team = play['Defending_Team']
        
        # Get time window: 2 seconds before to 5 seconds after
        time_window = tracking_df[
            (tracking_df['Total_Seconds'] >= event_time - 2.0) &
            (tracking_df['Total_Seconds'] <= event_time + 5.0)
        ].copy()
        
        if len(time_window) == 0:
            continue
        
        # Use ALL available timestamps to show complete defensive recovery
        # This is critical for showing how the hull expands and contracts
        frame_times = sorted(time_window['Total_Seconds'].unique())
        sampled_times = frame_times  # Use all timestamps - no sampling
        
        print(f"    Generating visualization for play {i+1}: {len(sampled_times)} timestamps in time window")
        
        # Prepare animation frames
        frames_data = []
        stable_area = play['Pre_Pass_Area'] * (1 + RECOVERY_THRESHOLD)
        
        for frame_time in sampled_times:
            frame_data = time_window[time_window['Total_Seconds'] == frame_time]
            
            if len(frame_data) == 0:
                continue
            
            defenders = identify_defending_skaters(frame_data, defending_team, home_team, away_team)
            
            # Include frames even if we don't have exactly 5 defenders (for smoother animation)
            # This is critical for showing the recovery process - we need ALL frames
            if len(defenders) < 2:  # Reduced from 3 to 2 to include more frames
                continue
            
            # Get offensive players (attacking team)
            # Map team names to tracking data format (Home/Away)
            team_mapping = {
                home_team: 'Home',
                away_team: 'Away'
            }
            attacking_team = away_team if defending_team == home_team else home_team
            attacking_team_code = team_mapping.get(attacking_team, 'Away')
            
            attackers = frame_data[
                (frame_data['Team'] == attacking_team_code) & 
                (frame_data['Player or Puck'] == 'Player')
            ].copy()
            
            # Ensure attackers have X and Y columns (normalized coordinates)
            # The tracking_df should already have X and Y columns from load_tracking_data
            # But ensure they exist
            if len(attackers) > 0:
                if 'X' not in attackers.columns:
                    if 'Rink Location X (Feet)' in attackers.columns:
                        attackers['X'] = pd.to_numeric(attackers['Rink Location X (Feet)'], errors='coerce')
                    else:
                        attackers['X'] = 0
                if 'Y' not in attackers.columns:
                    if 'Rink Location Y (Feet)' in attackers.columns:
                        attackers['Y'] = pd.to_numeric(attackers['Rink Location Y (Feet)'], errors='coerce')
                    else:
                        attackers['Y'] = 0
                # Also ensure Player Jersey Number exists
                if 'Player Jersey Number' not in attackers.columns:
                    attackers['Player Jersey Number'] = attackers.get('Player Id', '?').astype(str)
                # Remove duplicates by Player Id
                attackers = attackers.drop_duplicates(subset=['Player Id'], keep='first')
            
            # Calculate convex hull
            area, centroid = calculate_convex_hull_area(defenders)
            is_stable = area <= stable_area
            
            # Get hull points
            points = defenders[['X', 'Y']].values
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
            except:
                x_min, x_max = points[:, 0].min(), points[:, 0].max()
                y_min, y_max = points[:, 1].min(), points[:, 1].max()
                hull_points = np.array([
                    [x_min, y_min], [x_max, y_min],
                    [x_max, y_max], [x_min, y_max], [x_min, y_min]
                ])
            
            frames_data.append({
                'time': frame_time,
                'defenders': defenders,
                'attackers': attackers,
                'hull_points': hull_points,
                'area': area,
                'is_stable': is_stable,
                'centroid': centroid,
                'time_from_event': frame_time - event_time
            })
        
        if len(frames_data) == 0:
            print(f"    Warning: No frames generated for play {i+1}, skipping visualization")
            continue
        
        # Interpolate between frames to create smoother animation showing recovery
        # This is critical for visualizing defensive elasticity - we need to show the gradual recovery
        if len(frames_data) > 1 and len(frames_data) < 30:  # Only interpolate if we have few frames
            interpolated_frames = []
            for idx in range(len(frames_data) - 1):
                current_frame = frames_data[idx]
                next_frame = frames_data[idx + 1]
                
                # Add the current frame
                interpolated_frames.append(current_frame)
                
                # Create 1-2 interpolated frames between current and next
                # This shows the gradual recovery process
                time_diff = next_frame['time'] - current_frame['time']
                num_interp = 2 if time_diff > 0.3 else 1 if time_diff > 0.15 else 0
                
                if num_interp > 0 and len(current_frame['defenders']) > 0 and len(next_frame['defenders']) > 0:
                    # Match defenders by Player Id for interpolation
                    curr_def_dict = {row['Player Id']: row for _, row in current_frame['defenders'].iterrows()}
                    next_def_dict = {row['Player Id']: row for _, row in next_frame['defenders'].iterrows()}
                    
                    for interp_step in range(1, num_interp + 1):
                        interp_factor = interp_step / (num_interp + 1)
                        interp_time = current_frame['time'] + (time_diff * interp_factor)
                        
                        # Interpolate defender positions
                        interp_defenders_list = []
                        for player_id in curr_def_dict.keys():
                            if player_id in next_def_dict:
                                curr_def = curr_def_dict[player_id]
                                next_def = next_def_dict[player_id]
                                interp_def = curr_def.copy()
                                interp_def['X'] = curr_def['X'] + (next_def['X'] - curr_def['X']) * interp_factor
                                interp_def['Y'] = curr_def['Y'] + (next_def['Y'] - curr_def['Y']) * interp_factor
                                interp_defenders_list.append(interp_def)
                        
                        if len(interp_defenders_list) >= 2:
                            interp_defenders = pd.DataFrame(interp_defenders_list)
                            
                            # Recalculate hull for interpolated positions
                            points = interp_defenders[['X', 'Y']].values
                            try:
                                hull = ConvexHull(points)
                                hull_points = points[hull.vertices]
                                hull_points = np.vstack([hull_points, hull_points[0]])
                            except:
                                # Fallback to linear interpolation of hull
                                curr_hull = current_frame['hull_points']
                                next_hull = next_frame['hull_points']
                                if len(curr_hull) == len(next_hull):
                                    hull_points = curr_hull + (next_hull - curr_hull) * interp_factor
                                else:
                                    hull_points = current_frame['hull_points']
                            
                            area, centroid = calculate_convex_hull_area(interp_defenders)
                            is_stable = area <= stable_area
                            
                            interpolated_frames.append({
                                'time': interp_time,
                                'defenders': interp_defenders,
                                'attackers': current_frame['attackers'],  # Keep same attackers
                                'hull_points': hull_points,
                                'area': area,
                                'is_stable': is_stable,
                                'centroid': centroid,
                                'time_from_event': interp_time - event_time
                            })
            
            # Add the last frame
            interpolated_frames.append(frames_data[-1])
            frames_data = interpolated_frames
        
        print(f"    Generated {len(frames_data)} animation frames for play {i+1} (showing defensive recovery)")
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Defensive Formation', 'Convex Hull Area Over Time', 
                          'Recovery Metrics', 'Player Positions'),
            specs=[[{"type": "scatter", "colspan": 2}, None],
                  [{"type": "scatter"}, {"type": "scatter"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Draw rink on main plot with background image
        draw_rink(fig, row=1, col=1)
        
        # Add pass line (always visible)
        fig.add_trace(go.Scatter(
            x=[play['X1'], play['X2']],
            y=[play['Y1'], play['Y2']],
            mode='lines+markers',
            line=dict(color='#9b59b6', width=4, dash='dashdot'),
            marker=dict(size=14, color='#9b59b6', symbol='star', line=dict(width=2, color='white')),
            name='Royal Road Pass',
            hovertemplate='<b>Royal Road Pass</b><br>From: (%.1f, %.1f)<br>To: (%.1f, %.1f)<extra></extra>' % 
                         (play['X1'], play['Y1'], play['X2'], play['Y2']),
            showlegend=True
        ), row=1, col=1)
        
        # Add initial frame (first frame visible)
        initial_frame = frames_data[0]
        hull_color = '#2ecc71' if initial_frame['is_stable'] else '#e74c3c'
        hull_opacity = 0.4 if initial_frame['is_stable'] else 0.6
        
        # Convex hull trace (will be animated)
        fig.add_trace(go.Scatter(
            x=initial_frame['hull_points'][:, 0],
            y=initial_frame['hull_points'][:, 1],
            mode='lines',
            line=dict(color=hull_color, width=3),
            fill='toself',
            fillcolor=hull_color,
            opacity=hull_opacity,
            name="Defensive Hull",
            hovertemplate='<b>Time: %.2fs</b><br>Area: %.1f ft²<br>Status: %s<extra></extra>' % 
                         (initial_frame['time_from_event'], initial_frame['area'], 
                          'Stable' if initial_frame['is_stable'] else 'Expanding'),
            showlegend=True
        ), row=1, col=1)
        
        # Defender positions trace (will be animated)
        defenders_df = initial_frame['defenders']
        fig.add_trace(go.Scatter(
            x=defenders_df['X'],
            y=defenders_df['Y'],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#3498db',
                line=dict(width=1.5, color='white'),
                symbol='circle'
            ),
            text=defenders_df['Player Jersey Number'].astype(str),
            textposition='middle center',
            textfont=dict(size=9, color='white', family='Arial Black'),
            name='Defenders',
            hovertemplate='<b>Player #%{text}</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>',
            showlegend=True
        ), row=1, col=1)
        
        # Attacker positions trace (will be animated) - Always add, even if empty
        attackers_df = initial_frame.get('attackers', pd.DataFrame())
        if len(attackers_df) > 0:
            fig.add_trace(go.Scatter(
                x=attackers_df['X'],
                y=attackers_df['Y'],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='#e74c3c',
                    line=dict(width=1.5, color='white'),
                    symbol='circle'
                ),
                text=attackers_df['Player Jersey Number'].astype(str),
                textposition='middle center',
                textfont=dict(size=9, color='white', family='Arial Black'),
                name='Attackers',
                hovertemplate='<b>Player #%{text}</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>',
                showlegend=True
            ), row=1, col=1)
        else:
            # Add empty trace so it exists for animation frames
            fig.add_trace(go.Scatter(
                x=[],
                y=[],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='#e74c3c',
                    line=dict(width=1.5, color='white'),
                    symbol='circle'
                ),
                text=[],
                name='Attackers',
                showlegend=True
            ), row=1, col=1)
        
        # Area over time plot
        times = [f['time_from_event'] for f in frames_data]
        areas = [f['area'] for f in frames_data]
        stable_areas = [stable_area] * len(times)
        
        fig.add_trace(go.Scatter(
            x=times,
            y=areas,
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8, color='#e74c3c'),
            name='Hull Area',
            hovertemplate='<b>Time: %{x:.2f}s</b><br>Area: %{y:.1f} ft²<extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=times,
            y=stable_areas,
            mode='lines',
            line=dict(color='#2ecc71', width=2, dash='dash'),
            name='Stable Threshold',
            hovertemplate='Stable Area: %{y:.1f} ft²<extra></extra>'
        ), row=2, col=1)
        
        # Recovery metrics
        recovery_time = play['Recovery_Time']
        pre_pass_area = play['Pre_Pass_Area']
        max_area = max(areas) if areas else pre_pass_area
        
        metrics_data = {
            'Metric': ['Recovery Time', 'Pre-Pass Area', 'Max Area', 'Area Increase'],
            'Value': [
                f'{recovery_time:.2f}s',
                f'{pre_pass_area:.1f} ft²',
                f'{max_area:.1f} ft²',
                f'{((max_area/pre_pass_area - 1)*100):.1f}%'
            ]
        }
        
        fig.add_trace(go.Bar(
            x=metrics_data['Metric'],
            y=[recovery_time, pre_pass_area/100, max_area/100, (max_area/pre_pass_area - 1)*100],
            marker=dict(
                color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
                line=dict(width=2, color='white')
            ),
            text=metrics_data['Value'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
            name='Metrics'
        ), row=2, col=2)
        
        # Player positions heatmap (final frame)
        final_frame = frames_data[-1]
        defenders_df = final_frame['defenders']
        
        fig.add_trace(go.Scatter(
            x=defenders_df['X'],
            y=defenders_df['Y'],
            mode='markers+text',
            marker=dict(
                size=25,
                color='#3498db',
                line=dict(width=3, color='white'),
                symbol='circle'
            ),
            text=defenders_df['Player Jersey Number'].astype(str),
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            name='Final Positions',
            hovertemplate='<b>Player #%{text}</b><br>Final Position: (%.1f, %.1f)<extra></extra>',
            showlegend=False
        ), row=2, col=2)
        
        # Create animation frames - update hull, defenders, and attackers traces
        animation_frames = []
        for frame_idx in range(len(frames_data)):
            frame = frames_data[frame_idx]
            
            frame_data_list = [
                # Hull trace
                go.Scatter(
                    x=frame['hull_points'][:, 0],
                    y=frame['hull_points'][:, 1],
                    mode='lines',
                    line=dict(color='#2ecc71' if frame['is_stable'] else '#e74c3c', width=3),
                    fill='toself',
                    fillcolor='#2ecc71' if frame['is_stable'] else '#e74c3c',
                    opacity=0.4 if frame['is_stable'] else 0.6
                ),
                # Defenders trace
                go.Scatter(
                    x=frame['defenders']['X'],
                    y=frame['defenders']['Y'],
                    mode='markers+text',
                    marker=dict(size=12, color='#3498db', line=dict(width=1.5, color='white')),
                    text=frame['defenders']['Player Jersey Number'].astype(str),
                    textposition='middle center',
                    textfont=dict(size=9, color='white', family='Arial Black')
                )
            ]
            
            # Always add attackers trace (even if empty) to maintain consistent trace indices
            attackers_df = frame.get('attackers', pd.DataFrame())
            if len(attackers_df) > 0:
                frame_data_list.append(
                    go.Scatter(
                        x=attackers_df['X'],
                        y=attackers_df['Y'],
                        mode='markers+text',
                        marker=dict(size=12, color='#e74c3c', line=dict(width=1.5, color='white')),
                        text=attackers_df['Player Jersey Number'].astype(str),
                        textposition='middle center',
                        textfont=dict(size=9, color='white', family='Arial Black')
                    )
                )
            else:
                # Add empty trace to maintain trace index consistency
                frame_data_list.append(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode='markers+text',
                        marker=dict(size=12, color='#e74c3c', line=dict(width=1.5, color='white')),
                        text=[],
                    )
                )
            
            # Determine which traces to update
            # Traces on row=1, col=1 (in order):
            #   0: Calibrated Faceoff Dots (from draw_rink)
            #   1: Origin (0,0) (from draw_rink)
            #   2: Coordinate Corners (from draw_rink)
            #   3: Royal Road Pass
            #   4: Defensive Hull
            #   5: Defenders
            #   6: Attackers (always present)
            trace_indices = [4, 5, 6]  # Hull (index 4), Defenders (index 5), Attackers (index 6)
            
            animation_frames.append(go.Frame(
                data=frame_data_list,
                name=str(frame_idx),
                traces=trace_indices
            ))
        
        fig.frames = animation_frames
        
        # Add play/pause button and slider
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': '⏸ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'x': 0.1,
                'y': 0,
                'xanchor': 'left',
                'yanchor': 'bottom'
            },
            {
                'type': 'buttons',
                'showactive': True,
                'buttons': [
                    {
                        'label': '1x',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 100}}]
                    },
                    {
                        'label': '0.5x',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 200}}]
                    },
                    {
                        'label': '2x',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50}}]
                    }
                ],
                'x': 0.3,
                'y': 0,
                'xanchor': 'left',
                'yanchor': 'bottom'
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 16, 'color': '#1a1a1a'},
                    'prefix': 'Time: ',
                    'suffix': 's',
                    'xanchor': 'right'
                },
                'transition': {'duration': 50},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [[str(i)], {
                            'frame': {'duration': 50, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 50}
                        }],
                        'label': f'{frames_data[i]["time_from_event"]:.1f}s',
                        'method': 'animate'
                    }
                    for i in range(0, len(frames_data), max(1, len(frames_data)//20))  # Sample for slider
                ]
            }]
        )
        
        # Add calibration info button/annotation
        calibration_info = (
            "<b>Coordinate System Reference</b><br>"
            "• <b>Origin (0,0)</b> = Center Ice (red star)<br>"
            "• <b>X-axis</b>: -100 (left) to +100 (right)<br>"
            "• <b>Y-axis</b>: -42.5 (bottom) to +42.5 (top)<br>"
            "• <b>Calibrated Faceoff Dots</b>: Gold circles (toggle in legend)<br>"
            "• <b>Royal Road</b>: Y = 0 (yellow dashed line)<br>"
            "• <b>Coordinate Corners</b>: Green squares (toggle in legend)"
        )
        
        # Update layout with modern styling
        fig.update_layout(
            title=dict(
                text=f'<b>Defensive Elasticity Analysis</b><br>' +
                     f'<span style="font-size: 14px; color: #7f8c8d">{game_name}</span><br>' +
                     f'<span style="font-size: 16px; color: #2c3e50">Most Disruptive Play #{i+1} | Recovery: {play["Recovery_Time"]:.2f}s | Period {play["Period"]} @ {play["Clock"]}</span>',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#2c3e50', family='Arial Black')
            ),
            height=900,
            width=1400,
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa',
            font=dict(family='Arial', size=12, color='#2c3e50'),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                font=dict(size=10)
            ),
            annotations=[
                dict(
                    text="<b>ℹ️ Coordinate System Reference</b><br>" + calibration_info,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.98)",
                    bordercolor="#3498db",
                    borderwidth=3,
                    borderpad=10,
                    font=dict(size=11, color='#2c3e50', family='Arial'),
                    showarrow=False,
                    align="left"
                ),
                dict(
                    text="<b>🎯 Legend Controls</b><br>" +
                         "• Toggle elements on/off<br>" +
                         "• Calibrated Faceoff Dots: Gold circles<br>" +
                         "• Coordinate Corners: Green squares<br>" +
                         "• Use to verify alignment",
                    xref="paper", yref="paper",
                    x=0.98, y=0.98,
                    xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.98)",
                    bordercolor="#2ecc71",
                    borderwidth=3,
                    borderpad=10,
                    font=dict(size=10, color='#2c3e50', family='Arial'),
                    showarrow=False,
                    align="right"
                )
            ]
        )
        
        # Update axes - make much wider to show full rink
        fig.update_xaxes(
            range=[-140, 140],
            title_text="X Position (feet)",
            showgrid=True,
            gridcolor='#ecf0f1',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='#bdc3c7',
            zerolinewidth=2,
            row=1, col=1
        )
        fig.update_yaxes(
            range=[-65, 65],
            title_text="Y Position (feet)",
            showgrid=True,
            gridcolor='#ecf0f1',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='#bdc3c7',
            zerolinewidth=2,
            row=1, col=1
        )
        
        fig.update_xaxes(
            title_text="Time from Pass (seconds)",
            showgrid=True,
            gridcolor='#ecf0f1',
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Area (ft²)",
            showgrid=True,
            gridcolor='#ecf0f1',
            row=2, col=1
        )
        
        fig.update_xaxes(
            title_text="Metrics",
            showgrid=False,
            row=2, col=2
        )
        fig.update_yaxes(
            title_text="Value",
            showgrid=True,
            gridcolor='#ecf0f1',
            row=2, col=2
        )
        
        # Save figure with enhanced HTML template
        output_file = OUTPUT_DIR / f"{game_name.replace(' ', '_')}_disruptive_play_{i+1}.html"
        
        # Custom HTML template with premium styling
        custom_template = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Defensive Elasticity Analysis - {title}</title>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                    min-height: 100vh;
                    color: #1a1a1a;
                    line-height: 1.6;
                    -webkit-font-smoothing: antialiased;
                    -moz-osx-font-smoothing: grayscale;
                }}
                
                .container {{
                    max-width: 1800px;
                    margin: 0 auto;
                    background: #ffffff;
                    border-radius: 24px;
                    box-shadow: 0 25px 80px rgba(0,0,0,0.4), 0 10px 30px rgba(0,0,0,0.3);
                    padding: 0;
                    overflow: hidden;
                    margin-top: 40px;
                    margin-bottom: 40px;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
                    color: #ffffff;
                    padding: 50px 50px 40px 50px;
                    position: relative;
                    overflow: hidden;
                }}
                
                .header::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse"><path d="M 100 0 L 0 0 0 100" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/></pattern></defs><rect width="100%" height="100%" fill="url(%23grid)"/></svg>');
                    opacity: 0.3;
                }}
                
                .header-content {{
                    position: relative;
                    z-index: 1;
                }}
                
                .header h1 {{
                    margin: 0 0 12px 0;
                    font-size: 42px;
                    font-weight: 800;
                    letter-spacing: -1.5px;
                    line-height: 1.1;
                    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
                }}
                
                .header .subtitle {{
                    margin-top: 8px;
                    font-size: 16px;
                    opacity: 0.95;
                    font-weight: 400;
                    letter-spacing: 0.3px;
                }}
                
                .header .badge {{
                    display: inline-block;
                    background: rgba(255,255,255,0.2);
                    padding: 6px 14px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-top: 15px;
                    backdrop-filter: blur(10px);
                }}
                
                .content-wrapper {{
                    padding: 50px;
                }}
                
                .metrics-bar {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                    gap: 24px;
                    margin-bottom: 40px;
                }}
                
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 32px 28px;
                    border-radius: 16px;
                    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }}
                
                .metric-card::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    right: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                    opacity: 0;
                    transition: opacity 0.3s ease;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-4px);
                    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
                }}
                
                .metric-card:hover::before {{
                    opacity: 1;
                }}
                
                .metric-card:nth-child(1) {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
                }}
                
                .metric-card:nth-child(2) {{
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    box-shadow: 0 8px 24px rgba(245, 87, 108, 0.3);
                }}
                
                .metric-card:nth-child(3) {{
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    box-shadow: 0 8px 24px rgba(79, 172, 254, 0.3);
                }}
                
                .metric-card:nth-child(4) {{
                    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                    box-shadow: 0 8px 24px rgba(67, 233, 123, 0.3);
                }}
                
                .metric-card h3 {{
                    margin: 0 0 12px 0;
                    font-size: 13px;
                    text-transform: uppercase;
                    letter-spacing: 1.5px;
                    opacity: 0.95;
                    font-weight: 600;
                    position: relative;
                    z-index: 1;
                }}
                
                .metric-card .value {{
                    font-size: 40px;
                    font-weight: 800;
                    margin: 0;
                    line-height: 1;
                    position: relative;
                    z-index: 1;
                    letter-spacing: -1px;
                }}
                
                .controls-info {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    border-left: 5px solid #667eea;
                    padding: 28px 32px;
                    margin-bottom: 32px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                }}
                
                .controls-info h4 {{
                    margin: 0 0 16px 0;
                    color: #1a1a1a;
                    font-size: 18px;
                    font-weight: 700;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                
                .controls-info h4::before {{
                    content: '🎮';
                    font-size: 24px;
                }}
                
                .controls-info ul {{
                    margin: 0;
                    padding-left: 24px;
                    color: #2c3e50;
                    font-size: 15px;
                    line-height: 1.8;
                }}
                
                .controls-info li {{
                    margin: 8px 0;
                    font-weight: 500;
                }}
                
                .controls-info li strong {{
                    color: #667eea;
                    font-weight: 700;
                }}
                
                .plot-container {{
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border-radius: 16px;
                    padding: 32px;
                    margin-top: 32px;
                    box-shadow: inset 0 2px 8px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.08);
                    border: 1px solid rgba(0,0,0,0.05);
                }}
                
                #plotly-div {{
                    width: 100% !important;
                    height: 900px !important;
                    border-radius: 12px;
                    overflow: hidden;
                    background: white;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
                }}
                
                .footer {{
                    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                    color: #ecf0f1;
                    margin-top: 50px;
                    padding: 40px 50px;
                    text-align: center;
                    border-top: 1px solid rgba(255,255,255,0.1);
                }}
                
                .footer p {{
                    margin: 8px 0;
                    font-size: 14px;
                    opacity: 0.9;
                    font-weight: 400;
                }}
                
                .footer p:first-child {{
                    font-size: 16px;
                    font-weight: 600;
                    opacity: 1;
                    margin-bottom: 12px;
                }}
                
                @media (max-width: 1024px) {{
                    .content-wrapper {{
                        padding: 30px;
                    }}
                    
                    .header {{
                        padding: 40px 30px 30px 30px;
                    }}
                    
                    .header h1 {{
                        font-size: 32px;
                    }}
                }}
                
                @media (max-width: 768px) {{
                    body {{
                        padding: 0;
                    }}
                    
                    .container {{
                        margin: 0;
                        border-radius: 0;
                        margin-top: 0;
                        margin-bottom: 0;
                    }}
                    
                    .content-wrapper {{
                        padding: 20px;
                    }}
                    
                    .header {{
                        padding: 30px 20px 25px 20px;
                    }}
                    
                    .header h1 {{
                        font-size: 28px;
                    }}
                    
                    .header .subtitle {{
                        font-size: 14px;
                    }}
                    
                    .metrics-bar {{
                        grid-template-columns: 1fr;
                        gap: 16px;
                    }}
                    
                    .metric-card {{
                        padding: 24px 20px;
                    }}
                    
                    .metric-card .value {{
                        font-size: 32px;
                    }}
                    
                    .controls-info {{
                        padding: 20px 24px;
                    }}
                    
                    .plot-container {{
                        padding: 20px;
                    }}
                    
                    #plotly-div {{
                        height: 700px !important;
                    }}
                }}
                
                /* Smooth scroll behavior */
                html {{
                    scroll-behavior: smooth;
                }}
                
                /* Selection styling */
                ::selection {{
                    background: #667eea;
                    color: white;
                }}
                
                ::-moz-selection {{
                    background: #667eea;
                    color: white;
                }}
            </style>
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 2px solid #ecf0f1;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 12px;
                }}
                .controls-info {{
                    background: #e8f4f8;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                }}
                .controls-info h4 {{
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                    font-size: 14px;
                }}
                .controls-info ul {{
                    margin: 0;
                    padding-left: 20px;
                    color: #34495e;
                    font-size: 13px;
                }}
                .controls-info li {{
                    margin: 5px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="header-content">
                        <h1>🏒 Defensive Elasticity Analysis</h1>
                        <div class="subtitle">{game_name}</div>
                        <div class="subtitle" style="margin-top: 4px; font-size: 15px; opacity: 0.85;">
                            Most Disruptive Play #{play_num} • Period {period} @ {clock}
                        </div>
                        <div class="badge">Big Data Cup 2026</div>
                    </div>
                </div>
                
                <div class="content-wrapper">
                    <div class="metrics-bar">
                        <div class="metric-card">
                            <h3>Recovery Time</h3>
                            <p class="value">{recovery_time:.2f}s</p>
                        </div>
                        <div class="metric-card">
                            <h3>Pre-Pass Area</h3>
                            <p class="value">{pre_pass_area:.0f} ft²</p>
                        </div>
                        <div class="metric-card">
                            <h3>Defending Team</h3>
                            <p class="value">{defending_team}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Pass Team</h3>
                            <p class="value">{pass_team}</p>
                        </div>
                    </div>
                    
                    <div class="controls-info">
                        <h4>Interactive Controls</h4>
                        <ul>
                            <li><strong>Play/Pause:</strong> Animate the defensive formation over time</li>
                            <li><strong>Speed Controls:</strong> Adjust playback speed (0.5x, 1x, 2x)</li>
                            <li><strong>Time Slider:</strong> Scrub through frames manually</li>
                            <li><strong>Legend:</strong> Toggle reference points and data layers</li>
                            <li><strong>Hover:</strong> Get detailed information about any element</li>
                            <li><strong>Zoom/Pan:</strong> Use toolbar buttons or mouse wheel</li>
                        </ul>
                    </div>
                    
                    <div class="plot-container">
                        {plotly_div}
                    </div>
                </div>
                
                <div class="footer">
                    <p>Big Data Cup 2026 • Defensive Elasticity Score Analysis</p>
                    <p>Generated with Plotly • Interactive visualization of defensive coordination</p>
                </div>
            </div>
            
            {plotly_script}
        </body>
        </html>
        """
        
        # Generate HTML with custom template
        import re
        
        # Get the plotly JSON
        plotly_json = fig.to_json()
        
        # Format custom HTML with embedded Plotly
        formatted_html = custom_template.format(
            title=f"{game_name} - Play {i+1}",
            game_name=game_name,
            play_num=i+1,
            period=play['Period'],
            clock=play['Clock'],
            recovery_time=play['Recovery_Time'],
            pre_pass_area=play['Pre_Pass_Area'],
            defending_team=play['Defending_Team'],
            pass_team=play['Pass_Team'],
            plotly_div='<div id="plotly-div" style="width:100%;height:900px;"></div>',
            plotly_script=f'''
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                var plotlyData = {plotly_json};
                var config = {{
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                    toImageButtonOptions: {{
                        format: 'png',
                        filename: 'defensive_elasticity_{game_name.replace(" ", "_")}_{i+1}',
                        height: 900,
                        width: 1400,
                        scale: 2
                    }},
                    responsive: true,
                    doubleClick: 'reset',
                    showTips: true
                }};
                Plotly.newPlot('plotly-div', plotlyData.data, plotlyData.layout, config);
                if (plotlyData.frames && plotlyData.frames.length > 0) {{
                    Plotly.addFrames('plotly-div', plotlyData.frames);
                }}
            </script>
            '''
        )
        
        # Write enhanced HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_html)
        
        print(f"  Saved enhanced visualization: {output_file}")


def generate_team_rankings(all_analyses: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate CSV ranking teams by elasticity score.
    Returns team rankings and detailed play-by-play data."""
    all_plays = []
    
    for analysis in all_analyses:
        if 'disruptive_plays' not in analysis:
            continue
        
        for play in analysis['disruptive_plays']:
            all_plays.append({
                'Game': play['Game'],
                'Period': play['Period'],
                'Clock': play['Clock'],
                'Defending_Team': play['Defending_Team'],
                'Pass_Team': play['Pass_Team'],
                'Recovery_Time': play['Recovery_Time'],
                'Pre_Pass_Area': play['Pre_Pass_Area'],
                'Coordination_Lag_Player': play.get('Coordination_Lag_Player'),
                'Coordination_Lag_Time': play.get('Coordination_Lag_Time')
            })
    
    if not all_plays:
        return pd.DataFrame(), pd.DataFrame()
    
    plays_df = pd.DataFrame(all_plays)
    
    # Calculate elasticity score for each team
    # Lower recovery time = higher elasticity (better)
    team_stats = plays_df.groupby('Defending_Team').agg({
        'Recovery_Time': ['mean', 'median', 'std', 'count'],
        'Pre_Pass_Area': 'mean',
        'Coordination_Lag_Time': 'mean'
    }).reset_index()
    
    team_stats.columns = ['Team', 'Avg_Recovery_Time', 'Median_Recovery_Time', 
                         'Std_Recovery_Time', 'Num_Disruptive_Plays', 'Avg_Pre_Pass_Area',
                         'Avg_Coordination_Lag_Time']
    
    # Calculate elasticity score (inverse of recovery time, normalized)
    # Higher score = better elasticity
    max_recovery = team_stats['Avg_Recovery_Time'].max()
    if max_recovery > 0:
        team_stats['Elasticity_Score'] = 100 * (1 - team_stats['Avg_Recovery_Time'] / max_recovery)
    else:
        team_stats['Elasticity_Score'] = 0
    
    # Sort by elasticity score (descending)
    team_stats = team_stats.sort_values('Elasticity_Score', ascending=False)
    
    return team_stats, plays_df


def main():
    """Main analysis function."""
    print("=" * 60)
    print("Defensive Elasticity Score Analysis")
    print("=" * 60)
    
    # Find all event files
    event_files = list(DATA_DIR.glob("*.Events.csv"))
    
    if not event_files:
        print("No event files found!")
        return
    
    print(f"Found {len(event_files)} games to analyze\n")
    
    # Analyze each game
    all_analyses = []
    
    for event_file in event_files:
        try:
            analysis = analyze_game(event_file.name)
            if analysis:
                all_analyses.append(analysis)
                
                # Create visualization for this game
                create_visualization(analysis, top_n=3)
        except Exception as e:
            print(f"  Error analyzing {event_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate team rankings
    print("\n" + "=" * 60)
    print("Generating Team Rankings...")
    print("=" * 60)
    
    team_rankings, detailed_plays = generate_team_rankings(all_analyses)
    
    if not team_rankings.empty:
        # Save team rankings
        output_file = OUTPUT_DIR / "team_elasticity_rankings.csv"
        team_rankings.to_csv(output_file, index=False)
        print(f"\nTeam rankings saved to: {output_file}")
        print("\nTop 5 Teams by Elasticity Score:")
        print(team_rankings.head().to_string(index=False))
        
        # Save detailed play-by-play data
        if not detailed_plays.empty:
            detailed_file = OUTPUT_DIR / "detailed_disruptive_plays.csv"
            detailed_plays.to_csv(detailed_file, index=False)
            print(f"\nDetailed play-by-play data saved to: {detailed_file}")
    else:
        print("No team rankings generated (insufficient data)")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

