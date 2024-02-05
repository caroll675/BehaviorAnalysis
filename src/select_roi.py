import cv2
import numpy as np


def select_ROI(video_path):
    # Define a callback function for mouse events
    def select_roi(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['points'].append([(x, y)])
            param['dragging'] = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if param['dragging']:
                param['points'][-1].append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            param['dragging'] = False
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, frame = cap.read()
    # Initialize the ROI parameters
    roi_params = {'points': [], 'dragging': False}
    # Create a window to display the frame
    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    # Set the callback function for mouse events
    cv2.setMouseCallback('Select ROI', select_roi, roi_params)
    # Loop over the frames in the video
    while True:
        # Copy the frame
        roi_frame = frame.copy()
        # Draw the polygons on the ROI frame if the user is selecting regions
        if len(roi_params['points']) > 0:
            for points in roi_params['points']:
                points = np.array(points, np.int32)
                cv2.fillPoly(roi_frame, [points], (0, 255, 0))
        cv2.imshow('Select ROI', roi_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    # Release the video capture object and destroy the window
    cap.release()
    cv2.destroyAllWindows()

    # Combine the points into a single array and delete duplicates
    final_points = np.array(roi_params['points'][0], np.int32)
    for points in roi_params['points'][1:]:
        final_points = np.append(final_points, points, axis=0)

    return final_points

def is_in_roi(x, y, roi_arr):
    if cv2.pointPolygonTest(roi_arr, (x, y), False) >= 0:
        return True
    else:
        return False



def select_circular_ROI(video_path):
    # Define a callback function for mouse events
    def select_roi(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['center'] = (x, y)
            param['dragging'] = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if param['dragging']:
                radius = int(np.sqrt((x - param['center'][0]) ** 2 + (y - param['center'][1]) ** 2))
                param['radius'] = radius
        elif event == cv2.EVENT_LBUTTONUP:
            param['dragging'] = False

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, frame = cap.read()
    # Initialize the ROI parameters
    roi_params = {'center': None, 'radius': 0, 'dragging': False}
    # Create a window to display the frame
    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    # Set the callback function for mouse events
    cv2.setMouseCallback('Select ROI', select_roi, roi_params)
    # Loop over the frames in the video
    while True:
        # Copy the frame
        roi_frame = frame.copy()
        # Draw the circle on the ROI frame if the user is selecting a region
        if roi_params['center'] is not None:
            center = roi_params['center']
            radius = roi_params['radius']
            cv2.circle(roi_frame, center, radius, (0, 255, 0), 2)
        cv2.imshow('Select ROI', roi_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # Return the center and radius of the selected circle
    if roi_params['center'] is not None:
        center = roi_params['center']
        radius = roi_params['radius']
        return center, radius
    else:
        return None, 0


def is_in_circular_roi(x, y, center, radius):
    if np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius:
        return True
    else:
        return False
    


def select_rect_roi(video_path):
    def select_roi(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['active_corner'] = None
            for corner in param['corners']:
                if np.linalg.norm(np.array([x,y]) - np.array(corner)) < param['corner_radius']:
                    param['active_corner'] = corner
                    break
            if param['active_corner'] is None:
                param['corners'].append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE:
            if param['active_corner'] is not None:
                param['active_corner'][0] = x
                param['active_corner'][1] = y
        elif event == cv2.EVENT_LBUTTONUP:
            param['active_corner'] = None

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, frame = cap.read()
    # Initialize the ROI parameters
    roi_params = {'corners': [], 'active_corner': None, 'corner_radius': 5}
    # Create a window to display the frame
    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    # Set the callback function for mouse events
    cv2.setMouseCallback('Select ROI', select_roi, roi_params)
    # Loop over the frames in the video
    while True:
        # Copy the frame
        roi_frame = frame.copy()
        # Draw the ROI polygon on the frame
        if len(roi_params['corners']) > 0:
            roi_corners = np.array(roi_params['corners'])
            cv2.polylines(roi_frame, [roi_corners], True, (0, 255, 0), 2)
            for corner in roi_params['corners']:
                cv2.circle(roi_frame, tuple(corner), roi_params['corner_radius'], (0, 0, 255), -1)
        cv2.imshow('Select ROI', roi_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # Return the corners of the selected ROI
    if len(roi_params['corners']) > 0:
        print(roi_params['corners'])
        return np.array(roi_params['corners'])
    else:
        return None

def is_in_rect_roi(x, y, roi_arr):
    if cv2.pointPolygonTest(roi_arr, (x, y), False) >= 0:
        return True
    else:
        return False
    
