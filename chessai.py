from inference import get_model
import supervision as sv
import cv2
import numpy as np
import pygame
import os
import chess
import chess.engine
import platform
import subprocess
import time
import mediapipe as mp  

import pyttsx3

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

pygame.init()

class HandGestureDetector:
    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.gesture_start_time = None
        self.okay_detected = False
        self.required_duration = 2.0  

        self.current_duration = 0.0

    def detect_okay_gesture(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z))

                okay_gesture = self.is_okay_gesture(landmarks)

                current_time = time.time()

                if okay_gesture:
                    if not self.okay_detected:

                        self.okay_detected = True
                        self.gesture_start_time = current_time
                    else:

                        self.current_duration = current_time - self.gesture_start_time
                else:

                    self.okay_detected = False
                    self.current_duration = 0.0

                if self.okay_detected:
                    self.draw_progress_bar(frame, self.current_duration)

                if self.current_duration >= self.required_duration:

                    self.okay_detected = False
                    self.current_duration = 0.0
                    return True
        else:

            self.okay_detected = False
            self.current_duration = 0.0

        return False

    def is_okay_gesture(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return False

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        distance = np.sqrt((thumb_tip[0] - index_tip[0]) ** 2 +
                           (thumb_tip[1] - index_tip[1]) ** 2)

        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        middle_extended = middle_tip[1] < middle_mcp[1]
        ring_extended = ring_tip[1] < ring_mcp[1]
        pinky_extended = pinky_tip[1] < pinky_mcp[1]

        return distance < 0.1 and middle_extended and ring_extended and pinky_extended

    def draw_progress_bar(self, frame, duration):

        progress = min(duration / self.required_duration, 1.0)

        bar_width = 200
        bar_height = 30
        margin = 20

        start_x = (frame.shape[1] - bar_width) // 2
        start_y = margin

        cv2.rectangle(frame,
                      (start_x, start_y),
                      (start_x + bar_width, start_y + bar_height),
                      (100, 100, 100),
                      -1)

        progress_width = int(bar_width * progress)
        cv2.rectangle(frame,
                      (start_x, start_y),
                      (start_x + progress_width, start_y + bar_height),
                      (0, 255, 0),
                      -1)

        cv2.rectangle(frame,
                      (start_x, start_y),
                      (start_x + bar_width, start_y + bar_height),
                      (255, 255, 255),
                      2)

        cv2.putText(frame,
                    f"Hold Okay: {int(duration)}/{int(self.required_duration)}s",
                    (start_x, start_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

class ChessDisplay:
    def __init__(self):

        self.SQUARE_SIZE = 60
        self.BOARD_SIZE = self.SQUARE_SIZE * 8
        self.screen = pygame.display.set_mode((self.BOARD_SIZE, self.BOARD_SIZE))
        pygame.display.set_caption("Chess Game")

        self.WHITE = (255, 255, 255)
        self.BLACK = (120, 120, 120)
        self.HIGHLIGHT = (255, 255, 0)
        self.WHITE_PIECE = (240, 240, 240)  

        self.BLACK_PIECE = (30, 30, 30)  

        self.tts_engine = pyttsx3.init()

        self.board = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]

        self.chess_board = chess.Board()

        self.white_turn = True

        self.engine = None
        self.initialize_engine()

        if self.white_turn and self.engine is not None:
            self.auto_make_stockfish_move()

    def initialize_engine(self):
        
        try:

            stockfish_path = self.find_stockfish()
            if stockfish_path:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"Stockfish initialized")
                return
            self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            self.engine = None

    def find_stockfish(self):
        
        if platform.system() == "Darwin":  

            possible_paths = [
                "/usr/local/bin/stockfish",
                "/opt/homebrew/bin/stockfish",
                os.path.expanduser("~/stockfish"),
                os.path.expanduser("~/stockfish/stockfish"),
                os.path.expanduser("~/Downloads/stockfish/stockfish"),
                "/Applications/stockfish/stockfish",
                "./stockfish",
                "./stockfish/stockfish"
            ]

            for path in possible_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    return path

            print("Stockfish not found")
            user_path = input("> ").strip()
            if user_path and os.path.isfile(user_path) and os.access(user_path, os.X_OK):
                return user_path

        return None

    def draw_board(self):
        self.screen.fill(self.WHITE)

        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    pygame.draw.rect(self.screen, self.BLACK,
                                     (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE,
                                      self.SQUARE_SIZE, self.SQUARE_SIZE))

        font = pygame.font.SysFont('Arial', 40)
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != '.':

                    color = self.BLACK_PIECE if piece.islower() else self.WHITE_PIECE
                    text = font.render(piece.upper(), True, color)
                    text_rect = text.get_rect(center=(col * self.SQUARE_SIZE + self.SQUARE_SIZE / 2,
                                                      row * self.SQUARE_SIZE + self.SQUARE_SIZE / 2))
                    self.screen.blit(text, text_rect)

        pygame.display.flip()

    def notation_to_coords(self, notation):
        
        col = ord(notation[0]) - ord('a')
        row = 8 - int(notation[1])
        return row, col

    def make_move(self, from_square, to_square):
        from_row, from_col = self.notation_to_coords(from_square)
        to_row, to_col = self.notation_to_coords(to_square)

        self.board[to_row][to_col] = self.board[from_row][from_col]
        self.board[from_row][from_col] = '.'

        try:
            move = chess.Move.from_uci(f"{from_square}{to_square}")
            if move in self.chess_board.legal_moves:
                self.chess_board.push(move)
                print(f"{from_square} -> {to_square}")

                self.white_turn = not self.white_turn

                if self.white_turn and self.engine is not None:

                    pygame.display.flip()
                    time.sleep(0.5)  

                    self.auto_make_stockfish_move()
            else:

                self.chess_board.push(move)
                self.white_turn = not self.white_turn

                if self.white_turn and self.engine is not None:
                    pygame.display.flip()
                    time.sleep(0.5)
                    self.auto_make_stockfish_move()
        except Exception as e:
            print(f"Error: {e}")

        self.draw_board()

    def get_stockfish_move(self):
        
        if self.engine is None:
            print("Stockfish not available")
            return None

        try:

            result = self.engine.play(self.chess_board, chess.engine.Limit(time=1.0))
            best_move = result.move

            from_square = chess.square_name(best_move.from_square)
            to_square = chess.square_name(best_move.to_square)

            move_str = f"{from_square} to {to_square}"
            uci_str = f"{from_square}{to_square}"

            print("\n" + "=" * 50)
            print(f"STOCKFISH: {move_str} (UCI: {uci_str})")
            print("=" * 50 + "\n")

            self.announce_move(from_square, to_square)

            return best_move

        except Exception as e:
            print(f"Error: {e}")
            return None

    def announce_move(self, from_square, to_square):
        
        try:

            announcement = f"Best move: {from_square} to {to_square}"

            self.tts_engine.say(announcement)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error: {e}")

    def close_engine(self):
        
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception as e:
                print(f"Error: {e}")

    def auto_make_stockfish_move(self):
        
        best_move = self.get_stockfish_move()

        if best_move:

            from_square = chess.square_name(best_move.from_square)
            to_square = chess.square_name(best_move.to_square)

            print(f"Stockfish: {from_square} → {to_square}")

            self.make_move(from_square, to_square)

    def close_engine(self):
        
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception as e:
                print(f"Error: {e}")

class ChessboardAnalyzer:
    def __init__(self, chess_display):
        self.square_size = 62
        self.reference_edges = None
        self.chess_display = chess_display
        self.reference_frame = None
        self.movement_buffer = {}
        self.buffer_frames = 3
        self.consistent_threshold = 2

        self.debug_mode = True

    def get_square_notation(self, row, col):
        files = 'abcdefgh'
        ranks = '87654321'
        return f"{files[col]}{ranks[row]}"

    def get_edge_signature(self, square):
        processed_square = self.preprocess_square(square)

        edges = cv2.Canny(processed_square, 100, 150)  

        edge_count = np.count_nonzero(edges)
        avg_intensity = np.mean(processed_square)

        hist = cv2.calcHist([processed_square], [0], None, [16], [0, 256])
        hist_variance = np.var(hist)

        if self.debug_mode:
            cv2.imshow(f"Processed Square", processed_square)
            cv2.imshow(f"Edges", edges)

        return edge_count, avg_intensity, hist_variance

    def preprocess_square(self, square):
        if len(square.shape) == 3:
            gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        else:
            gray = square.copy()

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))  

        enhanced = clahe.apply(gray)

        gamma = 1.2
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, lookup_table)

        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        result = cv2.addWeighted(denoised, 0.7, adaptive_thresh, 0.3, 0)

        return result

    def analyze_squares(self, warped_frame):
        squares = {}
        for row in range(8):
            for col in range(8):
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size

                if y2 > warped_frame.shape[0] or x2 > warped_frame.shape[1]:
                    continue

                square = warped_frame[y1:y2, x1:x2]

                if square.size == 0:
                    continue

                edge_count, avg_intensity, hist_variance = self.get_edge_signature(square)
                notation = self.get_square_notation(row, col)
                squares[notation] = {
                    'edge_count': edge_count,
                    'intensity': avg_intensity,
                    'hist_variance': hist_variance,
                    'diff_from_ref': 0
                }

        return squares

    def is_square_occupied(self, square_notation):

        files = 'abcdefgh'
        ranks = '87654321'

        col = files.index(square_notation[0])
        row = ranks.index(square_notation[1])

        return self.chess_display.board[row][col] != '.'

    def detect_movement(self, current_squares):
        if self.reference_edges is None:
            self.reference_edges = current_squares
            return

        changes = {}
        for square, current_info in current_squares.items():
            if square not in self.reference_edges:
                continue

            ref_info = self.reference_edges[square]

            edge_diff = abs(current_info['edge_count'] - ref_info['edge_count'])
            intensity_diff = abs(current_info['intensity'] - ref_info['intensity'])

            hist_diff = 0
            if 'hist_variance' in current_info and 'hist_variance' in ref_info:
                hist_diff = abs(current_info['hist_variance'] - ref_info['hist_variance'])

            edge_diffs = []
            intensity_diffs = []
            hist_diffs = []

            for s in current_squares:
                if s in self.reference_edges:
                    edge_diffs.append(abs(current_squares[s]['edge_count'] - self.reference_edges[s]['edge_count']))
                    intensity_diffs.append(abs(current_squares[s]['intensity'] - self.reference_edges[s]['intensity']))
                    if 'hist_variance' in current_squares[s] and 'hist_variance' in self.reference_edges[s]:
                        hist_diffs.append(
                            abs(current_squares[s]['hist_variance'] - self.reference_edges[s]['hist_variance']))

            avg_edge_diff = np.mean(edge_diffs) if edge_diffs else 1
            avg_intensity_diff = np.mean(intensity_diffs) if intensity_diffs else 1
            avg_hist_diff = np.mean(hist_diffs) if hist_diffs else 1

            edge_score = edge_diff / max(avg_edge_diff, 1)
            intensity_score = intensity_diff / max(avg_intensity_diff, 1)
            hist_score = hist_diff / max(avg_hist_diff, 1) if hist_diffs else 0

            change_score = edge_score * 0.5 + intensity_score * 0.3 + hist_score * 0.2

            if change_score > 1.8:
                changes[square] = {
                    'change_score': change_score,
                    'edge_diff': edge_diff,
                    'intensity_diff': intensity_diff,
                    'hist_diff': hist_diff,
                    'current_intensity': current_info['intensity'],
                    'ref_intensity': ref_info['intensity']
                }

        if changes:

            changed_squares = sorted(changes.items(), key=lambda x: x[1]['change_score'], reverse=True)

            valid_source_candidates = []
            invalid_source_candidates = []

            for square, change_info in changed_squares:

                if self.is_square_occupied(square) and change_info['current_intensity'] > change_info['ref_intensity']:
                    valid_source_candidates.append((square, change_info))
                elif change_info['current_intensity'] > change_info['ref_intensity']:

                    invalid_source_candidates.append((square, change_info))

            dest_candidates = []
            for square, change_info in changed_squares:
                if change_info['current_intensity'] < change_info['ref_intensity']:
                    dest_candidates.append((square, change_info))

            from_square = None
            to_square = None

            if 'e2' in changes and 'e4' in changes and self.is_square_occupied('e2'):

                if changes['e2']['current_intensity'] > changes['e2']['ref_intensity'] and \
                        changes['e4']['current_intensity'] < changes['e4']['ref_intensity']:
                    from_square = 'e2'
                    to_square = 'e4'

            if not from_square and valid_source_candidates and dest_candidates:

                from_square = valid_source_candidates[0][0]

                to_square = dest_candidates[0][0]

            elif not from_square:

                if invalid_source_candidates and dest_candidates:

                    for square, change_info in changed_squares:
                        if self.is_square_occupied(square) and square not in [d[0] for d in dest_candidates]:
                            from_square = square
                            to_square = dest_candidates[0][0]
                            print(f"Found alternative source: {from_square}")
                            break
                else:
                    print("No move found")

            if from_square and to_square:

                if self.is_square_occupied(from_square):
                    self.chess_display.make_move(from_square, to_square)
                else:
                    print(f"Source square empty")
            else:
                print("No move found")

        self.reference_edges = None

def sort_corners(corners):
    if len(corners) < 4:
        return corners

    corners = np.array(corners)

    center = np.mean(corners, axis=0)

    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])

    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]

    top_left = sorted_corners[np.argmin(np.sum(sorted_corners, axis=1))]
    bottom_right = sorted_corners[np.argmax(np.sum(sorted_corners, axis=1))]

    remaining = [p for p in sorted_corners if not np.array_equal(p, top_left) and not np.array_equal(p, bottom_right)]
    top_right = remaining[np.argmin(remaining[0][1] + remaining[1][1])]
    bottom_left = [p for p in remaining if not np.array_equal(p, top_right)][0]

    return np.array([top_left, top_right, bottom_right, bottom_left])

def warp_perspective(frame, corners, last_valid_warped=None):
    if len(corners) < 4:
        return last_valid_warped if last_valid_warped is not None else np.zeros((500, 500, 3), dtype=np.uint8)

    corners = sort_corners(corners)

    side_length = np.mean([
        np.linalg.norm(corners[0] - corners[1]),  

        np.linalg.norm(corners[1] - corners[2]),  

        np.linalg.norm(corners[2] - corners[3]),  

        np.linalg.norm(corners[3] - corners[0])  

    ])

    pts1 = np.float32(corners)
    pts2 = np.float32([
        [0, 0],  

        [side_length, 0],  

        [side_length, side_length],  

        [0, side_length]  

    ])

    edges = np.diff(corners, axis=0, append=corners[0:1])
    edge_lengths = np.sqrt(np.sum(edges ** 2, axis=1))
    length_ratios = edge_lengths / np.mean(edge_lengths)

    if np.any(length_ratios > 1.5) or np.any(length_ratios < 0.5):
        return last_valid_warped if last_valid_warped is not None else np.zeros((500, 500, 3), dtype=np.uint8)

    try:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped_frame = cv2.warpPerspective(frame, matrix, (500, 500))

        warped_frame = cv2.GaussianBlur(warped_frame, (3, 3), 0)

        return warped_frame
    except cv2.error:
        return last_valid_warped if last_valid_warped is not None else np.zeros((500, 500, 3), dtype=np.uint8)

def stabilize_corners(current_corners, previous_corners, alpha=0.7):
    
    if len(current_corners) < 4 or previous_corners is None or len(previous_corners) < 4:
        return current_corners

    current_sorted = sort_corners(current_corners)
    previous_sorted = sort_corners(previous_corners)

    stabilized = alpha * previous_sorted + (1 - alpha) * current_sorted

    return stabilized.astype(np.int32)

def main():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    chess_display = ChessDisplay()
    analyzer = ChessboardAnalyzer(chess_display)
    hand_detector = HandGestureDetector()  

    try:
        model = get_model(model_id="chessboard-detection-yqcnu/3", api_key="APIKEY")
    except Exception as e:
        print(f"Error: {e}")
        return

    bounding_box_annotator = sv.BoxAnnotator(color=sv.Color(128, 0, 128))  

    last_valid_warped = None

    chess_display.draw_board()

    debug_mode = False

    try:
        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)

            corners = []
            for box in detections.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                corners.append([cx, cy])

            annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)

            warped_frame = warp_perspective(frame, corners, last_valid_warped)

            if len(corners) >= 4:
                last_valid_warped = warped_frame.copy()

            edges_frame = cv2.Canny(warped_frame, 140, 150)  

            if debug_mode and last_valid_warped is not None:

                squares_grid = last_valid_warped.copy()
                for r in range(8):
                    for c in range(8):
                        y1 = r * analyzer.square_size
                        x1 = c * analyzer.square_size
                        y2 = y1 + analyzer.square_size
                        x2 = x1 + analyzer.square_size

                        if y2 <= squares_grid.shape[0] and x2 <= squares_grid.shape[1]:

                            color = (0, 255, 0)  

                            cv2.rectangle(squares_grid, (x1, y1), (x2, y2), color, 1)

                            notation = analyzer.get_square_notation(r, c)
                            cv2.putText(squares_grid, notation, (x1 + 5, y1 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                cv2.imshow("Chess Grid", squares_grid)

            okay_gesture_triggered = hand_detector.detect_okay_gesture(annotated_frame)

            if okay_gesture_triggered:
                if last_valid_warped is not None:
                    current_squares = analyzer.analyze_squares(last_valid_warped)
                    analyzer.detect_movement(current_squares)
                else:
                    print("No valid board.")

            cv2.imshow("YOLOv8", annotated_frame)
            cv2.imshow("Edge Detection", edges_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if last_valid_warped is not None:
                    current_squares = analyzer.analyze_squares(last_valid_warped)
                    analyzer.detect_movement(current_squares)
                else:
                    print("No board detected")
            elif key == ord('r'):
                chess_display.chess_board = chess.Board()
                chess_display.board = [
                    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
                    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
                    ['.', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '.'],
                    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
                ]
                chess_display.white_turn = True
                chess_display.draw_board()

                if chess_display.engine is not None:
                    time.sleep(0.5)  

                    chess_display.auto_make_stockfish_move()
            elif key == ord('d'):

                debug_mode = not debug_mode
                analyzer.debug_mode = debug_mode
                if not debug_mode:
                    cv2.destroyWindow("Chess Grid")
                    cv2.destroyWindow("Processed Square")
                    cv2.destroyWindow("Edges")

    except KeyboardInterrupt:
        print("Exiting")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(chess_display, 'engine') and chess_display.engine is not None:
            chess_display.close_engine()
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()